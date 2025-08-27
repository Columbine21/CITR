import os
import sys
import torch

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import logging
import argparse
import torch.multiprocessing as mp

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from typing import  Any, Dict, List
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

def _set_logger(args, verbose_level):
    # base logger
    log_file_path = Path(args.checkpoint_dir) / f"training.log"
    logger = logging.getLogger('MC_TEST') 
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--model_dir', type=str, default='/home/yuanziqi/Work25/Locret-VL/resources/models/Qwen2-VL-7B-Instruct', 
                        choices=['/home/yuanziqi/Work25/Locret-VL/resources/models/Qwen2-VL-7B-Instruct'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./output_long_7b')
    parser.add_argument('--evict_vtoken_only', type=bool, default=True)
    parser.add_argument('--smooth_loss_weight', type=float, default=0.0)
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_7b_evict_vtoken_only_3k')
    args = parser.parse_args()
    
    if 'qwen2-vl-7b' in args.model_dir.lower():
        kv_head_num = 4
    elif 'qwen2-vl-72b' in args.model_dir.lower():
        kv_head_num = 8
    else:
        raise "Model type not supported yet!"

    train_config = {
        "lr": args.lr,
        "bs": args.bs,
        "gradient_accumulation_steps": 1,
        "datapath": f"{args.data_dir}",
        "is_warmup": True,
        "num_epochs": 1,
        "num_warmup_steps": 1000,
        "total_steps": 8000,
        "num_workers": 1,
        "max_len": 32768,
        "grad_clip": 0.5,
        "b1": 0.9,
        "b2": 0.95,
        "save_freq": 10000,
        "kv_head_num": kv_head_num,
    }
    return args, train_config


class CustomDataset(Dataset):
    def __init__(self, datapath, max_len):
        self.data = datapath
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        assert data["input_ids"].shape[0] < self.max_len

        new_data = {}
        input_ids = data['input_ids'][:self.max_len][None, :]
        weights = data['weights'][:, 0, :, :self.max_len][None, :]
        new_data["input_ids"] = input_ids
        new_data["weights"] = weights
        new_data["pixel_values_videos"] =  data["pixel_values_videos"]
        new_data["video_grid_thw"] = data["video_grid_thw"]
        new_data["loss_mask"] = [1] * input_ids.shape[1]

        return new_data


class DataCollatorWithPadding:
    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) == 1
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "weights": features[0]['weights'],
            "pixel_values_videos": features[0]["pixel_values_videos"],
            "video_grid_thw": features[0]["video_grid_thw"],
            "loss_mask": batch_loss_mask,
        }
        return batch


if __name__ == '__main__':
   
    mp.set_start_method('spawn')
    args, train_config = parse_args()
    assert args.evict_vtoken_only, "currently support for visual token eviction."
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

    if accelerator.is_main_process:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

    logger = _set_logger(args, 1)
    logger.info(args)
    logger.info(train_config)
    # setup tensorboard
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'training_logs'))
    # setup seeds and accelerator
    set_seed(0)
    torch.manual_seed(0)
    
    criterion = nn.SmoothL1Loss(reduction='none')
    mse_loss = nn.MSELoss()

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.bfloat16)

    # scan the datapath folder.
    datapath = []
    for root, directories, files in os.walk(train_config["datapath"]):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)

    traindatapath = datapath[:train_config["total_steps"]]
    traindataset = CustomDataset(traindatapath, train_config["max_len"])
    train_loader = DataLoader(
        traindataset, 
        batch_size=train_config["bs"], 
        shuffle=False,
        collate_fn=DataCollatorWithPadding(), 
        num_workers=train_config["num_workers"]
    )

    
    for param in model.parameters():
        param.requires_grad = False

    for n, param in model.named_parameters():
        if "retaining_head" in n:
            param.requires_grad = True
            
    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]

    if is_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


    model.train()
    length_list, tot_step = [], 0
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate((train_loader)):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
            
                output= model(
                    input_ids=data['input_ids'], 
                    pixel_values_videos=data['pixel_values_videos'], 
                    video_grid_thw=data['video_grid_thw'],
                    use_cache=True, 
                    output_attentions=False,
                    plain_attn=False,
                    visual_sliding_window=True
                )

                data['weights'] = data['weights'].to("cuda:0")
                data_weights = (data['weights'])
                if args.evict_vtoken_only:
                    v_token_mask = data['input_ids'] == model.config.video_token_id
                    std_v_token_mask = data['input_ids'] == model.config.video_token_id
                    
                scores = output.casual_importance_score

                tot_loss = 0
                tot_ent = 0
                sparsity = 0.1

                corr, corr_q = 0, 0
                tot_cnt, tot_cnt_q = 0, 0

                smooth_loss = 0
                for i, s in enumerate(scores):
                    s = s[0].transpose(0, 1)
                    std_s = data_weights[0, i]

                    if args.evict_vtoken_only: # Only use the video_token_id part for the training target.
                        s = s[:, v_token_mask[0]]
                        std_s = std_s[:, std_v_token_mask[0]]
                    # s = s[..., :std_s.shape[-1], :] # 
                    _k = int(sparsity * s.shape[-1])
                    top_s = torch.topk(s, k=_k, dim=-1).indices
                    top_std = torch.topk(std_s, k=_k, dim=-1).indices
                    for j in range(train_config['kv_head_num']):
                        overlap = len(
                            set(top_s[j].tolist()) & set(top_std[j].tolist())
                        )
                        corr += overlap
                        writer.add_scalar(f'Training/Acc(top 10%)_{i}_{j}', overlap / _k, tot_step)
                        tot_cnt += _k

                    _k = int(0.25 * s.shape[-1])
                    top_s = torch.topk(s, k=_k, dim=-1).indices
                    top_std = torch.topk(std_s, k=_k, dim=-1).indices
                    for j in range(train_config['kv_head_num']):
                        overlap = len(
                            set(top_s[j].tolist()) & set(top_std[j].tolist())
                        )
                        corr_q += overlap
                        # detail_acc_list[i].append(overlap / _k)
                        # writer.add_scalar(f'Training/Acc(top 10%)_{i}_{j}', overlap / _k, tot_step)
                        tot_cnt_q += _k

                    loss = criterion(s.to(std_s.device), std_s.float())
                    smooth_loss += mse_loss(s[:, :-1], s[:, 1:]) 
                    loss = loss.mean(dim=-1)
                    tot_loss += loss.sum()
                    tot_ent += loss.shape[-1]
                
                loss = tot_loss / tot_ent
                logger.info(f"epoch: {epoch:4d} | step: {batch_idx:4d} | loss: {loss:.10f} | smooth_loss: {smooth_loss:.10f} | acc10%: {(corr / tot_cnt): .4f} | acc25%: {(corr_q/ tot_cnt_q): .4f} | len: {data['input_ids'].shape[-1]} | tot_step: {tot_step}")
                # record the loss, smooth_loss and acc using tensorboard
                writer.add_scalar('Training/Loss', loss, tot_step)
                writer.add_scalar('Training/Smooth_Loss', smooth_loss, tot_step)
                writer.add_scalar('Training/Acc(top 10%)', corr / tot_cnt, tot_step)
                writer.add_scalar('Training/Acc(top 25%)', corr_q / tot_cnt_q, tot_step)


                loss += smooth_loss.to(loss.device) * args.smooth_loss_weight
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
                optimizer.step()
                if is_warmup:
                    scheduler.step()
            tot_step += 1
            if accelerator.is_local_main_process and tot_step % train_config['save_freq'] == 0:
                accelerator.save_state(output_dir=f"{args.checkpoint_dir}/state_{epoch}_{tot_step}")
                
    model.save_pretrained(f"{args.checkpoint_dir}/final_model")
    writer.close()
