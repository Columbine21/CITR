import os
import sys
import argparse
import pickle
import torch
import copy
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

current_path = Path(__file__).resolve()
sys.path.append(str(current_path.parent.parent.parent.parent))

from transformers.cache_utils import DynamicCache
from transformers import AutoProcessor, AutoTokenizer
from src.eval.nextqa.utils import parse_multi_choice_response
from src.model.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

DATA_MAP = {
    'nextqa_mc_200k_1_mt': "/home/yuanziqi/Work25/Locret-VL/locret_vl/benchmark/nextqa/output/72b/nextqa_200704_1.0_mt.pkl",
}

def _set_logger(args, verbose_level):

    # base logger
    if "7b" in args.tokenizer_dir.lower():
        model_name = args.model + '-7b' + '-citr' if args.citr else args.model + '-7b'
    else:
        model_name = args.model + '-72b' + '-citr' if args.citr else args.model + '-72b' 
    local_len =  local_len if args.local_len != -1 else "auto"
    if args.citr:
        suffix = "-evict_video_only" if args.evict_video_pad_only else ""
        log_file_path = Path(args.log_dir) / f"{model_name}_{args.dataset}_{args.budget_size}-{local_len}-{args.chunk_size}-{args.stabilizers_size}{suffix}.log"
    else:
        log_file_path = Path(args.log_dir) / f"{model_name}_{args.dataset}.log"
    
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger('MT-TEST') 
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
    parser = argparse.ArgumentParser(description='inference test')
    parser.add_argument('--citr', type=bool, default=True, help='Whether to use citr inference.')
    parser.add_argument('--model', type=str, default="qwen2-vl-long", choices=["qwen2-vl-long", "qwen2-vl-ori"])
    parser.add_argument('--dataset', type=str, default='nextqa_mc_200k_1_mt', choices=['nextqa_mc_200k_1_mt'])

    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])

    parser.add_argument('--model_dir', type=str, default='checkpoints_7b_evict_vtoken_only_3k/final_model', help='The directory of model')
    parser.add_argument('--tokenizer_dir', type=str, default='Qwen2-VL-7B-Instruct', help='The directory of model')
    
    # generation hyperparameters.
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--sample', type=bool, default=False)

    # hyperparameters.
    parser.add_argument('--budget_size', type=int, default=512)
    parser.add_argument('--local_len', type=int, default=-1) # -1 for auto local_len (last user instruction is maintained without eviction.)
    parser.add_argument('--chunk_size', type=int, default=4096)
    parser.add_argument('--stabilizers_size', type=int, default=128)
    parser.add_argument('--evict_video_pad_only', type=bool, default=True)

    parser.add_argument('--log_dir', type=str, default="./logdir")

    args = parser.parse_args()
    return args

def deepcopy_kvcache(past_key_values, device):
    past_key_values_copy = []
    for i in range(len(past_key_values)):
        k = copy.deepcopy(past_key_values[i][0].detach().to(device[i]))
        v = copy.deepcopy(past_key_values[i][1].detach().to(device[i]))
        past_key_values_copy.append((k,v))
    
    return DynamicCache.from_legacy_cache(past_key_values_copy)

@torch.no_grad()
def prefilling(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,

) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    
    inputs_embeds = self.model.embed_tokens(input_ids)
    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
    video_embeds = self.visual.forward_sliding_window(pixel_values_videos, grid_thw=video_grid_thw)

    video_mask = (
        (input_ids == self.config.video_token_id)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    position_ids, rope_deltas = self.get_rope_index(
        input_ids, None, video_grid_thw, attention_mask
    )
    self.rope_deltas = rope_deltas

    seq_len = input_ids.shape[-1]
    past_key_values = None
    scores = [None for _ in range(100)]

    cache_position = torch.ones([seq_len], dtype=torch.int64).cumsum(0) - 1
    output = self.model(
        input_ids=None,
        position_ids=position_ids,
        # attention_mask=attention_mask[..., :e] if attention_mask is not None else None,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=False,
        return_dict=True,
        cache_position=cache_position.to(inputs_embeds.device),
        plain_attn=False,
    )
    past_key_values = output.past_key_values

    pruned_kv_cache = []
    for j in range(self.model.config.num_hidden_layers):
        
        k = past_key_values[j][0]
        v = past_key_values[j][1]

        pruned_kv_cache.append((k, v))
    past_key_values = DynamicCache.from_legacy_cache(pruned_kv_cache)

    del pruned_kv_cache, output
    torch.cuda.empty_cache()

    return past_key_values, position_ids

def generate_with_q(
    self,
    input_ids: torch.LongTensor = None,
    prefilling_kv_cache: DynamicCache = None,
    prefilling_position_ids: Optional[torch.LongTensor] = None,
    eos_token_id: int = 151645,
    max_new_tokens: int = 128,
):
    inputs_embeds = self.model.embed_tokens(input_ids)

    cache_position = torch.ones([inputs_embeds.shape[-2]], dtype=torch.int64).cumsum(0).cuda() - 1
    cache_position += prefilling_kv_cache.get_seq_length()
    position_ids = (torch.ones([inputs_embeds.shape[-2]], dtype=torch.int64).cumsum(0).cuda() + torch.max(prefilling_position_ids[0])).unsqueeze(0).unsqueeze(1).expand(3, -1, -1)
    output = self.model(
            position_ids=position_ids,
            past_key_values=prefilling_kv_cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            cache_position=cache_position.to(inputs_embeds.device),
            plain_attn=False,
        )
    del prefilling_kv_cache
    past_key_values = output.past_key_values
    
    logits = self.lm_head(output[0])
    input_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [input_tokens.item()]

    cache_position = torch.LongTensor([past_key_values.get_seq_length()])
    position_ids = torch.max(position_ids[0] + 1).view(-1).unsqueeze(0).unsqueeze(1).expand(3, -1, -1)

    for i in range(max_new_tokens - 1):
        
        output = self.model(
            input_ids=input_tokens, 
            position_ids=position_ids.cuda(),
            past_key_values=past_key_values,
            cache_position=cache_position.cuda(),
            use_cache=True,
            return_dict=True,
        )
        
        logits = self.lm_head(output[0])
        input_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(input_tokens.item())

        if input_tokens.item() == eos_token_id:
            break
        
        cache_position += 1
        position_ids = position_ids + 1
        past_key_values = output.past_key_values

    generated_tokens = torch.tensor(generated_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    input_ids = torch.cat((input_ids, generated_tokens), dim=-1)
    return input_ids
    

if __name__ == "__main__":

    args = parse_args()
    logger = _set_logger(args, 1)

    logger.info(args)

    if args.model == "qwen2-vl-long" or args.citr:
        from src.model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    else:
        assert args.model_dir == args.tokenizer_dir, "If not citr mode, model_dir and tokenizer_dir should be the same."
        from transformers import Qwen2VLForConditionalGeneration
    Qwen2VLForConditionalGeneration.prefilling = prefilling
    Qwen2VLForConditionalGeneration.generate_with_q = generate_with_q

    with open(DATA_MAP[args.dataset], "rb") as f:
        data = pickle.load(f)

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype="auto", device_map="auto", attn_implementation=args.attn_implementation)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    generation_suffix = tokenizer("<|im_start|>assistant\n", return_tensors="pt").input_ids

    answer_list, label_list, length_list = [], [], []
    metrics = {'exact_match': {"overall": []}}

    if not args.citr:
        if args.model == "qwen2-vl-long":
            kwargs = {"visual_sliding_window": True, "output_attentions": False, "plain_attn": False,}
        elif args.model == "qwen2-vl-ori":
            kwargs = {}
        for i, ins in enumerate(data):
            prefilling_kv_cache, prefilling_position_ids = model.prefilling(
                input_ids=ins['input_ids'].cuda(),
                attention_mask=torch.ones_like(ins['input_ids']).cuda(),
                use_cache=True,
                pixel_values_videos=ins['pixel_values_videos'].cuda(),
                video_grid_thw=ins['video_grid_thw'].cuda()
            )
            device = []
            # offload the prefilling_kv_cache.
            for layer_idx in range(model.config.num_hidden_layers):
        
                device.append(prefilling_kv_cache.key_cache[layer_idx].device)

                prefilling_kv_cache.key_cache[layer_idx] = prefilling_kv_cache.key_cache[layer_idx].cpu()
                prefilling_kv_cache.value_cache[layer_idx] = prefilling_kv_cache.value_cache[layer_idx].cpu()

            torch.cuda.empty_cache()

            for j, c_turn in enumerate(ins['multiturn']):

                q_tokens, gt_answer, gt_answer_f, qtype =  c_turn
                input_ids_c = torch.cat([q_tokens, generation_suffix], dim=1)
                length_list.append(ins['input_ids'].shape[-1] + input_ids_c.shape[-1])
                
                prefilling_kv_cache_c = deepcopy_kvcache(prefilling_kv_cache, device)

                try:
                    cont = model.generate_with_q(
                        input_ids = input_ids_c.cuda(),
                        prefilling_kv_cache=prefilling_kv_cache_c,
                        prefilling_position_ids=prefilling_position_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens
                    )
                    del prefilling_kv_cache_c
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.info(f"Error on {i}.{j}-{qtype}: Len: {length_list[-1]}, {e}")
                    del prefilling_kv_cache_c
                    torch.cuda.empty_cache()
                    continue

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids_c, cont)]
                ori_answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pre_answer = parse_multi_choice_response(ori_answer[0], gt_answer_f)
                answer_list.append(pre_answer)
                label_list.append(gt_answer)
                
                if metrics['exact_match'].get(qtype) is None:
                    metrics['exact_match'][qtype] = []
                metrics['exact_match']["overall"].append(gt_answer == pre_answer)
                metrics['exact_match'][qtype].append(gt_answer == pre_answer)
                logger.info(f"{i}.{j}-{qtype}: Len: {length_list[-1]}, Result: {gt_answer == pre_answer}, GT: {gt_answer}, PD: {pre_answer}, GT_F: {gt_answer_f}, PD_F: {ori_answer[0]}")
                
        
    else:
        assert args.model == "qwen2-vl-long"

        for i, ins in enumerate(data):
            prefilling_kv_cache, prefilling_position_ids = model.citr_prefilling(
                input_ids=ins['input_ids'].cuda(),
                attention_mask=torch.ones_like(ins['input_ids']).cuda(),
                use_cache=True,
                pixel_values_videos=ins['pixel_values_videos'].cuda(),
                video_grid_thw=ins['video_grid_thw'].cuda(),
                budget_size = args.budget_size,
                local_len = args.local_len,
                chunk_size = args.chunk_size,
                stabilizers = args.stabilizers_size,
                evict_video_pad_only = args.evict_video_pad_only
            )

            for j, c_turn in enumerate(ins['multiturn']):
                q_tokens, gt_answer, gt_answer_f, qtype =  c_turn
                input_ids_c = torch.cat([q_tokens, generation_suffix], dim=1)
                length_list.append(ins['input_ids'].shape[-1] + input_ids_c.shape[-1])
                prefilling_kv_cache_c = copy.deepcopy(prefilling_kv_cache)
                
                cont = model.citr_generate_with_q(
                    input_ids = input_ids_c.cuda(),
                    prefilling_kv_cache=prefilling_kv_cache_c,
                    prefilling_position_ids=prefilling_position_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_new_tokens
                )
                
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids_c, cont)]
                ori_answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pre_answer = parse_multi_choice_response(ori_answer[0], gt_answer_f)
                answer_list.append(pre_answer)
                label_list.append(gt_answer)
                
                if metrics['exact_match'].get(qtype) is None:
                    metrics['exact_match'][qtype] = []
                metrics['exact_match']["overall"].append(gt_answer == pre_answer)
                metrics['exact_match'][qtype].append(gt_answer == pre_answer)
                logger.info(f"{i}.{j}-{qtype}: Result: {gt_answer == pre_answer}, GT: {gt_answer}, PD: {pre_answer}, GT_F: {gt_answer_f}, PD_F: {ori_answer[0]}")

    logger.info(f"Average Dataset Length: {np.mean(length_list)}")
    logger.info(f"Final Overall Acc: {sum(metrics['exact_match']['overall']) / len(metrics['exact_match']['overall'])}")
    for qtype in metrics['exact_match'].keys():
        if qtype == 'overall':
            continue
        else:
            logger.info(f"Final {qtype} Acc: {sum(metrics['exact_match'][qtype]) / len(metrics['exact_match'][qtype])}")
