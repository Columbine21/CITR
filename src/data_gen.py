import os
import sys
import torch
import psutil
import argparse

from tqdm import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Current Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def chat_cons(video_id, question, answer, video_dir):
    video_path = os.path.join(video_dir, str(video_id) + '.mp4')
    prompt = {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 256 * 28 * 28, "fps": 1},
            {"type": "text", "text": question}
        ],
    }
    answer = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": answer}
        ],
    }
    return prompt, answer

def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--num', type=int, default=8000, help='The number of trained entries generated.')
    parser.add_argument('--model_dir', type=str, default='/home/yuanziqi/Work25/Locret-VL/resources/models/Qwen2-VL-7B-Instruct', help='The directory of model')
    parser.add_argument('--data_path', type=str, default='/home/yuanziqi/Work25/Locret-VL/resources/datasets/NExTQA', help='The path of dataset.')
    parser.add_argument('--video_path', type=str, default="/home/yuanziqi/Work25/Locret/datasets/NExTQA/NExTVideo")
    parser.add_argument('--save_dir', type=str, default='./output_long_7b', help='The output directory of generated data entries.')

    args = parser.parse_args()
    return args

def build_dataset_rank(config, processor, dataset, model_type):
    
    original_columns = dataset.column_names
    num_proc = 1

    # THIS FUNCTION NEEDS TO ADAPT TO THE SPECIFIC MODEL!!!
    # Phi3 and Llama-3 happens to share the same function
    def preprocess_function(examples):
        new_examples = {
            "input_ids": [],
            "pixel_values_videos": [],
            "video_grid_thw": [],
            "loss_mask": [],
        }
        for i in range(len(examples['video'])):
            prompt, answer = chat_cons(
                examples['video'][i], 
                examples['question'][i], 
                examples['answer'][i],
                config.video_path
            )
            chat = [prompt, answer]
            
            prompt = processor.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
            prompt_length = len(processor.tokenizer(prompt).input_ids)
            conversation = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

            image_inputs, video_inputs = process_vision_info(chat)

            inputs = processor(
                text=[conversation],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            n_videopad_token = torch.prod(inputs.video_grid_thw).item() // 4
            prompt_length = prompt_length - 1 + n_videopad_token
            loss_mask = torch.ones_like(inputs.input_ids)
            loss_mask[0, :prompt_length] = 0

            new_examples["input_ids"].append(inputs.input_ids)
            new_examples["pixel_values_videos"].append(inputs.pixel_values_videos)
            new_examples["video_grid_thw"].append(inputs.video_grid_thw)
            new_examples["loss_mask"].append(loss_mask)

        return new_examples
    
    ds1 = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1

def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

@torch.inference_mode()
def generate_data(model, data):
    input_ids = data["input_ids"].cuda()
    pixel_values_videos = data["pixel_values_videos"].cuda()
    video_grid_thw = data["video_grid_thw"].cuda()
    prefix_length = input_ids.shape[1] - data["loss_mask"].sum()

    cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

    # prefilling state.
    output = model(
        input_ids=input_ids[...,:prefix_length], 
        pixel_values_videos=pixel_values_videos, 
        video_grid_thw=video_grid_thw, 
        use_cache=True, 
        output_attentions=False, 
        plain_attn=False, 
        visual_sliding_window=True
    )
    output = model(
        input_ids=input_ids[..., prefix_length:], 
        past_key_values=output.past_key_values,
        output_pre_act_attentions=True, 
        plain_attn=True,
    )
    attentions = output.pre_act_attentions
    
    weights = []
    for attn in attentions:
        batch, num_head, input_seqlen, output_seqlen = attn.shape
        num_key_value_heads = model.config.num_key_value_heads
        attn = attn.view(batch, num_key_value_heads, num_head // num_key_value_heads, input_seqlen, output_seqlen)
        attn = attn.max(dim=2).values
        weights.append(
            attn[:, :, :, :prefix_length].max(dim=-2).values
        )
    
    weights = torch.stack(weights, dim=0)
    new_data = {
        "input_ids": input_ids.cpu()[0][:prefix_length],
        "pixel_values_videos": pixel_values_videos.cpu(),
        "video_grid_thw": video_grid_thw.cpu(),
        "weights": weights
    }
    return new_data


if __name__ == '__main__':
    args = parse_args()

    if 'qwen2-vl-7b' in args.model_dir.lower():
        model_type = "qwen2-vl-7b-instruct"
        model_id = "qwen2_vl"
    elif 'qwen2-vl-72b' in args.model_dir.lower():
        model_type = "qwen2-vl-72b-instruct"
        model_id = "qwen2_vl_72b"
    else:
        raise "Model type not supported yet!"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir)

    if model_type == "qwen2-vl-7b-instruct" or model_type == "qwen2-vl-72b-instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir,  device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    outdir = f'{args.save_dir}/{model_id}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cnt = 0
    succ = 0

    for start_id in range(0, 10000, 40):
        dataset = load_dataset(path=args.data_path, name="OE")
        # merge validation and test
        dataset = concatenate_datasets([dataset['test'], dataset['validation']])
        dataset = dataset.shuffle(seed=42)
        dataset_sub = dataset.select(range(start_id, start_id + 40)) 
        print(dataset_sub)
        dataset_sub = build_dataset_rank(args, processor, dataset_sub, model_type) # TODO: construct the dataset function to implement.
        

        for data in tqdm(dataset_sub):
            try:
                outdata = generate_data(model, data)
                writedata(outdir, outdata)
                succ += 1
            except Exception as e:
                print(f"error on idx: {cnt}", e)
            cnt += 1
            if succ >= args.num:
                break
        print(f"Generation finished. {succ} entries is generated by {cnt} original data. {cnt - succ} failed.")
        
        del dataset_sub
        del dataset
