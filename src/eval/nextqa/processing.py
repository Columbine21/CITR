""" Processing the NExTQA MC (Multiple Choice) Task For Qwen2-VL Model.
"""
import os
import sys
import pickle
import argparse

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

current_path = Path(__file__).resolve()
sys.path.append(str(current_path.parent.parent.parent.parent))

from src.eval.nextqa.utils import message_cons, OPTIONS

def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    # parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size of chunked prefill') TODO
    parser.add_argument('--model_dir', type=str, default='/home/yuanziqi/Work25/Locret-VL/resources/models/Qwen2-VL-7B-Instruct', help='The directory of model')
    parser.add_argument('--data_path', type=str, default='/home/yuanziqi/Work25/Locret-VL/resources/datasets/NExTQA', help='The path of dataset.')
    parser.add_argument('--video_path', type=str, default="/home/yuanziqi/Work25/Locret/datasets/NExTQA/NExTVideo")

    parser.add_argument('--max_pixels', type=int, default=200704) # 256*28*28 / 576*28*28=451584;
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--prefix', type=bool, default=True)
    parser.add_argument('--n_ins', type=int, default=400)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_dir)

    nextqa_mc_test = load_dataset(path=args.data_path, name="MC")['test']
    nextqa_mc_test = nextqa_mc_test.shuffle(seed=42)
    if args.n_ins != -1:
        nextqa_mc_test = nextqa_mc_test.select(range(0, args.n_ins))

    features = []
    pbar = tqdm(total=len(nextqa_mc_test), desc=f"Processing")

    prefix_str = "Please answer directly in the format of X. <Content of X>, where X can be A, B, C, D, or E." if args.prefix else None

    for i, doc in enumerate(nextqa_mc_test):
        chat = message_cons(doc, video_dir=args.video_path, max_pixels=args.max_pixels, fps=args.fps, prefix=prefix_str)
        conversation = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(chat)
        inputs = processor(text=[conversation],images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        info = {
            'id': "_".join([str(doc['video']), str(doc['qid'])]),
            'answer': OPTIONS[doc['answer']],
            'answer_full': f"{OPTIONS[doc['answer']]}. {doc[f"a{doc['answer']}"]}"
        }
        info.update(inputs.__dict__)
        features.append(info)
        pbar.update(1)
            
    pbar.close()
    
    save_path = Path(__file__).parent / "output" / f"nextqa_{args.n_ins}_{args.max_pixels}_{args.fps}.pkl"
    
    print(f"Processing the NExTQA-MC dataset finished, Saving Results to {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    main()

