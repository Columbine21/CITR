""" Processing the NExTQA MC (Multiple Choice) Task For Qwen2-VL Model.
"""
import os
import sys
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    # parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size of chunked prefill') TODO
    parser.add_argument('--model_dir', type=str, default='Qwen2-VL-72B-Instruct', help='The directory of model')
    parser.add_argument('--data_path', type=str, default='/home/yuanziqi/Work25/Locret-VL/resources/datasets/NExTQA', help='The path of dataset.')
    parser.add_argument('--video_path', type=str, default="/home/yuanziqi/Work25/Locret/datasets/NExTQA/NExTVideo")

    parser.add_argument('--max_pixels', type=int, default=200704) # 256*28*28 / 576*28*28=451584;
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--prefix', type=bool, default=True)

    args = parser.parse_args()
    return args

def gather(dataset):
    num_processes = 1
    results = {}

    original_columns1 = dataset.column_names

    def process_doc(doc):
        choice_list = ["A", "B", "C", "D", "E"]
        video_id = doc["video"]
        question = [doc["question"].strip()]
        for i, opt in enumerate(choice_list):
            question.append(f"{opt}. {doc[f'a{i}'].strip()}")
        question = "\n".join(question)
        question = question + "\n" + "Please answer directly in the format of X. <Content of X>, where X can be A, B, C, D, or E."

        return {
            "video_id": video_id,
            "question": question,
            "answer": choice_list[doc["answer"]],
            "answer_full": f"{choice_list[doc['answer']]}. {doc[f"a{doc['answer']}"]}",
            "type": doc["type"]
        }

    processed_data = dataset.map(process_doc, num_proc=num_processes, batched=False, remove_columns=original_columns1)
    
    for item in processed_data:
        video_id, question, answer, answer_full, qtype = item["video_id"], item["question"], item["answer"], item["answer_full"], item["type"]
        if str(video_id) not in results:
            results[str(video_id)] = []
        results[str(video_id)].append([question, answer, answer_full, qtype,])

    return results

def message_cons(video_id, video_path, max_pixels, fps, question):
    message = [{"role": "system", "content": "You are a helpful assistant."}]

    visual = os.path.join(video_path, video_id + '.mp4')
    if os.path.exists(visual):
        # message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": max_pixels, "fps": fps}]})
        if question is not None:
            message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": max_pixels, "fps": fps}, {"type": "text", "text": question}]})
        else: 
            message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": max_pixels, "fps": fps}]})
    return message


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    nextqa_mc_test = load_dataset(path=args.data_path, name="MC")['test']
    nextqa_mc_test = gather(nextqa_mc_test)

    prefix, suffix = "<|im_start|>user\n", "<|im_end|>\n"
    question_count_list = []
    features = []

    pbar = tqdm(total=len(nextqa_mc_test), desc=f"Processing")
    for vid, content in nextqa_mc_test.items():
        question_count_list.append(len(content))
        multiturn = []
        for i in range(len(content)):
            question = prefix + content[i][0] + suffix
            question_token = tokenizer(question, return_tensors="pt").input_ids
            answer, answer_full, qtype = content[i][1], content[i][2], content[i][3]
            multiturn.append([question_token, answer, answer_full, qtype])
        
        chat = message_cons(vid, args.video_path, args.max_pixels, args.fps, None)
        conversation = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(chat)
        inputs = processor(text=[conversation],images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        info = {
            'input_ids': inputs.input_ids,
            'pixel_values_videos': inputs.pixel_values_videos,
            'video_grid_thw': inputs.video_grid_thw,
            'multiturn': multiturn
        }
        features.append(info)
        pbar.update(1)
    
    pbar.close()

    save_path = Path(__file__).parent / "output" / f"nextqa_{args.max_pixels}_{args.fps}_mt.pkl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Processing the NExTQA-MC dataset finished, Saving Results to {save_path}")

    with open(save_path, 'wb') as f:
        pickle.dump(features, f)

if __name__ == "__main__":
    main()

