""" Processing the VideoMME (Long) Task For Qwen2-VL Model.
"""

import os
import pickle
import decord
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor
from torchvision import transforms
from qwen_vl_utils import smart_resize
from torchvision.transforms import InterpolationMode

IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2

def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    # parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size of chunked prefill') TODO
    parser.add_argument('--model_dir', type=str, default='Qwen2-VL-7B-Instruct', help='The directory of model')
    parser.add_argument('--data_path', type=str, default='resources/datasets/Video-MME', help='The path of dataset.')

    parser.add_argument('--question_awared', type=bool, default=False) #  quesion-awared

    parser.add_argument('--use_nframes', type=bool, default=True)
    parser.add_argument('--fps', type=float, default=0.5)
    parser.add_argument('--nframes', type=int, default=256) # NOTE Qwen2-vl processor has default 768 max_frames. fps should not be used.
    parser.add_argument('--max_pixels', type=int, default=200704) # 256*28*28 / 576*28*28=451584;
    
    args = parser.parse_args()
    return args

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR):
    
    vr = decord.VideoReader(ele['video'])

    vlen = len(vr)
    fps = vr.get_avg_fps()
    duration = vlen / float(fps)

    idx = torch.linspace(0, vlen - 1, ele['nframes']).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)

    nframes, _, height, width = video.shape

    msg = f"The video lasts for {duration:.2f} seconds, and {ele['nframes']} frames are uniformly sampled from it. "

    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max_pixels)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video, msg


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_dir)

    videomme_test = load_dataset(path=args.data_path)
    videomme_long_test = videomme_test.filter(lambda x: x['duration'] == 'long')['test']

    index_keys = ['video_id', 'duration', 'domain', 'sub_category', 'videoID']
    value_keys = ['question_id', 'task_type', 'question', 'options', 'answer']

    data_dict = {}
    for data in videomme_long_test:
        key = tuple(data[k] for k in index_keys)
        value = {key: data[key] for key in data.keys() if key in value_keys}

        if key in data_dict:
            data_dict[key].append(value)
        else:
            data_dict[key] = [value]
    
    data_list = [dict(zip(index_keys + ['questions'], list(k) + [v])) for k, v in data_dict.items()]
    if args.use_nframes:
        meta_v_infos = {"type": "video", "video": None, "max_pixels": args.max_pixels, "nframes": args.nframes} 
        if args.question_awared:
            save_path = Path(__file__).parent / "output" / f"videomme_long_wosub_{args.max_pixels}_{args.nframes}_timeprompt_qa.pkl"
        else:   
            save_path = Path(__file__).parent / "output" / f"videomme_long_wosub_{args.max_pixels}_{args.nframes}_timeprompt.pkl"
    else:
        meta_v_infos = {"type": "video", "video": None, "max_pixels": args.max_pixels, "fps": args.fps}
        save_path = Path(__file__).parent / "output" / f"videomme_long_wosub_{args.max_pixels}_{args.fps}_timeprompt.pkl"

    features = []
    pbar = tqdm(total=len(data_list), desc=f"Processing")

    for video_info in data_list:
        video_path = os.path.join(args.data_path, "data", video_info['videoID']+ ".mp4")
        meta_v_infos.update({"video": video_path})
        video_inputs, time_prompt = fetch_video(meta_v_infos)
        video_inputs = [video_inputs]

        videos_inputs = processor.image_processor(images=None, videos=video_inputs, return_tensors="pt")
        video_grid_thw = videos_inputs["video_grid_thw"]

        questions = []
        for q in video_info['questions']:

            if args.question_awared:
                text_ins = '\n'.join(["You are given a multiple choice question and a video. Please answer the question according to the given video directly with only the letter (A, B, C, or D) of the correct option. " + time_prompt,
                                    q['question']] + q['options'])
                chat = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "text", "text": text_ins}, {"type": "video", "video": video_path}]}
                ]

            else: 
                text_ins = '\n'.join([time_prompt + 'Please answer the multiple choice question related to this video directly with only the letter (A, B, C, or D) of the correct option.', q['question']] + q['options'])
                chat = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text_ins}]}
                ]
            conversation = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            merge_length = processor.image_processor.merge_size**2
            
            while processor.video_token in conversation:
                conversation = conversation.replace(
                    processor.video_token, "<|placeholder|>" * (video_grid_thw[0].prod() // merge_length), 1
                )
            conversation = conversation.replace("<|placeholder|>", processor.video_token)

            text_inputs = processor.tokenizer(conversation, return_tensors="pt")

            questions.append({
                "question_id": q['question_id'], "task_type": q['task_type'], "input_ids": text_inputs.input_ids, 
                "answer": q['answer'],
            })

        info = {
            "videoID": video_info["videoID"],
            "pixel_values_videos": videos_inputs.pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "questions": questions,
        }

        features.append(info)
        pbar.update(1)

    pbar.close()
    
    print(f"Processing the VideoMME Long dataset (Without Subtitle) finished, Saving Results to {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    main()

