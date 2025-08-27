#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/08/22 15:11:57
@email: fjjth98@163.com
@description: Video-MME Evaluation
================================================
"""
import json
import os
import random
import pysubs2
import os.path as osp
from copy import deepcopy
from typing import Any

import torch
from pandas import read_parquet
from decord import VideoReader, cpu
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch.utils.data import default_collate

OPTIONS = ["A", "B", "C", "D"]

def parse_multi_choice_response(response, all_choices=OPTIONS):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        raise Exception("Output candidate number greater than 1, which is not allowed.... ")
    else:
        pred_index = candidates[0]

    return pred_index


def uniform_indices(num_frames: int, total_frames: int) -> list[int]:
    """Get uniform indices 

    Args:
        num_frames (int): number of frames
        total_frames (int): total number of frames

    Returns:
        list[int]: Output frame indices
    """
    if num_frames < total_frames:
        splits = torch.linspace(0, total_frames, num_frames+1, dtype=int)
        indices = ((splits[:-1] + splits[1:]) // 2).tolist()
    else:
        indices = list(range(total_frames))

    return indices

def fps_indices(input_fps: float, total_frames: int, output_fps: float = None, max_num_frames: int = -1) -> list[int]:
    """Get indices according to the output_fps

    Args:
        input_fps (float): input fps
        total_frames (int): total number of frames
        output_fps (float, optional): output fps. Defaults to None, means output_fps==input_fps.
        max_num_frames (int, optional): max number of frames. Defaults to -1, means no limitation.

    Returns:
        list[int]: Output frame indices
    """
    delta = 1 if output_fps is None else input_fps / output_fps
    indices = torch.arange(0, total_frames, delta).round().to(int)
    indices = [e for e in indices if e < total_frames]
    if 0 < max_num_frames < len(indices):
        indices = indices[:max_num_frames]

    return indices

def load_subtitle(sub_path: str, indices: list[int], fps: float) -> str:
    """Load subtitle related to given indices

    Args:
        sub_path (str): subtitle path
        indices (list[int]): frame indices
        fps (float): video average fps

    Returns:
        str: subtitle
    """
    subs = pysubs2.load(sub_path, encoding='utf-8')
    subtitles = []
    for idx in indices:
        sub_text = []
        cur_time = pysubs2.make_time(fps=fps, frames=idx)
        for sub in subs:
            if sub.end < cur_time:
                continue
            elif sub.start < cur_time:
                sub_text.append(sub.text.replace('\\N', ' '))
                break   # in accordance with the official Video-MME Benchmark
            else:
                break
        sub_text = ' '.join(sub_text)
        if sub_text.strip():
            subtitles.append(sub_text)
    subtitles = '\n'.join(subtitles)

    return subtitles

def load_decord(src_path: str, sample_type: str, sub_path: str = None, **kwargs) -> list[Image.Image]:
    """Load video using decord, optionally load subtitles

    Args:
        src_path (str): video path
        sample_type (str): 'uniform' or 'fps'
        sub_path (str): subtitle path, .srt
        kwargs: for 'uniform', require 'num_frames'; for 'fps', optionally require 'output_fps' and 'max_num_frames'

    Returns:
        list[Image.Image] | tuple[list[Image.Image], str]: frame list, subtitle str (optional)
    """
    vr = VideoReader(src_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if sample_type == 'uniform':
        num_frames = kwargs.pop('num_frames')
        indices = uniform_indices(num_frames, total_frames)
    elif sample_type == 'fps':
        input_fps = float(vr.get_avg_fps())
        output_fps = kwargs.pop('output_fps', None)
        max_num_frames = kwargs.pop('max_num_frames', -1)
        indices = fps_indices(input_fps, total_frames, output_fps, max_num_frames)
    else:
        raise ValueError(f'Do not support {sample_type} sample type')
    frames = vr.get_batch(indices).asnumpy()        # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frame) for frame in frames]

    if sub_path is None:
        return frames
    elif osp.exists(sub_path):
        subtitles = load_subtitle(sub_path, indices=indices, fps=float(vr.get_avg_fps()))
        return frames, subtitles
    else:
        return frames, ''


def video_collate_fn(batch: Any) -> Any:
    """this collate function address dict video inputs, support to process variable number of frames for different inputs

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(batch[0], dict) and 'video' in batch[0]:
        video = [b.pop('video') for b in batch]
        batch = default_collate(batch)
        batch['video'] = video
    else:
        batch = default_collate(batch)
    return batch

class VideoMMEDataset(Dataset):
    """Video-MME dataset. By default, videos are saved in `video/` and subtitles are saved in `subtitle/`
    """

    def __init__(self, dataset_path: str, sample_config: dict, use_subtitle: bool = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.sample_config = sample_config
        self.use_subtitle = use_subtitle

        data_dict = {}
        index_keys = ['video_id', 'duration', 'domain', 'sub_category', 'videoID']
        value_keys = ['question_id', 'task_type', 'question', 'options', 'answer']
        df = read_parquet(osp.join(dataset_path, 'videomme', 'test-00000-of-00001.parquet'))
        df['options'] = df['options'].apply(list)
        for _, data in df.iterrows():
            key = tuple(data[k] for k in index_keys)
            value = data[value_keys].to_dict()
            if key in data_dict:
                data_dict[key].append(value)
            else:
                data_dict[key] = [value]
        self.data_list = [dict(zip(index_keys + ['questions'], list(k) + [v])) for k, v in data_dict.items()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> dict:
        if self.use_subtitle:
            frames, subtitles = load_decord(
                src_path=osp.join(self.dataset_path, 'data', self.data_list[idx]['videoID'] + '.mp4'),
                sub_path=osp.join(self.dataset_path, 'subtitle', self.data_list[idx]['videoID'] + '.srt'),
                **self.sample_config
            )
            text = ['\n'.join([
                "This video's subtitles are listed below:",
                subtitles,
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.data_list[idx]['questions']]
        else:
            frames = load_decord(
                src_path=osp.join(self.dataset_path, 'video', self.data_list[idx]['videoID'] + '.mp4'),
                **self.sample_config
            )
            text = ['\n'.join([
                'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.',
                i['question']
            ] + i['options']) for i in self.data_list[idx]['questions']]
            subtitles = ''

        return dict(
            video=frames,
            text=text
        )


@torch.inference_mode
def evaluate(
    model,
    tokenizer,
    image_processor,
    dataset_path: str,
    output_path: str,
    sample_config: dict,
    batch_size: int = 4,
    question_prompt: str = "Answer with the option's letter from the given choices directly."
):
    if not osp.exists(output_path):
        os.makedirs(output_path)

    dataset = VideoMMEDataset(
        dataset_path=dataset_path,
        sample_config=sample_config
    )
    for use_subtitle in (False, True):
        dataset.use_subtitle = use_subtitle
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=video_collate_fn
        )
        results = []
        for data in tqdm(dataloader):
            response = []
            images = data['video']
            for i in range(len(data['text'])):
                messages = [[{
                    'role': 'user',
                    'content': f'<video>\n{question}\n{question_prompt}'
                }] for question in data['text'][i]]
                response.append(model.chat(messages, images, tokenizer, image_processor, max_new_tokens=100, do_sample=False))
            response = [[response[i][j] for i in range(len(response))] for j in range(len(response[0]))]
            results.extend(response)

        outputs = []
        for data, responses in zip(dataset.data_list, results):
            data = deepcopy(data)
            data.pop('videoID')
            for question, response in zip(data['questions'], responses):
                question['response'] = response
            outputs.append(data)

        suffix = 'w_sub' if use_subtitle else 'wo_sub'
        with open(osp.join(output_path, f'output_{suffix}.json'), 'w') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":

    dataset = VideoMMEDataset(
        dataset_path="/data/public/multimodal/yuanziqi/datasets/Video-MME",
        sample_config=dict(
            sample_type='uniform',
            num_frames=96
        ),
        use_subtitle=True,
    )

    print(dataset[0])

    print(11111)