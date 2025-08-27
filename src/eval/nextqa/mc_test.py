import os
import sys
import argparse
import pickle
import torch
import logging
import numpy as np
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer
current_path = Path(__file__).resolve()
sys.path.append(str(current_path.parent.parent.parent.parent))

from src.eval.nextqa.utils import parse_multi_choice_response

DATA_MAP = {
    'nextqa_mc_400_200k_1': "src/eval/nextqa/output/nextqa_400_200704_1.0.pkl"
}

def _set_logger(args, verbose_level):

    model_name = args.model + '-citr' if args.citr else args.model
    local_len =  local_len if args.local_len != -1 else "auto"
    if args.citr:
        suffix = "-evict_video_only" if args.evict_video_pad_only else ""
        log_file_path = Path(args.log_dir) / f"{model_name}_{args.dataset}_{args.budget_size}-{local_len}-{args.chunk_size}-{args.stabilizers_size}{suffix}.log"
    else:
        log_file_path = Path(args.log_dir) / f"{model_name}_{args.dataset}.log"

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
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
    parser = argparse.ArgumentParser(description='inference test')
    # parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size of chunked prefill') TODO
    parser.add_argument('--citr', type=bool, default=True, help='Whether to use citr inference.')
    parser.add_argument('--model', type=str, default="qwen2-vl-long", choices=["qwen2-vl-long", "qwen2-vl-ori"])
    parser.add_argument('--dataset', type=str, default='nextqa_mc_400_200k_1', choices=['nextqa_mc_400_200k_1'])

    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])

    parser.add_argument('--model_dir', type=str, default='checkpoints_7b_evict_vtoken_only_3k', help='The directory of model')
    parser.add_argument('--tokenizer_dir', type=str, default='Qwen2-VL-7B-Instruct', help='The directory of model')
    
    # generation hyperparameters.
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--sample', type=bool, default=False)

    # hyperparameters.
    parser.add_argument('--budget_size', type=int, default=1024)
    parser.add_argument('--local_len', type=int, default=-1) # -1 for auto local_len (last user instruction is maintained without eviction.)
    parser.add_argument('--chunk_size', type=int, default=4096)
    parser.add_argument('--stabilizers_size', type=int, default=128)
    parser.add_argument('--evict_video_pad_only', type=bool, default=True)

    parser.add_argument('--log_dir', type=str, default="/home/yuanziqi/CITR/src/eval/nextqa/log_qwen2-vl-7b")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    logger = _set_logger(args, 1)

    logger.info(args)

    if args.model == "qwen2-vl-long" or args.citr:
        from src.model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    else:
        assert args.model_dir == args.tokenizer_dir, "If not citr mode, model_dir and tokenizer_dir should be the same."
        from transformers import Qwen2VLForConditionalGeneration

    with open(DATA_MAP[args.dataset], "rb") as f:
        data = pickle.load(f)

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype="auto", device_map="auto", attn_implementation=args.attn_implementation)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    answer_list, label_list, length_list = [], [], []
    metrics = {'exact_match': []}

    if not args.citr:
        if args.model == "qwen2-vl-long":
            kwargs = {"visual_sliding_window": True, "output_attentions": False, "plain_attn": False,}
        elif args.model == "qwen2-vl-ori":
            kwargs = {}   

        for i, ins in enumerate(data):
            try:
                cont = model.generate(
                        input_ids = ins['data']['input_ids'].cuda(),
                        attention_mask = ins['data']['attention_mask'].cuda(),
                        pixel_values_videos = ins['data']['pixel_values_videos'].cuda(),
                        video_grid_thw = ins['data']['video_grid_thw'].cuda(),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        **kwargs
                    )
            except Exception as e:
                logger.info(f"error on idx: {i}, {e}")
                continue
                

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(ins['data']['input_ids'], cont)]
            ori_answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answers = parse_multi_choice_response(ori_answers[0], ins['answer_full'])
            answer_list.append(answers)
            label_list.append(ins['answer'])
            metrics['exact_match'].append(ins['answer'] == answers)

            logger.info(f"{i}: Instance length: {ins['data']['input_ids'].shape[-1]}. Result: {ins['answer'] == answers}, GT: {ins['answer']}, PD: {answers}, GT(F): {ins['answer_full']}, PD(F): {ori_answers[0]}")

        logger.info(f"Final Exact Match Acc: {sum(metrics['exact_match']) / len(metrics['exact_match'])}")
    else:
        assert args.model == "qwen2-vl-long"

        for i, ins in enumerate(data):
            length_list.append(ins['data']['input_ids'].shape[-1])
        
            cont = model.citr_generate(
                input_ids = ins['data']['input_ids'].cuda(),
                attention_mask = ins['data']['attention_mask'].cuda(),
                pixel_values_videos = ins['data']['pixel_values_videos'].cuda(),
                video_grid_thw = ins['data']['video_grid_thw'].cuda(),
                max_new_tokens = args.max_new_tokens,
                eos_token_id =tokenizer.eos_token_id,
                budget_size = args.budget_size,
                local_len = args.local_len,
                chunk_size = args.chunk_size,
                stabilizers = args.stabilizers_size,
                evict_video_pad_only = args.evict_video_pad_only
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(ins['data']['input_ids'], cont)]
            ori_answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answers = parse_multi_choice_response(ori_answers[0], ins['answer_full'])
            answer_list.append(answers)
            label_list.append(ins['answer'])
            metrics['exact_match'].append(ins['answer'] == answers)

            logger.info(f"{i}: Instance length: {ins['data']['input_ids'].shape[-1]}. Result: {ins['answer'] == answers}, GT: {ins['answer']}, PD: {answers}, GT(F): {ins['answer_full']}, PD(F): {ori_answers[0]}")

        logger.info(f"Average Dataset Length: {np.mean(length_list)}")
        logger.info(f"Final Exact Match Acc: {sum(metrics['exact_match']) / len(metrics['exact_match'])}")
