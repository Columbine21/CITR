import os
import sys
import argparse
import pickle
import logging
import numpy as np
import random
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer

current_path = Path(__file__).resolve()
sys.path.append(str(current_path.parent.parent.parent.parent))

from src.model.qwen2_vl.monkey_patch import patch_qwen2vl
from src.eval.videomme.utils import parse_multi_choice_response

DATA_MAP = {
    'videomme_mc_wo_long_200k_256_tp': "src/eval/videomme/output/videomme_long_wosub_200704_256_timeprompt.pkl",
}

def _set_logger(args, verbose_level):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # base logger
    model_name = args.model + '-citr' if args.citr else args.model
    local_len =  local_len if args.local_len != -1 else "auto"
    if args.citr:
        suffix = "-evict_video_only" if args.evict_video_pad_only else ""
        log_file_path = Path(args.log_dir) / f"{model_name}_{args.dataset}_{args.budget_size}-{local_len}-{args.chunk_size}-{args.stabilizers_size}-{args.merge_threshold}-{args.merge_ratio_upper_bound}{suffix}.log"
    else:
        log_file_path = Path(args.log_dir) / f"{model_name}_{args.dataset}.log"
    
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
    parser.add_argument('--model', type=str, default="qwen2-vl-long", choices=["qwen2-vl-long", "qwen2-vl-ori", "qwen2.5-vl-ori"])
    parser.add_argument('--dataset', type=str, default='videomme_mc_wo_long_200k_256_tp', choices=['videomme_mc_wo_long_200k_256_tp'])

    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])

    parser.add_argument('--model_dir', type=str, default='checkpoints_7b_evict_vtoken_only_3k/final_model', help='The directory of model')
    parser.add_argument('--tokenizer_dir', type=str, default='Qwen2-VL-7B-Instruct', help='The directory of model')
    
    # generation hyperparameters.
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--sample', type=bool, default=False)

    # hyperparameters.
    parser.add_argument('--budget_size', type=int, default=8192)
    parser.add_argument('--local_len', type=int, default=-1) # -1 for auto local_len (last user instruction is maintained without eviction.)
    parser.add_argument('--chunk_size', type=int, default=4096)
    parser.add_argument('--stabilizers_size', type=int, default=128)
    parser.add_argument('--evict_video_pad_only', type=bool, default=True)

    parser.add_argument('--use_similarity_merge', type=bool, default=True)
    parser.add_argument('--merge_threshold', type=float, default=0.4) # Similarity thresholds.
    parser.add_argument("--merge_ratio_upper_bound", type=float, default=0.4, help="Projector merging ratio upper bound") 
    parser.add_argument('--use_frame_similarity_merge', type=bool, default=False)
    parser.add_argument('--use_varlen_budget', type=bool, default=False)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--log_dir', type=str, default="src/eval/videomme")

    args = parser.parse_args()
    return args

def setup_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    args = parse_args()
    setup_seed(args.seed)

    if 'qwen2-vl' in args.model:
        if '7b' in args.tokenizer_dir.lower():
            args.log_dir = args.log_dir + "/log_qwen2-vl-7b"
        else:
            args.log_dir = args.log_dir + "/log_qwen2-vl-72b"

        if args.model == "qwen2-vl-long" or args.citr:
            from src.model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        else:
            assert args.model_dir == args.tokenizer_dir, "If not citr mode, model_dir and tokenizer_dir should be the same."
            from transformers import Qwen2VLForConditionalGeneration

    logger = _set_logger(args, 1)
    logger.info(args)

    
    logger.info(f"Begin to load dataset - {args.dataset} ...")
    with open(DATA_MAP[args.dataset], "rb") as f:
        data = pickle.load(f)

    logger.info(f"Begin to load pretrained model - {args.model} ...")
    if 'qwen2-vl' in args.model:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype="auto", device_map="auto", attn_implementation=args.attn_implementation)
    
    if args.use_varlen_budget:
        
        patch_qwen2vl()
    
    model.eval()
    processor = AutoProcessor.from_pretrained(args.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    answer_list, label_list, length_list = [], [], []
    metrics = {'exact_match': {"overall": []}}

    if not args.citr:
        if args.model == "qwen2-vl-long":
            kwargs = {"visual_sliding_window": True, "output_attentions": False, "plain_attn": False,}
        elif args.model == "qwen2-vl-ori" or "qwen2.5-vl-ori":
            kwargs = {}

        for i, ins in enumerate(data):

            for j, q in enumerate(ins['questions']):
                question_id_c, task_type_c, input_ids_c, gt_answer_c = q['question_id'], q['task_type'], q['input_ids'], q['answer']
                length_list.append(input_ids_c.shape[-1])

                try:
                    cont = model.generate(
                            input_ids = input_ids_c.cuda(),
                            pixel_values_videos = ins['pixel_values_videos'].cuda(),
                            video_grid_thw = ins['video_grid_thw'].cuda(),
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
                

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids_c, cont)]
                ori_answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answers = parse_multi_choice_response(ori_answers[0])
                answer_list.append(answers)
                label_list.append(gt_answer_c)
                if metrics['exact_match'].get(task_type_c) is None:
                    metrics['exact_match'][task_type_c] = []
                metrics['exact_match']["overall"].append(gt_answer_c == answers)
                metrics['exact_match'][task_type_c].append(gt_answer_c == answers)
                logger.info(f"{i}.{j}-{task_type_c}: Length: {input_ids_c.shape[-1]}, Result: {gt_answer_c == answers}, GT: {gt_answer_c}, PD: {answers}, PD_F: {ori_answers[0]}")

    else:
        assert args.model == "qwen2-vl-long"

        for i, ins in enumerate(data):

            for j, q in enumerate(ins['questions']):
                question_id_c, task_type_c, input_ids_c, gt_answer_c = q['question_id'], q['task_type'], q['input_ids'], q['answer']
                length_list.append(input_ids_c.shape[-1])
                
                cont, p_cr = model.citr_generate(
                    input_ids = input_ids_c.cuda(),
                    pixel_values_videos = ins['pixel_values_videos'].cuda(),
                    video_grid_thw = ins['video_grid_thw'].cuda(),
                    max_new_tokens = args.max_new_tokens,
                    eos_token_id =tokenizer.eos_token_id,
                    budget_size = args.budget_size,
                    local_len = args.local_len,
                    chunk_size = args.chunk_size,
                    stabilizers = args.stabilizers_size,
                    evict_video_pad_only = args.evict_video_pad_only,
                    use_similarity_merge=args.use_similarity_merge,
                    use_varlen_budget=args.use_varlen_budget,
                    merge_threshold=args.merge_threshold, 
                    merge_ratio_upper_bound=args.merge_ratio_upper_bound
                )

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids_c, cont)]
                ori_answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answers = parse_multi_choice_response(ori_answers[0])
                answer_list.append(answers)
                label_list.append(gt_answer_c)
                if metrics['exact_match'].get(task_type_c) is None:
                    metrics['exact_match'][task_type_c] = []
                metrics['exact_match']["overall"].append(gt_answer_c == answers)
                metrics['exact_match'][task_type_c].append(gt_answer_c == answers)
                logger.info(f"{i}.{j}-{task_type_c}: Length: {input_ids_c.shape[-1]}, Projector Compression Ratio: {p_cr:.2f}, Result: {gt_answer_c == answers}, GT: {gt_answer_c}, PD: {answers}, PD_F: {ori_answers[0]}")


    logger.info(f"Average Dataset Length: {np.mean(length_list)}")
    logger.info(f"Final Overall Acc: {sum(metrics['exact_match']['overall']) / len(metrics['exact_match']['overall'])}")
    for qtype in metrics['exact_match'].keys():
        if qtype == 'overall':
            continue
        else:
            logger.info(f"Final {qtype} Acc: {sum(metrics['exact_match'][qtype]) / len(metrics['exact_match'][qtype])}")
