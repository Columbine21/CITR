import os
import random

OPTIONS = ["A", "B", "C", "D", "E"]

def message_cons(doc, video_dir, max_pixels, fps, prefix=None):
    message = [{"role": "system", "content": "You are a helpful assistant."}]
    
    question = [doc["question"].strip()]
    for i in range(5):
        question.append(f"{OPTIONS[i]}. {doc[f'a{i}'].strip()}")
    question = "\n".join(question)

    if prefix:
        question = question + "\n" + prefix

    visual = os.path.join(video_dir, str(doc['video']) + '.mp4')
    
    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
        message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": max_pixels, "fps": fps}, {"type": "text", "text": question}]})
        
    return message

def parse_multi_choice_response(response, answer_full, all_choices=OPTIONS):
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

    if len(candidates) == 0:
        # TODO.
        gt_choice, gt_ans = answer_full[0], answer_full[2:]
        if gt_ans.lower() in response.lower():
            candidates.append(gt_choice)

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        raise Exception("Output candidate number greater than 1, which is not allowed.... ")
    else:
        pred_index = candidates[0]

    return pred_index