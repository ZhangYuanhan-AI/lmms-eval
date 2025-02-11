import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file




CATEGORIES = [
    'Element Attributes', 
    'Element Attributes (Optimal Illusion)', 
    'Element Localizations', 
    'Element Counting', 
    'Positional Relationship', 
    'Event Attributes', 
    'Event localization', 
    'Element Counting', 
    'Event Duration', 
    'Character Emotion', 
    'Displacement', 
    'Plot Attributes', 
    'Plot Attributes (Montage)', 
    'Object Causality', 
    'Object Causality (Editing)', 
    'Professional Knowledge', 
    'Character Reaction', 
    'Character Motivation'
]


# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "hardvideo.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]




def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)




def hardvideo_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def hardvideo_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"] + "\n" + doc["question_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:



def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []

# for i in VIDEO_TYPE:
#     for j in CATEGORIES:
#         for k in SUB_CATEGORIES:
#             for l in TASK_CATEGORIES:
#                 matrices.append(f"{i}_{j}_{k}_{l}")


def hardvideo_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case hardvideo score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["capability"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"hardvideo_perception_score": data_dict for metric in matrices}
    return {f"hardvideo_perception_score": data_dict}


def hardvideo_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}


    for result in results:
        category = result["category"]
        category2score[category]["answered"] += 1
        category2score[category]["correct"] += result["pred_answer"] == result["answer"]

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
