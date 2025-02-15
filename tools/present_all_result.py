import json
import logging


# Set up logging
log_file = "result_presentation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s",
)

CATEGORIES = ['Objective Causality', 
              'Objective Causality (Videography Phenomenon & Illusion)', 
              'Element Attributes (Optical Illusion)', 
              'Displacement Attribute', 
              'Plot Attribute (Montage)', 
              'Plot Attribute', 
              'Element Attributes', 
              'Element Counting', 
              'Professional Knowledge', 
              'Character Motivation Causality', 
              'Element Localization', 
              'Character Reaction Causality', 
              'Event Counting', 
              'Local Event Attribute', 
              'Event Localization', 
              'Positional Relationship', 
              'Event Duration & Speed Attribute', 
              'Character Emotion Attribute'
              ]


def hardvideo_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for category in CATEGORIES:
        category2score[category] = {"correct": 0, "answered": 0}


    for result in results:
        # import pdb;pdb.set_trace()
        capability = result["hardvideo_perception_score"]["capability"]
        category2score[capability]["answered"] += 1
        category2score[capability]["correct"] += result["hardvideo_perception_score"]["pred_answer"] == result["hardvideo_perception_score"]["answer"]

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        logging.info(f"{total_answered} Evaluation on capability: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    logging.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    logging.info('*'*100)
    return 100 * total_correct / total_answered if total_answered > 0 else 0


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# File paths
# jsonl_a_path = "/opt/tiger/lmms-eval/logs/gpt-4o-2024-05-13/20250213_081234_samples_hardvideo.jsonl"
# jsonl_b_path = "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250213_094224_samples_hardvideo.jsonl"
# jsonl_c_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250213_121142_samples_hardvideo_single_mc.jsonl"
# jsonl_d_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2_2/20250213_123707_samples_hardvideo_single_mc.jsonl"

jsonl_b_path = "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250214_115439_samples_hardvideo_single_mc.jsonl"
# jsonl_c_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250214_130258_samples_hardvideo_single_mc.jsonl"
jsonl_d_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2/20250214_120151_samples_hardvideo_single_mc.jsonl"


# hardvideo_aggregate_results(load_jsonl(jsonl_a_path))
hardvideo_aggregate_results(load_jsonl(jsonl_b_path))
# hardvideo_aggregate_results(load_jsonl(jsonl_c_path))
hardvideo_aggregate_results(load_jsonl(jsonl_d_path))

