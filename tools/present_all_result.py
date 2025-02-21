import json
import logging


# Set up logging
log_file = "result_presentation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s",
)

ABILITY_CATEGORIES = ['Objective Causality', 
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

QUESTION_CATEGORIES = [
    "Non-leading Open-ended Question",
    "Paraphrased Open-ended Question",
    "Correctly-led Open-ended Question",
    "Wrongly-led Open-ended Question",
    "Multiple-choice Question with a Single Correct Answer",
]

QUESTION_CATEGORIES_MAPPTING = {
    "0": "Non-leading Open-ended Question",
    "1": "Paraphrased Open-ended Question",
    "2": "Correctly-led Open-ended Question",
    "3": "Wrongly-led Open-ended Question",
    "7": "Multiple-choice Question with a Single Correct Answer",




}

evaluate_dimension = "question"

CATEGORIES = ABILITY_CATEGORIES if evaluate_dimension == "ability" else QUESTION_CATEGORIES

log_name = "hardvideo_open_ended_score"

oe_threshold = 2
robustness_threshold = 4


with open("/opt/tiger/lmms-eval/deleted_qid_llava_7b_qwen2_5_7b_llava_72b_internvl_38B_over_2.json", "r") as f:
    deleted_qid = json.load(f)

deleted_qid = [_.split("-")[0] for _ in deleted_qid]



def hardvideo_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    qid2score = {}

    for category in CATEGORIES:
        category2score[category] = {"correct": 0, "answered": 0}


    for result in results:
        # import pdb;pdb.set_trace()
        qid = result["doc"]["qid"]
        suffix = qid.split("-")[-1]
        preffix = qid.split("-")[0]
        if preffix in deleted_qid:
            continue
        if evaluate_dimension == "ability":
            capability = result[log_name]["capability"]
            category2score[capability]["answered"] += 1
            category2score[capability]["correct"] += result[log_name]["pred_answer"] == result[log_name]["answer"]
        else:
            category = QUESTION_CATEGORIES_MAPPTING[suffix]
            category2score[category]["answered"] += 1
            category2score[category]["correct"] += int(result[log_name]["correctness"] >=oe_threshold)
            if preffix not in qid2score:
                qid2score[preffix] = []
            qid2score[preffix].append(int(result[log_name]["correctness"] >=oe_threshold))


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

    for qid, score in qid2score.items():
        # import pdb;pdb.set_trace()
        qid2score[qid] = 1 if sum(score) >= robustness_threshold else 0

    logging.info(f"Robustness Performance: {100 * sum(qid2score.values()) / len(qid2score) :.1f}%")
    # return 100 * sum(qid2score.values()) / len(qid2score) if len(qid2score) > 0 else 0
    logging.info('*'*100)




def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# File paths
# jsonl_a_path = "/opt/tiger/lmms-eval/logs/gpt-4o-2024-05-13/20250213_081234_samples_hardvideo.jsonl"
# jsonl_b_path = "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250213_094224_samples_hardvideo.jsonl"
# jsonl_c_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250213_121142_samples_hardvideo_single_mc.jsonl"
# jsonl_d_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2_2/20250213_123707_samples_hardvideo_single_mc.jsonl"

jsonl_b_path = "/opt/tiger/lmms-eval/logs/gpt-4o-2024-05-13/20250220_210415_samples_hardvideo_all.jsonl"
# jsonl_c_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250214_130258_samples_hardvideo_single_mc.jsonl"
# jsonl_d_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2/20250214_120151_samples_hardvideo_single_mc.jsonl"


# hardvideo_aggregate_results(load_jsonl(jsonl_a_path))
hardvideo_aggregate_results(load_jsonl(jsonl_b_path))
# hardvideo_aggregate_results(load_jsonl(jsonl_c_path))
# hardvideo_aggregate_results(load_jsonl(jsonl_d_path))

