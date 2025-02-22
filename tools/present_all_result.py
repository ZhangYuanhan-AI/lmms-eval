import json
import logging
import re


ABILITY_CATEGORIES = [
            'Element Attributes', 
            'Element Attributes (Optical Illusion)', 
            'Element Localization', 
            'Element Counting', 
            'Positional Relationship', 
            'Local Event Attribute', 
            'Event Localization', 
            'Event Counting', 
            'Event Duration & Speed Attribute', 
            'Character Emotion Attribute',
            'Displacement Attribute', 
            'Plot Attribute', 
            'Plot Attribute (Montage)', 
            'Objective Causality', 
            'Objective Causality (Videography Phenomenon & Illusion)', 
            'Professional Knowledge', 
            'Character Reaction Causality', 
            'Character Motivation Causality', 
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


evaluate_dimension = "ability" # "ability" or "question"

# Set up logging
log_file = f"result_presentation_{evaluate_dimension}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s",
)

CATEGORIES = ABILITY_CATEGORIES if evaluate_dimension == "ability" else QUESTION_CATEGORIES

model_results = []

log_name = "hardvideo_open_ended_score"

oe_threshold = 3
robustness_threshold = 4


with open("/opt/tiger/lmms-eval/deleted_qid_llava_7b_qwen2_5_7b_llava_72b_internvl_38B_over_2.json", "r") as f:
    deleted_qid = json.load(f)

deleted_qid = [_.split("-")[0] for _ in deleted_qid]

with open("/opt/tiger/yx/hardvideo_all_mc_gt.json", "r") as f:
    mc_gt = json.load(f)



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

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]

def hardvideo_aggregate_results(results,model_name,model_idx):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    qid2score = {}
    cur_model_results = []

    for category in CATEGORIES:
        category2score[category] = {"correct": 0, "answered": 0}


    for result in results:
        # import pdb;pdb.set_trace()
        qid = result["doc"]["qid"]
        suffix = qid.split("-")[-1]
        preffix = qid.split("-")[0]
        if preffix in deleted_qid:
            continue

        capability = result[log_name]["capability"]
        category = QUESTION_CATEGORIES_MAPPTING[suffix]
        dimension = result[log_name]["capability"] if evaluate_dimension == "ability" else QUESTION_CATEGORIES_MAPPTING[suffix]

        if evaluate_dimension == "ability" and category != "Multiple-choice Question with a Single Correct Answer":
            continue
            
        category2score[dimension]["answered"] += 1
        if category == "Multiple-choice Question with a Single Correct Answer":
            # import pdb;pdb.set_trace()
            regrex_answer = extract_characters_regex(result["filtered_resps"][0])
            if model_name == "human":
                category2score[dimension]["correct"] += int(regrex_answer == mc_gt[str(result["doc_id"])])
            else:
                category2score[dimension]["correct"] += int(regrex_answer == result[log_name]["answer"])

        else:
            category2score[dimension]["correct"] += int(result[log_name]["correctness"] >=oe_threshold)
        if preffix not in qid2score:
            qid2score[preffix] = []
        qid2score[preffix].append(int(result[log_name]["correctness"] >=oe_threshold))


    
    # import pdb;pdb.set_trace()

    for idx, category in enumerate(CATEGORIES):
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        logging.info(f"{total_answered} Evaluation on capability: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        # import pdb;pdb.set_trace()
        cur_model_results.append(round(100 * total_correct / total_answered if total_answered > 0 else 0,1))

    model_results.append(cur_model_results)

        

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    logging.info(f"{total_answered} Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

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

jsonl_b_path = "/opt/tiger/lmms-eval/logs/human/20250221_141934_samples_hardvideo_all.jsonl"
jsonl_c_path = "/opt/tiger/lmms-eval/logs/OpenGVLab__InternVL2_5-38B/20250221_134034_samples_hardvideo_all.jsonl"
jsonl_d_path = "/opt/tiger/lmms-eval/logs/gpt-4o-2024-05-13/20250220_210415_samples_hardvideo_all.jsonl"
jsonl_e_path = "/opt/tiger/lmms-eval/logs/gpt-4o-2024-05-13/20250221_165054_samples_hardvideo_all.jsonl"  
jsonl_f_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2/20250221_131601_samples_hardvideo_all.jsonl"
jsonl_j_path = "/opt/tiger/lmms-eval/logs/Qwen2-VL-72B-Instruct/20250221_232424_samples_hardvideo_all.jsonl"
jsonl_k_path = "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250222_084414_samples_hardvideo_all.jsonl"
jsonl_l_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2-VL-7B-Instruct/20250222_094002_samples_hardvideo_all.jsonl"
jsonl_m_path = "/opt/tiger/lmms-eval/logs/OpenGVLab__InternVL2_5-8B/20250222_092932_samples_hardvideo_all.jsonl"



for idx,_ in enumerate([jsonl_b_path, jsonl_c_path, jsonl_d_path,jsonl_e_path,jsonl_f_path,jsonl_j_path,jsonl_k_path,jsonl_l_path,jsonl_m_path]):
    # Load JSONL files
    results = load_jsonl(_)
    # Process results
    model_name = _.split("/")[-2]
    logging.info(f"Model Name: {model_name}")
    hardvideo_aggregate_results(results,model_name,idx)

for idx in [3,2,1,4,5,0]:
    print(model_results[idx])


