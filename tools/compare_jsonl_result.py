import json
import logging

# Set up logging
log_file = "compare_0218.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(message)s",
)

# File paths
jsonl_a_path = "/opt/tiger/lmms-eval/logs/gpt-4o-2024-05-13/20250218_113410_samples_hardvideo_single_mc.jsonl"
# jsonl_b_path = "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250213_094224_samples_hardvideo.jsonl"
# jsonl_c_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250213_121142_samples_hardvideo_single_mc.jsonl"
# jsonl_d_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2_2/20250213_123707_samples_hardvideo_single_mc.jsonl"

# jsonl_b_path = "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250214_115439_samples_hardvideo_single_mc.jsonl"
# jsonl_c_path =  "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250214_130258_samples_hardvideo_single_mc.jsonl"
# jsonl_d_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2/20250214_120151_samples_hardvideo_single_mc.jsonl"


jsonl_b_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2/20250218_113722_samples_hardvideo_single_mc.jsonl"
jsonl_c_path =  "/opt/tiger/lmms-eval/logs/LLaVA-NeXT-Video-7B-Qwen2__/20250218_113410_samples_hardvideo_single_mc.jsonl"
jsonl_d_path = "/opt/tiger/lmms-eval/logs/OpenGVLab__InternVL2_5-38B/20250218_135857_samples_hardvideo_single_mc.jsonl"
jsonl_e_path = "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250218_130939_samples_hardvideo_single_mc.jsonl"

model_b_name = jsonl_b_path.split("/")[-2]
model_c_name = jsonl_c_path.split("/")[-2]
model_d_name = jsonl_d_path.split("/")[-2]
model_e_name = jsonl_e_path.split("/")[-2]

jsonl_b_oe_path =  "/opt/tiger/lmms-eval/logs/OpenGVLab__InternVL2_5-38B/20250218_142638_samples_hardvideo_no_leading_oe.jsonl"
# jsonl_c_oe_path =  "/opt/tiger/lmms-eval/logs/Qwen__Qwen2.5-VL-7B-Instruct/20250213_143102_samples_hardvideo_no_leading_oe.jsonl"
# jsonl_d_oe_path = "/opt/tiger/lmms-eval/logs/checkpoints__LLaVA-Video-72B-Qwen2/20250213_140028_samples_hardvideo_no_leading_oe.jsonl"


model_b_oe_name = jsonl_b_oe_path.split("/")[-2]

distrubution = {}


# Load JSONL files
def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

jsonl_a = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_a_path)}
jsonl_b = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_b_path)}
jsonl_c = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_c_path)}
jsonl_d = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_d_path)}
jsonl_e = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_e_path)}

meta_b_oe = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_b_oe_path)}
# meta_c_oe = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_c_oe_path)}
# meta_d_oe = {entry["doc"]["qid"]: entry for entry in load_jsonl(jsonl_d_oe_path)}

refine_qid = []
counter = 0

for qid in jsonl_a:
    base_qid = qid[:-2] + "-0"

    capability = jsonl_a[qid]["doc"]["capability"]

    if qid not in jsonl_b or qid not in jsonl_c or qid not in jsonl_d or qid not in jsonl_e:
        continue

    meta_a, meta_b, meta_c, meta_d, meta_e = jsonl_a[qid],jsonl_b[qid], jsonl_c[qid], jsonl_d[qid], jsonl_e[qid]
    
    if base_qid not in meta_b_oe:
        continue
    
    meta_a_prediction = meta_a["hardvideo_perception_score"].get('pred_answer', 'E') or 'E'
    meta_b_prediction = meta_b["hardvideo_perception_score"]["pred_answer"]
    meta_c_prediction = meta_c["hardvideo_perception_score"]["pred_answer"]
    meta_d_prediction = meta_d["hardvideo_perception_score"]["pred_answer"]
    meta_e_prediction = meta_e["hardvideo_perception_score"]["pred_answer"]

    answer_a = meta_a["hardvideo_perception_score"]["answer"]
    answer_b = meta_b["hardvideo_perception_score"]["answer"]
    answer_c = meta_c["hardvideo_perception_score"]["answer"]
    answer_d = meta_d["hardvideo_perception_score"]["answer"]
    answer_e = meta_e["hardvideo_perception_score"]["answer"]

    meta_b_prediction_oe = meta_b_oe[base_qid]["filtered_resps"][0]
    # meta_c_prediction_oe = meta_c_oe[base_qid]["hardvideo_perception_score"]["pred_answer"]
    # meta_d_prediction_oe = meta_d_oe[base_qid]["hardvideo_perception_score"]["pred_answer"]

    question_oe = meta_b_oe[base_qid]["doc"]["question"]

    # if 2 <= (int(meta_b_prediction == answer_b) + int(meta_c_prediction == answer_c) + int(meta_d_prediction == answer_d) + int(meta_e_prediction == answer_e)) <= 5 and meta_a_prediction != answer_a:
    if 1 <= (int(meta_d_prediction == answer_d)) <= 5 and meta_a_prediction != answer_a:

        refine_qid.append(qid)

        if capability not in distrubution:
            distrubution[capability] = 0
        
        distrubution[capability] += 1


        try:

            if meta_a_prediction != 'D':
                meta_a_prediction = meta_a["doc"]["question"].split(f"{meta_a_prediction}. ")[1].split("\n")[0]
            else:
                meta_a_prediction = meta_a["doc"]["question"].split(f"{meta_a_prediction}. ")[1]
            
            if meta_b_prediction!= 'D':
                meta_b_prediction = meta_b["doc"]["question"].split(f"{meta_b_prediction}. ")[1].split("\n")[0]
            else:
                meta_b_prediction = meta_b["doc"]["question"].split(f"{meta_b_prediction}. ")[1]

            if meta_c_prediction!= 'D':
                meta_c_prediction = meta_c["doc"]["question"].split(f"{meta_c_prediction}. ")[1].split("\n")[0]
            else:
                meta_c_prediction = meta_c["doc"]["question"].split(f"{meta_c_prediction}. ")[1]

            if meta_d_prediction!= 'D':
                meta_d_prediction = meta_d["doc"]["question"].split(f"{meta_d_prediction}. ")[1].split("\n")[0]
            else:
                meta_d_prediction = meta_d["doc"]["question"].split(f"{meta_d_prediction}. ")[1]

            if meta_e_prediction!= 'D':
                meta_e_prediction = meta_e["doc"]["question"].split(f"{meta_e_prediction}. ")[1].split("\n")[0]
            else:
                meta_e_prediction = meta_e["doc"]["question"].split(f"{meta_e_prediction}. ")[1]


            if answer_a!= 'D':
                answer_b = meta_b["doc"]["question"].split(f"{answer_b}. ")[1].split("\n")[0]
            else:
                answer_b = meta_b["doc"]["question"].split(f"{answer_b}. ")[1]

        except:
            continue
        
        output = [
            f'Qid: {qid}',
            f'Question: {meta_b["doc"]["question"]}',
            f'Youtube URL: {meta_b["doc"].get("youtube_url", "N/A")}',
            f'Answer: {answer_b}',
            f'GPT-4o: {meta_a_prediction}',
            f'{model_b_name}: {meta_b_prediction}',
            f'{model_c_name}: {meta_c_prediction}',
            f'{model_d_name}: {meta_d_prediction}',
            f'{model_e_name}: {meta_e_prediction}',
            '*'*100,
            f'Question: {question_oe}',
            f'{model_b_oe_name}: {meta_b_prediction_oe}',
            '-'*100
        ]
        
        log_message = "\n".join(output)
        print(log_message)
        logging.info(log_message)
        
        counter += 1

print(f'Total refined questions: {counter}')
logging.info(f'Total refined questions: {counter}')
print(distrubution)

# Save refined question IDs
# with open("deleted_qid_llava_7b_qwen2_5_7b_llava_72b_internvl_38B_over_2.json", "w") as f:
# with open("internvl_38B_correct_id.json", "w") as f:
#     json.dump(refine_qid, f)
