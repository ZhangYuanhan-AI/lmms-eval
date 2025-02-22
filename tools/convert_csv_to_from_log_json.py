import pandas as pd
import copy
import os
import json

FROM_LOG_TASK_TEMPLATE_JSON = {
    "model_configs": {
        "task": ""
    },
    "logs": []
}


csv_file_path = "/opt/tiger/02_21.csv"
eval_set = "hardvideo_all" #csv_file_path.split("/")[-1].split(".")[0]

# 读取 CSV 文件
df = pd.read_csv(csv_file_path)

# save_question_type = "Non-leading Open-ended Question" 
# save_question_type = "Multiple-choice Question with a Single Correct Answer"
# save_question_type = "Paraphrased Open-ended Question"
# save_question_type = "Correctly-led Open-ended Question"
# save_question_type = "Wrongly-led Open-ended Question"
save_question_type = "all"

task_json = copy.deepcopy(FROM_LOG_TASK_TEMPLATE_JSON)
task_json["model_configs"]["task"] = eval_set
assert 'pred' in df.columns, "pred column not found in eval set"
mc_results = {}
for index, row in df.iterrows():
    # 检查是否存在'doc_id'列
    if save_question_type != "all":
        if row["question_type"] != save_question_type:
            continue
    if 'doc_id' in df.columns:
        doc_id = int(row['doc_id'])
    else:
        doc_id = index  # 使用行索引作为doc_id
    task_json["logs"].append({
        "doc_id": doc_id,
        "resps": [
            [
                row['pred']
            ]
        ],
    })
    # import pdb;pdb.set_trace()
    if row["question_type"] == "Multiple-choice Question with a Single Correct Answer":
        mc_results[doc_id] = row['answer']
task_json_path = os.path.join("/opt/tiger/yx", f"{eval_set}.json")
mc_gt_json_path = os.path.join("/opt/tiger/yx", f"{eval_set}_mc_gt.json")
with open(task_json_path, 'w') as f:
    f.write(json.dumps(task_json, indent=4))

with open(mc_gt_json_path, 'w') as f:
    f.write(json.dumps(mc_results, indent=4))





