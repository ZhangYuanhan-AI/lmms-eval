import pandas as pd
import json

# Load JSONL file
jsonl_file_path = "/opt/tiger/lmms-eval/logs/20250220_112911_samples_hardvideo_all.jsonl"
csv_file_path = "/opt/tiger/02_20_from_yx.csv"  # Replace with your actual CSV file path
output_csv_path = "/opt/tiger/02_20_from_yx_with_score.csv"  # Path to save the updated CSV

# Read JSONL file
with open(jsonl_file_path) as f:
    lines = f.readlines()
    results = [json.loads(line) for line in lines]

# Create a mapping of qid to score
qid_2_predict = {
    item["doc"]["qid"]: int(item["hardvideo_open_ended_score"]["correctness"] >= 4)
    for item in results
}

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Add a new column "score" based on qid mapping
df["score"] = df["row_id"].map(qid_2_predict).fillna(0).astype(int)  # Default to 0 if qid not found

# Save the updated DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Updated CSV saved to {output_csv_path}")
