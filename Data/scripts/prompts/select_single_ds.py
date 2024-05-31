import os
import sys
sys.path.append("/vepfs-sha/liying/LLM_JUDGER/Data/scripts/process")
from llm_bar import format_train_data
import glob
import json
from combine import combine

train_data_path = "/vepfs-sha/liying/LLM_JUDGER/Data/train"
single_ds_path = "/vepfs-sha/liying/LLM_JUDGER/Data/single_ds"
panda_lm_ds_name = "panda_lm.jsonl"
auto_j_ds_name = "auto_j.jsonl"
single_ds_item_num = 2500
# copy single ds items number of panda_lm.jsonl and auto_j.jsonl to single_ds_path
os.makedirs(single_ds_path, exist_ok=True)
print("Copying single ds files...")

def get_reversed_data(data):
    explanation = data["explanation"].replace("Assistant B", "Assistant D").replace("Assistant A", "Assistant C")\
        .replace("Assistant C", "Assistant B").replace("Assistant D", "Assistant A")\
        .replace("[[A]]", "[[C]]").replace("[[B]]", "[[A]]").replace("[[C]]", "[[B]]")
    reversed_data = {
        "query": data["query"],
        "response1": data["response2"],
        "response2": data["response1"],
        "explanation": explanation,
        "gt_label": 1 if data["gt_label"] == 0 else 0 if data["gt_label"] == 1 else -1
    }
    return reversed_data

with open(f"{single_ds_path}/{panda_lm_ds_name}", "w") as f:
    panda_lm_lines = open(f"{train_data_path}/{panda_lm_ds_name}", "r").readlines()
    for line in panda_lm_lines[:single_ds_item_num]:
        data = json.loads(line)
        reversed_data = get_reversed_data(data)
        json.dump(data, f)
        f.write("\n")
        json.dump(reversed_data, f)
        f.write("\n")
with open(f"{single_ds_path}/{auto_j_ds_name}", "w") as f:
    auto_j_lines = open(f"{train_data_path}/{auto_j_ds_name}", "r").readlines()
    for line in auto_j_lines[:single_ds_item_num]:
        data = json.loads(line)
        reversed_data = get_reversed_data(data)
        json.dump(data, f)
        f.write("\n")
        json.dump(reversed_data, f)
        f.write("\n")

llm_bar_CoT_path = "/vepfs-sha/liying/LLM_JUDGER/Data/scripts/prompts/llm_bar_cot_raw"
llm_bar_CoT_files = glob.glob(f"{llm_bar_CoT_path}/*.json")

print("Processing llm_bar_cot files...")
llm_bar_cot_data = []
for file in llm_bar_CoT_files:
    with open(file, "r") as f:
        lines = f.read()
        llm_bar_cot_list = json.loads(lines)
        for llm_bar_cot in llm_bar_cot_list:
            formatted_1, formatted_2 = format_train_data(llm_bar_cot)
            if formatted_1 is not None:
              llm_bar_cot_data.append(formatted_1)
            if formatted_2 is not None:
              llm_bar_cot_data.append(formatted_2)

with open(f"{single_ds_path}/llm_bar_cot.jsonl", "w") as f:
    for data in llm_bar_cot_data[:single_ds_item_num]:
        reversed_data = get_reversed_data(data)
        json.dump(data, f)
        f.write("\n")
        json.dump(reversed_data, f)
        f.write("\n")

    
# convert to prompt format
print("Converting to prompt format...")
combine([f"{single_ds_path}/{panda_lm_ds_name}"], "train_openai_panda_lm")
combine([f"{single_ds_path}/{auto_j_ds_name}"], "train_openai_auto_j")
combine([f"{single_ds_path}/llm_bar_cot.jsonl"], "train_openai_llm_bar_cot")