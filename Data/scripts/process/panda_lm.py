# example train data
"""
{
    "input_sequence": "Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n
    ### Instruction:\nGive three tips for staying healthy.\n\n
    ### Response 1:\n1. Eat a balanced and nutritious diet.\n2. Get regular exercise.\n3. Get enough sleep.\n\n
    ### Response 2:\n1. Eat a balanced diet with plenty of fruits, vegetables, and whole grains.\n2. Get regular physical activity, such as walking, jogging, or swimming.\n3. Get enough sleep and practice healthy sleeping habits.\n\n
    ### Evaluation:\n", 
    "output_sequence": "2\n\n### Reason: Response 2 is better because it provides more specific and detailed tips for staying healthy.\n\n### Reference: 1. Eat a balanced diet with plenty of fruits, vegetables, and whole grains. 2. Get regular physical activity, such as walking, jogging, or swimming. 3. Get enough sleep and practice healthy sleeping habits.\n"
    },
"""
"""

"""
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split

train_file_path = "/vepfs-sha/liying/LLM_JUDGER/Data/raw/panda_lm/pandalm_train.json"
test_file_path = "/vepfs-sha/liying/LLM_JUDGER/Data/raw/panda_lm/testset-v1.json"
output_folder = "/vepfs-sha/liying/LLM_JUDGER/Data/"


def format_train_data(raw_data):
    input_sequence = raw_data["input_sequence"]
    _, instruction, response1, response2, _ = re.split("### Instruction:|### Response 1:|### Response 2:|### Evaluation:", input_sequence)
    instruction, response1, response2 = map(lambda x: re.sub(r'\#{3,3}', "", x).strip(), [instruction, response1, response2])
    explanation = raw_data["output_sequence"]
    label, explanation, reference = re.split("### Reason:|### Reference:", explanation)
    label, explanation, reference = map(lambda x: x.strip(), [label, explanation, reference])
    label = int(label.strip()) - 1 if "Tie" not in label else -1
    if label == 0:
        explanation = explanation + " \n\nFinal Verdict: [[A]]"
    elif label == 1:
        explanation = explanation + " \n\nFinal Verdict: [[B]]"
    elif label == -1:
        explanation = explanation + " \n\nFinal Verdict: [[C]] for a tie."
    else:
        print(explanation)
        raise ValueError("Final decision not found in explanation")
    
    explanation = explanation.replace("Response 1", "Assistant A's response").replace("Response 2", "Assistant B's response")
    return {
        "query": instruction,
        "response1": response1,
        "response2": response2,
        "explanation": explanation,
        "gt_label": label,
    }

def format_test_data(raw_data):
    prompt = raw_data["instruction"]
    input = raw_data["input"]
    response1 = raw_data["response1"]
    response2 = raw_data["response2"]
    annotator1 = raw_data["annotator1"]
    annotator2 = raw_data["annotator2"]
    annotator3 = raw_data["annotator3"]
    uni, index, counts = np.unique([annotator1, annotator2, annotator3], return_index=True, return_counts=True)
    gt_label = int(uni[index[np.argmax(counts)]]) - 1
    return {
        "query": f"{prompt}\n\n{input}",
        "response1": response1,
        "response2": response2,
        "gt_label": gt_label,
    }

def read_train_file(file_path):
    train_data = []
    with open(file_path, "r") as f:
        line = f.readline()
        data_list = json.loads(line)
        for data in data_list:
            formatted = format_train_data(data)
            train_data.append(formatted)
    return train_data

def read_test_file(file_path):
    test_data = []
    with open(file_path, "r") as f:
        line = f.read().strip()
        data_list = json.loads(line)
        for data in data_list:
            test_data.append(format_test_data(data))
    return test_data

formatted_train_data = read_train_file(train_file_path)
formatted_test_data = read_test_file(test_file_path)

with open(output_folder + "train/panda_lm.jsonl", "w") as f:
    for data in formatted_train_data:
        json.dump(data, f)
        f.write("\n")

with open(output_folder + "test/panda_lm.jsonl", "w") as f:
    for data in formatted_test_data:
        json.dump(data, f)
        f.write("\n")
