from common import conv_judge_chatglm, conv_judge_pair_gemma
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import json
from tqdm import trange
import torch
import glob
import os
import argparse

def parse_label(pred_text):
    """Parse the label from the prediction text."""
    match = re.search(r"Final Verdict: \[\[(.*)\]\]", pred_text)
    switch = {
        "A": 0,
        "B": 1,
        "C": -1,
        "None": -1,
    }
    try:
        return switch[match.group(1)] 
    except:
        return -1

def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    return tokenizer, model

def load_test_data(file_path, st_index, end_index):
    test_data = []
    with open(file_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data[st_index:end_index]

def run_eval(test_data, model, tokenizer, pred_file, conv_judge_pair, device="cuda"):
    print("Running evaluation...")
    pred_labels = []
    batch_size = 4
    template = conv_judge_pair.template
    for i in trange(0, len(test_data), batch_size):
        for index, data in enumerate(test_data[i:i+batch_size]):
            data_sample = conv_judge_pair.system + '\n' + template.format(query=data['query'],
                                            response1=data['response1'],
                                            response2=data['response2'],
                                            ) + conv_judge_pair.appendix
            inputs = tokenizer(data_sample, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            output_ids = model.generate(**inputs, max_new_tokens=2048)
            output = output_ids[0][len(inputs['input_ids'][0]) :]
            pred = tokenizer.decode(output, skip_special_tokens=True, spaces_between_special_tokens=False)
            pred_label = parse_label(pred)
            pred_labels.append(pred_label)
            with open(pred_file, "a") as f:
                item = {}
                item["pred"] = pred
                json.dump(item, f)
                f.write("\n")
                f.close()
    return pred_labels

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="gemma-7b")
    args.add_argument("--model_path", type=str, default="/vepfs-sha/liying/LLM_JUDGER/LLaMA-Factory/saves/gemme-7b/lora/sft/checkpoint-500")
    args.add_argument("--test_file_folder", type=str, default="/vepfs-sha/liying/LLM_JUDGER/Data/test/")
    args.add_argument("--device", type=str, default="cuda:0")
    args.add_argument("--max_test_num", type=int, default=10000)
    args.add_argument("--conv_judge_template", type=str, default="conv_judge_pair_gemma")
    args = args.parse_args()
    MODEL_NAME = args.model_name
    MODEL_PATH = args.model_path
    TEST_FILE_FOLDER = args.test_file_folder
    assert args.conv_judge_template in ["conv_judge_pair_gemma", "conv_judge_chatglm"], "conv_judge_template should be one of ['conv_judge_pair_gemma', 'conv_judge_chatglm']"
    CONV_JUDGE_TEMPLATE = conv_judge_pair_gemma if args.conv_judge_template == "conv_judge_pair_gemma" else conv_judge_chatglm if args.conv_judge_template == "conv_judge_chatglm" else None
    DEVICE = args.device
    MAX_TEST_NUM = args.max_test_num
    test_files = glob.glob(f"{TEST_FILE_FOLDER}*jsonl")
    tokenizer, model = load_model(MODEL_PATH, DEVICE)
    os.makedirs(f"judge_results/{MODEL_NAME}", exist_ok=True)
    for test_file in test_files:
        test_file_name = test_file.split("/")[-1].split(".")[0]
        pred_file = f"judge_results/{MODEL_NAME}/pred_{test_file_name}.jsonl"
        test_data = load_test_data(test_file, 0, MAX_TEST_NUM)
        pred_labels = run_eval(test_data, model, tokenizer, pred_file, CONV_JUDGE_TEMPLATE, DEVICE)
        
        # get the number of 1 in sequential_pred_win_list
        win_num = pred_labels.count(0)
        tie_num = pred_labels.count(-1)
        lose_num = pred_labels.count(1)

        # print the win, tie, and lose number, use format {}
        print("Assistant 1's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(win_num, tie_num, lose_num))
        print("Assistant 2's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}".format(lose_num, tie_num, win_num))
        
        gt_labels = [x["gt_label"] for x in test_data]
        acc = sum([1 for x, y in zip(gt_labels, pred_labels) if x == y]) / len(gt_labels)
        print("Accuracy: ", acc)
        recall = sum([1 for x, y in zip(gt_labels, pred_labels) if x == 0 and y == 0]) / sum([1 for x in gt_labels if x == 0])
        print("Recall: ", recall)
        precision = sum([1 for x, y in zip(gt_labels, pred_labels) if x == 0 and y == 0]) / sum([1 for y in pred_labels if y == 0])
        print("Precision: ", precision)
        f1 = 2 * recall * precision / (recall + precision+1e-8)
        print("F1: ", f1)
        with open(f"judge_results/{MODEL_NAME}/{test_file_name}.txt", "w") as f:
            f.write("Assistant 1's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}\n".format(win_num, tie_num, lose_num))
            f.write("Assistant 2's reuslts ---> win_num: {}, tie_num: {}, lose_num: {}\n".format(lose_num, tie_num, win_num))
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"F1: {f1}\n")
            f.close()



