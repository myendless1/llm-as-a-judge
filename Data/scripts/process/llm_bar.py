# example train data
"""
  {
    "input": "What does the phrase \"smaller fish to fry\" mean?",
    "output_1": "The phrase \"bigger fish to fry\" is an idiomatic expression that means having more important or pressing matters to attend to. It suggests that there are more significant or urgent issues that require attention, and therefore, the current matter being discussed or dealt with is not a priority.",
    "output_2": "The phrase \"smaller fish to fry\" refers to prioritizing smaller or simpler tasks before tackling larger, more complicated ones. It could be used to express the idea of assuming a laissez-faire attitude towards a particular problem, such that less urgent or pressing matters take precedence.",
    "label": 2,
    "results": [
      {
        "swap = False": {
          "completion": [
            "Output (a) does not correctly execute the instruction as it explains the phrase \"bigger fish to fry\" instead of \"smaller fish to fry\". On the other hand, Output (b) accurately explains the meaning of the phrase \"smaller fish to fry\" as per the instruction. Therefore, Output (b) is better.",
            "stop"
          ],
          "winner": "2"
        },
        "swap = True": {
          "completion": [
            "Output (a) correctly explains the meaning of the phrase \"smaller fish to fry\" as per the instruction. On the other hand, Output (b) explains the phrase \"bigger fish to fry\" which is not what the instruction asked for. Therefore, Output (a) is better.",
            "stop"
          ],
          "winner": "2"
        }
      }
    ]
  },
"""
import json
from sklearn.model_selection import train_test_split

train_file_paths = {
    "GPTInst":"/vepfs-sha/liying/LLM_JUDGER/Data/raw/llm_bar/GPTInst/GPT-4-CoT/result.json",
    "GPTOut":"/vepfs-sha/liying/LLM_JUDGER/Data/raw/llm_bar/GPTOut/GPT-4-CoT/result.json",
    "Manual":"/vepfs-sha/liying/LLM_JUDGER/Data/raw/llm_bar/Manual/GPT-4-CoT/result.json",
    "Neighbor":"/vepfs-sha/liying/LLM_JUDGER/Data/raw/llm_bar/Neighbor/GPT-4-CoT/result.json",
    "Natural":"/vepfs-sha/liying/LLM_JUDGER/Data/raw/llm_bar/Natural/GPT-4-CoT/result.json",
    }
output_folder = "/vepfs-sha/liying/LLM_JUDGER/Data/"


def format_train_data(raw_data):
    query = raw_data["input"]
    response1 = raw_data["output_1"]
    response2 = raw_data["output_2"]
    gt_label = raw_data["label"] - 1
    gpt4_label = int(raw_data["results"][0]["swap = False"]["winner"]) - 1 if raw_data["results"][0]["swap = False"]["winner"] is not None else None
    if gt_label == gpt4_label:
      explanation = raw_data["results"][0]["swap = False"]["completion"][0]
      explanation = explanation.replace("Output (a)", "Assistant A's response").replace("Output (b)", "Assistant B's response")
      if gt_label == 0:
          explanation = explanation + " \n\nFinal Verdict: [[A]]"
      elif gt_label == 1:
          explanation = explanation + " \n\nFinal Verdict: [[B]]"
      else:
          print(explanation)
          raise ValueError("Final decision not found in explanation")
      formatted_1 = {
          "query": query,
          "response1": response1,
          "response2": response2,
          "explanation": explanation,
          "gt_label": gt_label,
      }
    else:
        formatted_1 = None
    gpt4_label = int(raw_data["results"][0]["swap = True"]["winner"]) - 1 if raw_data["results"][0]["swap = True"]["winner"] is not None else None
    if gt_label == gpt4_label:
      explanation = raw_data["results"][0]["swap = True"]["completion"][0]
      explanation = explanation.replace("Output (a)", "Assistant A's response").replace("Output (b)", "Assistant B's response")
      gt_label = 1 - gt_label
      if gt_label == 0:
          explanation = explanation + " \n\nFinal Verdict: [[A]]"
      elif gt_label == 1:
          explanation = explanation + " \n\nFinal Verdict: [[B]]"
      else:
          print(explanation)
          raise ValueError("Final decision not found in explanation")
          
      formatted_2 = {
          "query": query,
          "response1": response2,
          "response2": response1,
          "explanation": explanation,
          "gt_label": gt_label,
      }
    else:
        formatted_2 = None
    return formatted_1, formatted_2

def format_test_data(raw_data):
    query = raw_data["input"]
    response1 = raw_data["output_1"]
    response2 = raw_data["output_2"]
    gt_label = raw_data["label"] - 1
    explanation = raw_data["results"][0]["swap = False"]["completion"][0]
    explanation = explanation.replace("Output (a)", "Assistant A's response").replace("Output (b)", "Assistant B's response")
    if gt_label == 0:
        explanation = explanation + " \n\nFinal Verdict: [[A]]"
    elif gt_label == 1:
        explanation = explanation + " \n\nFinal Verdict: [[B]]"
    else:
        print(explanation)
        raise ValueError("Final decision not found in explanation")
    formatted_1 = {
        "query": query,
        "response1": response1,
        "response2": response2,
        "explanation": explanation,
        "gt_label": gt_label,
    }
    explanation = raw_data["results"][0]["swap = True"]["completion"][0]
    explanation = explanation.replace("Output (a)", "Assistant A's response").replace("Output (b)", "Assistant B's response")
    gt_label = 1 - gt_label
    if gt_label == 0:
        explanation = explanation + " \n\nFinal Verdict: [[A]]"
    elif gt_label == 1:
        explanation = explanation + " \n\nFinal Verdict: [[B]]"
    else:
        print(explanation)
        raise ValueError("Final decision not found in explanation")
        
    formatted_2 = {
        "query": query,
        "response1": response2,
        "response2": response1,
        "gt_label": gt_label,
    }
    return formatted_1, formatted_2

def read_train_file(file_path):
    train_data = []
    test_data = []
    with open(file_path, "r") as f:
        line = f.read()
        data_list = json.loads(line)
        train_list, test_list = train_test_split(data_list, test_size=0.1, random_state=42)
        for data in train_list:
            formatted_1, formatted_2 = format_train_data(data)
            if formatted_1 is not None:
              train_data.append(formatted_1)
            if formatted_2 is not None:
              train_data.append(formatted_2)
        for data in test_list:
            formatted_1, formatted_2 = format_test_data(data)
            if formatted_1 is not None:
              test_data.append(formatted_1)
            if formatted_2 is not None:
              test_data.append(formatted_2)
    return train_data, test_data

for key, train_file_path in train_file_paths.items():
    formatted_train_data, formatted_test_data = read_train_file(train_file_path)
    with open(output_folder + f"train/llm_bar_{key}.jsonl", "w") as f:
        for data in formatted_train_data:
            json.dump(data, f)
            f.write("\n")

    with open(output_folder + f"test/llm_bar_{key}.jsonl", "w") as f:
        for data in formatted_test_data:
            json.dump(data, f)
            f.write("\n")