# example train data
"""
You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: 
******* ALL OF MY LOADS REQUIRE MACRO POINT TRACKING=******
 
Pu:  st charles il 
Del:  bucyrus oh 
 
Rdy:  1000et today
Del:  by 2200 tonight or 0800 et 1-5 let me kno= 
12 at 17400lbs no stack
48x42x20
Let me know
John

Extract pickup_date, pickup_time, drop_date, drop_time, quantity, weight, stackable_flag, dimensions from the above email body
***
[Response 1]: The email appears to be in the format of a technical document or manual, and it includes information such as a macro point tracking system and instructions on how to use it. The information provided is organized into sections, including a "Pu" for "Problem," "Del" for "Description," "Rdy" for "Required Data," "Del" for "Data Download," "Extract Pickup Date," "Pickup Time," "Quantity," "Weight," "Stackable Flag," and "Dimensions." These sections are followed by a series of instructions and a complete explanation of what the system is and how to use it. The instructions suggest that the macro point tracking system can be used to complete the tasks set by the system.
***
[Response 2]: The pickup date is 10/01/2021, the pickup time is 17:40, the drop date is 10/02/2021, the drop time is 08:00, the quantity is 12, the weight is 17400 lbs, the stackable flag is false, and the dimensions are 48x42x20.
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.

"""
example_test_json = {
    "scenario": "post_summarization", 
    "label": 1, 
    "prompt": "Summarize the following post"
        "Title: I [21f] don't like to take my sister [14f] to the dog park with me. How can I explain this to my parents [45f, 47m]?"
        "Post: I [21f] don't like to take my sister [14f] to the dog park with me. How can I explain this to my parents [45f, 47m]? "
        "I do not live with my parents, but I see them on the regular. They live near my apartment and there is a dog park near them. It is one of the few good ones and I have been going there for a few years. "
        "I have a special needs sister who loves dogs. They are her favorite thing in the world. Dogs hate my sister and want nothing to do with her. She is loud, makes sudden movements, and throws herself at any strange dog she sees. "
        "I don't want to bring her anymore. I have friends I meet there, she is bad with dogs, and it is not fun when she is around. I don't want to be a caregiver and it turns time spent with my sister into a super unpleasant experience. "
        "I am not sure how to explain this to my parents. I have stopped bringing my dogs by because my sister is unable to handle pets. They had to rehome a cat she accidentally hurt. "
        "There are some other issues, but this is the main one. How can I explain to them that I am no longer taking my sister to the dog park? And how can I put up boundaries that are healthy?", 
    "response 1": " I [21f] don't like to take my sister [14f] to the dog park with me. How can I explain this to my parents [45f, 47m]?", 
    "response 2": " I don't like take my disabled sister to the dog park bc she is unable to handle dogs properly. Need boundaries set on our spending time together. How to explain to parents? "
    }
 

import re
import json

# prompt_path = "/vepfs-sha/liying/LLM_JUDGER/Data/scripts/prompts/system_prompt.txt"
# prompt_str = open(prompt_path, "r").read()
train_file_path = "/vepfs-sha/liying/LLM_JUDGER/Data/raw/auto_judge/pairwise_traindata.jsonl"
test_file_path = "/vepfs-sha/liying/LLM_JUDGER/Data/raw/auto_judge/testdata_pairwise.jsonl"

output_folder = "/vepfs-sha/liying/LLM_JUDGER/Data/"


def format_train_data(raw_data):
    usermsg = raw_data["usrmsg"]
    begin, _, query, response1, response2, end = re.split("\[BEGIN DATA\]|\[Response 1\]:|\[Response 2\]:|\[END DATA\]|\[Query\]:", usermsg)
    query, response1, response2 = map(lambda x: re.sub(r'\*{3,3}', "", x).strip(), [query, response1, response2])
    query = query.replace("[Query]: ", "").strip()
    response1 = response1.replace("[Response 1]: ", "").strip()
    response2 = response2.replace("[Response 2]: ", "").strip()

    explanation = raw_data["target_output"].strip()
    if "So, the final decision is Response 1" in explanation:
        explanation = explanation + " \n\nFinal Verdict: [[A]]"
    elif "So, the final decision is Response 2" in explanation:
        explanation = explanation + " \n\nFinal Verdict: [[B]]"
    elif "So, the final decision is Tie" in explanation:
        explanation = explanation + " \n\nFinal Verdict: [[C]] for a tie."
    elif "So, the final decision is neither Response 1 nor Response 2" in explanation:
        explanation = explanation + " \n\nFinal Verdict: [[C]] for a tie."
    else:
        print(explanation)
        raise ValueError("Final decision not found in explanation")
    
    explanation = explanation.replace("Response 1", "Assistant A's response").replace("Response 2", "Assistant B's response")
    gt_label = raw_data["gt_label"] if raw_data["gt_label"] < 2 else -1
    return {
        "query": query,
        "response1": response1,
        "response2": response2,
        "explanation": explanation,
        "gt_label": gt_label,
    }

def format_test_data(raw_data):
    prompt = raw_data["prompt"]
    response1 = raw_data["response 1"]
    response2 = raw_data["response 2"]
    return {
        "query": prompt,
        "response1": response1,
        "response2": response2,
        "gt_label": raw_data["label"] if raw_data["label"] < 2 else -1,
    }

def read_train_file(file_path):
    train_data = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            formatted = format_train_data(data)
            if not formatted:
                exit(0)
            train_data.append(formatted)
    return train_data

def read_test_file(file_path):
    test_data = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            test_data.append(format_test_data(data))
    return test_data

formatted_train_data = read_train_file(train_file_path)
formatted_test_data = read_test_file(test_file_path)

with open(output_folder + "train/auto_j.jsonl", "w") as f:
    for data in formatted_train_data:
        json.dump(data, f)
        f.write("\n")

with open(output_folder + "test/auto_j.jsonl", "w") as f:
    for data in formatted_test_data:
        json.dump(data, f)
        f.write("\n")
