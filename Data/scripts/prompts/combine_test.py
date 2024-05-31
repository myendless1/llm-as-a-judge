import glob
import json
test_files_folder = "/vepfs-sha/liying/LLM_JUDGER/Data/test"
system_prompt = open("system.txt", "r").read()
user_prompt = open("user.txt", "r").read()

test_files = glob.glob(test_files_folder + "/*.jsonl")

for file in test_files:
    test_data = []
    with open(file, "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            _user_prompt = user_prompt.format(
                query=data['query'],
                response1=data['response1'],
                response2=data['response2']
                )
            test_data.append({
                "user_prompt": _user_prompt,
                "gt_label": data["gt_label"]
                })
    combined_json = f"{file.split('/')[-1].split('.')[0]}_openai.json"
    print(test_data[0])
    with open(f"test/{combined_json}", "w") as f:
        for data in test_data:
            f.write(json.dumps(data))
            f.write("\n")