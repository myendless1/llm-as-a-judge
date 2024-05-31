import glob
import json
def combine(train_files_folder_or_files, output_file):
    system_prompt = open("system.txt", "r").read()
    user_prompt = open("user.txt", "r").read()
    assistant_prompt = open("assistant.txt", "r").read()
    if isinstance(train_files_folder_or_files, list):
        train_files = train_files_folder_or_files
    else:
        train_files = glob.glob(train_files_folder_or_files + "/*.jsonl")
    train_data = []
    for file in train_files:
        with open(file, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                _user_prompt = user_prompt.format(
                    query=data['query'].replace("{","{{").replace("}","}}"),
                    response1=data['response1'].replace("{","{{").replace("}","}}"),
                    response2=data['response2'].replace("{","{{").replace("}","}}")
                    )
                _assistant_prompt = assistant_prompt.format(explanation=data['explanation'])
                train_data.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": _user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": _assistant_prompt
                        }
                    ]
                    })
    combined_json = f"{output_file}.json"
    print(train_data[0])
    with open(combined_json, "w") as f:
        json.dump(train_data, f)

if __name__ == "__main__":
    train_files_folder = "/vepfs-sha/liying/LLM_JUDGER/Data/train"
    output_file = "train_openai"
    combine(train_files_folder, output_file)
