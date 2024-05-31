import os
import glob
from combine import combine
train_data_path = "/vepfs-sha/liying/LLM_JUDGER/Data/train"
half_panda_train_data_path = "/vepfs-sha/liying/LLM_JUDGER/Data/train_half_panda"
train_files = glob.glob(f"{train_data_path}/*.jsonl")
no_panda_lm_train_files = [train_file for train_file in train_files if "panda_lm" not in train_file]
panda_lm_file = [train_file for train_file in train_files if "panda_lm" in train_file][0]
print(no_panda_lm_train_files)
print(panda_lm_file)

# copy train_files to half_panda_train_data_path
os.makedirs(half_panda_train_data_path, exist_ok=True)
print("Copying train files...")
for train_file in no_panda_lm_train_files:
    os.system(f"cp {train_file} {half_panda_train_data_path}")

# select half of the lines in panda_lm.jsonl
import random
panda_lm_lines = open(panda_lm_file, "r").readlines()
print("Raw panda_lm.jsonl length:", len(panda_lm_lines))
random.shuffle(panda_lm_lines)
panda_lm_lines = panda_lm_lines[:len(panda_lm_lines)//2]
print("Half panda_lm.jsonl length:", len(panda_lm_lines))

# write to half_panda_train_data_path
with open(f"{half_panda_train_data_path}/panda_lm.jsonl", "w") as f:
    for line in panda_lm_lines:
        f.write(line)

# combine all files in half_panda_train_data_path
print("Combining files...")
combine(half_panda_train_data_path, "train_openai_half_panda")