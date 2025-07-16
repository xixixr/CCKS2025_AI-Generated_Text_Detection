#%%
# 将训练集进行处理成指定格式符合大模型微调
import datasets
import json
import os
from tqdm import tqdm

train_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/train.jsonl"
output_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/train_processed.jsonl"
train_data = datasets.load_dataset("json", data_files=train_path, split="train",num_proc=8)

#%%
train_data

#%%

from util_xr import map_function
# 使用map函数处理数据集
train_data_processed = train_data.map(map_function, num_proc=8)
keep_columns = ["instruction", "input", "output"]
# 保留指定的列
removed_columns = [col for col in train_data_processed.column_names if col not in keep_columns]
train_data_processed = train_data_processed.remove_columns(removed_columns)
# %%
# 保存处理后的数据集到指定路径
with open(output_path, "w") as f:
    for data in tqdm(train_data_processed):
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
# %%
