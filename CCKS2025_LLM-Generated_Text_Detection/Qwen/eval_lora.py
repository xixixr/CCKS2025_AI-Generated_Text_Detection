#%%
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from peft import PeftModel
from torch.utils.data import DataLoader
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden_size = self.base_model.config.hidden_size
        self.pad_token_id = self.base_model.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.base_model.config.bos_token_id
        self.regression_head = nn.Linear(self.hidden_size, num_labels)
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        hidden_states = output.last_hidden_state
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            last_token_mask = (input_ids != self.pad_token_id) #(batch_size, sequence_length)
        else:
            batch_size = inputs_embeds.shape[0]
            last_token_mask = (inputs_embeds != 0).any(dim=-1)
        if input_ids is None and attention_mask is None:
            raise ValueError("At least one of input_ids or attention_mask must be provided.")
        torch_indices = torch.arange(input_ids.shape[-1], device=input_ids.device)  # (sequence_length)
        last_non_pad_token = (torch_indices * last_token_mask).argmax(dim=-1)  # (batch_size)
        last_token_hidden_states = hidden_states[torch.arange(batch_size), last_non_pad_token]  # (batch_size, hidden_size)
        logits = self.regression_head(last_token_hidden_states)  # (batch_size, num_labels)
        return logits

# 载入模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "/data/shared_workspace/xiarui/huggingface/Qwen/Qwen2.5-0.5B-Instruct"
model = CustomModel(model_name, num_labels=1)


adapter_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/Qwen/Qwen2.5-0.5B-Instruct/finetune_lora/lora"
model.base_model = PeftModel.from_pretrained(model.base_model,adapter_path)

regression_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/Qwen/Qwen2.5-0.5B-Instruct/finetune_lora"
regression_state = torch.load(os.path.join(regression_path, "regression_head.pth"))
model.regression_head.load_state_dict(regression_state)
# 检查模型是否加载成功
if model is None:
    raise ValueError("Failed to load the model. Please check the model path and ensure it exists.")
model.to(device)
# 设置模型为评估模式
model.eval()


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True, padding_side="left")
#%%
# 加载数据集
import datasets
train_data_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/train.jsonl"
test_data_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/test.jsonl"
train_data = datasets.load_dataset("json", data_files={"train": train_data_path}, num_proc=8)["train"]
test_data = datasets.load_dataset("json", data_files={"test": test_data_path}, num_proc=8)["test"]
def map_function(example):
    model_input = tokenizer(
        example["text"],
        add_special_tokens=True
    )
    return {
        "input_ids": model_input["input_ids"],
        "attention_mask": model_input["attention_mask"],
        "label": example["label"]
    }

train_data = train_data.map(map_function,num_proc=8 ,remove_columns=["text"])
from tqdm import tqdm
batch_size = 16

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        features = [{"input_ids":item["input_ids"],
                     "attention_mask":item["attention_mask"]} for item in batch]
        # 使用 tokenizer 进行批量 padding
        batch_padded = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"  # 返回张量而不是列表
        )

        # 添加 label（需手动处理）
        labels = torch.tensor([item["label"] for item in batch])

        batch_padded["labels"] = labels
        return batch_padded
    return collate_fn
        

collate_fn = create_collate_fn(tokenizer)
train_loader = DataLoader(train_data,batch_size = batch_size,shuffle = False,num_workers = 8,collate_fn=collate_fn)
# 进行batch推理

predictions = []
gt = []
process_bar = tqdm(train_loader)
for batch in process_bar:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    with torch.no_grad():
        preds = model(
            input_ids=torch.tensor(input_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device)
        )
    predictions.extend(preds.view(-1).tolist())
    gt.extend(labels)
    
results = []
#%%
# 打印模型结构，查看是否包含 LoRA 层
print(model)

# 检查活动的适配器
print(model.base_model.active_adapters)  # 应该输出 ['default']
#%%
# 构建每条记录
for pred, label in zip(predictions, gt):
    record = {
        "prediction": float(pred),
        "ground_truth": float(label)
    }
    results.append(record)
# 保存预测结果
output_file = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/predictions/train_predictions.jsonl"
# 保存

import json
with open(output_file, 'w', encoding='utf-8') as f:
    for record in results:
        # 确保中文等非ASCII字符正确保存
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"预测结果已保存至: {output_file}")
    
    
# %%
# 测试数据集测试
import datasets
def map_function(example):
    model_input = tokenizer(
        example["text"],
        add_special_tokens=True
    )
    return {
        "input_ids": model_input["input_ids"],
        "attention_mask": model_input["attention_mask"],
    }
def create_collate_fn(tokenizer):
    def collate_fn(batch):
        features = [{"input_ids":item["input_ids"],
                     "attention_mask":item["attention_mask"]} for item in batch]
        # 使用 tokenizer 进行批量 padding
        batch_padded = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"  # 返回张量而不是列表
        )
        return batch_padded
    return collate_fn
batch_size = 8
test_data_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/test.jsonl"
test_data = datasets.load_dataset("json", data_files={"test": test_data_path}, num_proc=8)["test"]
test_data = test_data.map(map_function,num_proc=8 ,remove_columns=["text"])
collate_fn = create_collate_fn(tokenizer)
test_loader = DataLoader(test_data,batch_size = batch_size,shuffle = False,num_workers = 8,collate_fn=collate_fn)
predictions = []
from tqdm import tqdm
process_bar = tqdm(test_loader)
for batch in process_bar:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        preds = model(
            input_ids=torch.tensor(input_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device)
        )
    predictions.extend(preds.view(-1).tolist())
output_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/submit_A/submit.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for pred in predictions:
        if pred >= 0.6: #0.513
            f.write("1\n")
        else:
            f.write("0\n")
# %%
