#%%
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from peft import PeftModel
from torch.utils.data import DataLoader
from typing import Optional

mean_pooling=False
concat_layers = [6,12,-1]
use_BCE = False  # 是否使用BCE损失函数
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels,mean_pooling,concat_layers):
        super().__init__()
        self.num_labels = num_labels
        self.mean_pooling = mean_pooling
        self.concat_layers = concat_layers
        self.base_model = AutoModel.from_pretrained(model_name,local_files_only = True)
        self.hidden_size = self.base_model.config.hidden_size
        self.pad_token_id = self.base_model.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.base_model.config.bos_token_id
        output_len = len(self.concat_layers)*self.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(output_len, output_len // 2),
            nn.SiLU(),             # 与主模型激活一致
            nn.Dropout(0.1),       # 防过拟合
            nn.Linear(output_len // 2, self.num_labels)
            )
        # self.regression_head = nn.Linear(self.hidden_size,self.num_labels)
    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None):

        
        output = self.base_model(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds = inputs_embeds,output_hidden_states = True)
        hidden_states = output.hidden_states
        selected_states = [hidden_states[i] for i in self.concat_layers] # list consist of (batch_size,seq_len,hidden_size)
        hidden_states = output.last_hidden_state #(batch_size,seq_len,hidden_size)
        if self.mean_pooling:
            pooled_output = []
            for hidden_states in selected_states:
                masked_hidden = hidden_states*attention_mask.unsqueeze(-1)#(batch_size,hidden_size)
                pooled_output.append(masked_hidden.sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)) # list of (batch_size,hidden_size)
            pooled_output = torch.cat(pooled_output,dim=-1) #(batch_size,hidden_size*len(self.concat_layers))
            logits = self.regression_head(pooled_output) #(batch_size,num_labels)
            return logits
        # find the last token hidden state
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            non_pad_mask = (input_ids != self.pad_token_id) # (batch_size,seq_len)
        else:
            batch_size = inputs_embeds.shape[0]
            non_pad_mask = attention_mask.bool()
        if input_ids is None and attention_mask is None:
            raise ValueError("At least one of input_ids or attention_mask must be provided.")
        seq_len = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[1]
        token_indices = torch.arange(seq_len,device=input_ids.device) # (seq_len)
        last_non_pad_token = (token_indices*non_pad_mask).argmax(dim = -1) #(batch_size)
        last_token_hidden_states = []
        for hidden_states in selected_states:
            if hidden_states.shape[1] <= last_non_pad_token.max():
                raise ValueError("The last non-pad token index exceeds the sequence length of the hidden states.")
            last_token_hidden_states.append(hidden_states[torch.arange(batch_size),last_non_pad_token])  # list of (batch_size,hidden_size)
        last_token_hidden_states = torch.cat(last_token_hidden_states,dim=-1) # (batch_size,hidden_size*len(self.concat_layers))
        logits = self.regression_head(last_token_hidden_states) # (batch_size,num_labels)
        return logits
    
# 载入模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "/data/shared_workspace/xiarui/huggingface/Qwen/Qwen2.5-0.5B-Instruct"
model = CustomModel(model_name, num_labels=1,mean_pooling=mean_pooling,concat_layers=concat_layers)

path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/Qwen/Qwen2.5-0.5B-Instruct/finetune_lora/regression_update_concat_layers_fgm"
adapter_path = path +"/lora"
print(f"载入模型为{adapter_path.split('/')[-2]}")
model.base_model = PeftModel.from_pretrained(model.base_model,adapter_path)

regression_path = path
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
train_data = datasets.load_dataset("json", data_files={"train": train_data_path}, num_proc=8)["train"]
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
batch_size = 8

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
    if use_BCE:
        preds = torch.sigmoid(preds).view(-1)  # 使用sigmoid函数将输出转换为概率
    else:
        preds = preds.view(-1)
    predictions.extend(preds.tolist())
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
test_data = datasets.load_dataset("json", data_files={"test": test_data_path}, num_proc=4)["test"]
test_data = test_data.map(map_function,num_proc=8 ,remove_columns=["text"])
collate_fn = create_collate_fn(tokenizer)
test_loader = DataLoader(test_data,batch_size = batch_size,shuffle = False,num_workers =4,collate_fn=collate_fn)
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
    if use_BCE:
        preds = torch.sigmoid(preds).view(-1)
    else:
        preds = preds.view(-1)
    predictions.extend(preds.tolist())
output_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/submit_B/submit.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for pred in predictions:
        if pred >= 0.565:
            f.write("1\n")
        else:
            f.write("0\n")
# %%
