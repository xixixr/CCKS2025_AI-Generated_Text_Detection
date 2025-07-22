# 加载命令行参数,可调整的超参数
import argparse
from transformers import AutoTokenizer,AutoModel
import datasets
import torch
from torch.utils.data import Dataset,DataLoader
import os
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# 设置随机数种子
random_seed = 114514
torch.manual_seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num",type=int,default=1,help="Number of GPUs to use (default: 1)")
    parser.add_argument("--batch_size",type=int,default=8,help="Batch size for training")
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="Learning rate")
    parser.add_argument("--epochs",type=int,default=1,help="Number of training epochs")
    parser.add_argument("--num_labels",type=int,default=1,help="Number of labels")
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self,train_data,tokenizer):
        super().__init__()
        self.texts = train_data["text"]
        self.tokenizer = tokenizer
        self.labels = train_data["label"]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        model_input = self.tokenizer(
            self.texts[index],
            add_special_tokens = True,
            # padding = True,
            # return_tensors = "pt" # 为了之后的batch处理能够得到正确的padding,返回list[int],而不是batchtensor
            )
        return {
            "input_ids": model_input["input_ids"],
            "attention_mask": model_input["attention_mask"],
            "label": self.labels[index] 
        }
        
def create_collate_fn(tokenizer):
    def collate_fn(batch):
        # 将 input_ids 和 attention_mask 提取出来作为字典列表
        features = [{
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"]
        } for item in batch]

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

class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(model_name,local_files_only = True)
        self.hidden_size = self.base_model.config.hidden_size
        self.pad_token_id = self.base_model.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.base_model.config.bos_token_id
        self.regression_head = nn.Linear(self.hidden_size,self.num_labels)
    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None):
        output = self.base_model(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds = inputs_embeds)
        hidden_states = output.last_hidden_state
        # find the last token hidden state
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            non_pad_mask = (input_ids != self.pad_token_id) # (batch_size,seq_len)
        else:
            batch_size = inputs_embeds.shape[0]
            non_pad_mask = attention_mask.bool()
        if input_ids is None and attention_mask is None:
            raise ValueError("At least one of input_ids or attention_mask must be provided.")
        token_indices = torch.arange(input_ids.shape[-1],device=input_ids.device) # (seq_len)
        last_non_pad_token = (token_indices*non_pad_mask).argmax(dim = -1) #(batch_size)
        last_token_hidden_states = hidden_states[torch.arange(batch_size),last_non_pad_token] # (batch_size,hidden_size)
        logits = self.regression_head(last_token_hidden_states) # (batch_size,num_labels)
        return logits
    
# 训练循环
def train(data_iterator,model,optimizer,scheduler,epochs,writer):
    model.train()
    for epoch in range(epochs):
        process_bar  = tqdm(data_iterator, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in process_bar:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.amp.autocast(device_type='cuda'): # 使用混合精度计算
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.mse_loss(logits.view(-1), labels.float().view(-1))  # 使用均方误差损失
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # 更新学习率
            writer.add_scalar("train/loss", loss.item(), epoch * len(data_iterator) + process_bar.n)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch * len(data_iterator) + process_bar.n)
            process_bar.set_postfix({"loss": loss.item()})
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    writer.close()  # 关闭TensorBoard记录器    
            
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = parse_args()
    log_dir = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/Qwen/Qwen2.5-0.5B-Instruct/finetune_all/logs"
    os.makedirs(log_dir, exist_ok=True)
    # 设置TensorBoard日志记录器
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard日志目录
    # 加载模型和分词器
    model_name = "/data/shared_workspace/xiarui/huggingface/Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True,trust_remote_code=True,padding_side="left")
    model = CustomModel(model_name,args.num_labels)
    print(f"Model loaded from {model_name}, Finish loading model")
    
    
    # 加载数据集
    train_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/train.jsonl"
    train_data = datasets.load_dataset("json",data_files={"train":train_path})["train"]
    # dataset = TextDataset(train_data,tokenizer)
    # 优化方案
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
    dataset = train_data.map(
        map_function,
        remove_columns=["text"],  # 移除原始文本列
        num_proc=8,  # 使用8个进程进行并行处理
        desc="Processing train data"
    )
    
    collate_fn = create_collate_fn(tokenizer)
    train_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle = True,num_workers = 8,collate_fn=collate_fn)
    print(f"Dataset size: {len(train_loader)}, Finish loading dataset")
    
    
    
    # 优化器和学习率调度器
    total_step = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 使用Accelerator进行分布式训练
    from accelerate import Accelerator
    accelerator = Accelerator()
    model,optimizer,train_loader = accelerator.prepare(model,optimizer,train_loader)
    
    from transformers.optimization import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(0.1*total_step*args.epochs),num_training_steps=total_step*args.epochs)
    print("Finish preparing optimizer and scheduler")
    # 开始训练
    print("Start training...")
    train(train_loader,model,optimizer,scheduler,args.epochs,writer)
    # 保存
    save_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/Qwen/Qwen2.5-0.5B-Instruct/finetune_all"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + "/model.pth")
    # tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")