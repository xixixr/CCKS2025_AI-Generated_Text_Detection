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
from peft import get_peft_model,LoraConfig,TaskType
import torch.distributed as dist

# 设置随机数种子
random_seed = 114514
torch.manual_seed(random_seed)
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.MSELoss()  # 使用均方误差损失函数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num",type=int,default=1,help="Number of GPUs to use (default: 1)")
    parser.add_argument("--batch_size",type=int,default=2,help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=4,help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="Learning rate")
    parser.add_argument("--epochs",type=int,default=1,help="Number of training epochs")
    parser.add_argument("--num_labels",type=int,default=1,help="Number of labels")
    parser.add_argument("--mean_pooling",type=bool,default=False,help="Whether to use mean pooling for the last hidden state")
    parser.add_argument("--concat_layers",type = str,default="3,8,16,-1",help="Use what layers to regression")
    parser.add_argument("--use_FGM",type=bool,default=False,help="Whether to use FGM for adversarial training")
    parser.add_argument("--model_name_or_path", 
                        type=str, 
                        default="/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/model/Qwen/Qwen2.5-7B-Instruct", 
                        help="Pretrained model path")
    parser.add_argument("--train_file", type=str, 
                        default= "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/processed/train.jsonl", 
                        help="Path to training file")
    parser.add_argument("--val_file", 
                        type=str, 
                        default= "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/processed/val.jsonl", 
                        help="Path to validation file")
    parser.add_argument("--output_dir", 
                        type=str, 
                        default="/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/Qwen/Qwen2.5-7B-Instruct/finetune_lora/dropout_fgm/", 
                        help="Directory to save model outputs")
    return parser.parse_args()
args = parse_args()
args.concat_layers = [int(i) for i in args.concat_layers.split(",")]  # 转换为整数列表


class FGM:
    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
        self.emb_name = emb_name

    def attack(self):
        # 遍历模型中的参数，寻找 embedding 层
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is None:
                    continue
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # 恢复参数
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


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
            nn.Dropout(0.2),       # 防过拟合
            # nn.Linear(output_len // 2, self.num_labels)
            nn.Linear(output_len // 2, output_len//4),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(output_len//4, num_labels)
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
    



def evaluate(model, val_loader, writer, step, accelerator):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1), labels.float().view(-1))
            preds = logits.view(-1) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total_loss += accelerator.gather(loss).mean().item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    avg_acc = correct / (num_batches * args.batch_size)
    if accelerator.is_main_process:
        writer.add_scalar("val/loss", avg_loss, step)
        writer.add_scalar("val/acc", avg_acc, step)
    return avg_loss

    
# 训练循环
def train(data_iterator,model,optimizer,scheduler,epochs,writer, val_loader,accelerator,use_fgm,fgm=None):
    model.train()
    train_total_loss = 0
    batches = 0
    for epoch in range(epochs):
        process_bar  = tqdm(data_iterator, desc=f"Epoch {epoch + 1}/{epochs}")
        for step,batch in enumerate(process_bar):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.amp.autocast(device_type='cuda'): # 使用混合精度计算
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits.view(-1), labels.float().view(-1))
                loss = loss/ args.gradient_accumulation_steps  # 梯度累积
            accelerator.backward(loss)
            if use_fgm:
                fgm.attack()  # 添加扰动
                with torch.amp.autocast(device_type='cuda'):
                    logits_adv = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss_adv = loss_fn(logits_adv.view(-1), labels.float().view(-1))
                    loss_adv = loss_adv / args.gradient_accumulation_steps

                accelerator.backward(loss_adv)
                fgm.restore()  # 恢复原参数
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # 更新学习率
                process_bar.set_postfix({"loss": loss.item()})
            if (step+1)%10==0:
                writer.add_scalar("train/loss", loss.item(), epoch * len(data_iterator) + (step+1)//10)
            if use_fgm:
                train_total_loss += (loss.item() + loss_adv.item())
            else:
                train_total_loss += loss.item()
            batches += 1
            if (step+1)%(800)==0:
                val_loss = evaluate(model, val_loader, writer, (step+1)//(800), accelerator)
                model.train()
        accelerator.print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        val_loss = evaluate(model, val_loader, writer,(step+1)//(800)+1 , accelerator)
        accelerator.print(f"Train Loss after Epoch {epoch + 1}: {train_total_loss/batches:.4f}")
        accelerator.print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")
        model.train()
        
    writer.close()  # 关闭TensorBoard记录器    
            
if __name__ == "__main__":
    
    log_dir = args.output_dir + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    # 设置TensorBoard日志记录器
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard日志目录
    # 加载模型和分词器
    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True,trust_remote_code=True,padding_side="left")
    model = CustomModel(model_name,args.num_labels,args.mean_pooling,args.concat_layers)
    model = model.bfloat16() # 使用bfloat16精度
    lora_config = LoraConfig(
        r= 8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha
        target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"], # 需要应用LoRA的模块
        bias = "none",  # LoRA偏置
        lora_dropout=0.2,
        task_type= TaskType.FEATURE_EXTRACTION
    )
    print(f"Model loaded from {model_name}, Finish loading model")
    model.base_model = get_peft_model(model.base_model, lora_config)
    
    
    print("Frozen base model parameters except LoRA layers:")
    # 冻结基础模型的参数
    for name, param in model.base_model.named_parameters():
        if not param.requires_grad:
            continue  # 已是非可训练参数，无需处理
        if "lora" not in name:
            param.requires_grad = False


    # 加载数据集
    train_path = args.train_file
    val_path = args.val_file
    data = datasets.load_dataset("json",data_files={"train":train_path,"val":val_path},num_proc=4)
    train_data = data["train"]
    val_data = data["val"]
    # dataset = TextDataset(train_data,tokenizer)
    # 优化方案
    def map_function(example):
        model_input = tokenizer(
            example["text"],
            add_special_tokens=True,
            max_length=1024,
            truncation=True
        )
        return {
            "input_ids": model_input["input_ids"],
            "attention_mask": model_input["attention_mask"],
            "label": example["label"]
        }
    train_dataset = train_data.map(
        map_function,
        remove_columns=["text"],  # 移除原始文本列
        num_proc=4,  # 使用8个进程进行并行处理
        desc="Processing train data"
    )
    
    collate_fn = create_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle = True,num_workers = 4,collate_fn=collate_fn)
    val_dataset = val_data.map(
    map_function,
    remove_columns=["text"],
    num_proc=4,
    desc="Processing val data"
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    
    
    # 优化器和学习率调度器
    total_step = len(train_loader)*args.epochs//args.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # 使用Accelerator进行分布式训练
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    
    from transformers.optimization import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(0.1*total_step),num_training_steps=total_step)
    accelerator.print("Finish preparing optimizer and scheduler")
    # FGM对抗训练
    use_fgm = args.use_FGM
    if use_fgm:
        fgm = FGM(model, epsilon=1.0, emb_name='embed_tokens')
        print("Using FGM for adversarial training")
    else:
        fgm = None
    # 开始训练
    accelerator.print("Start training...")
    train(train_loader,model,optimizer,scheduler,args.epochs,writer, val_loader,accelerator,use_fgm,fgm)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.base_model.save_pretrained(os.path.join(args.output_dir, "lora"))
        torch.save(unwrapped_model.regression_head.state_dict(), os.path.join(args.output_dir, "regression_head.pth"))

    if accelerator.distributed_type != accelerator.distributed_type.NO:
        accelerator.wait_for_everyone()  # 确保所有进程同步
        dist.destroy_process_group()  # 销毁进程组
