import torch
import argparse
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from generated_text_detector.utils.preprocessing import preprocessing_text
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import os
from tqdm import tqdm
import json
import pandas as pd
from torch.cuda.amp import GradScaler, autocast  # 导入混合精度工具
# 设置随机数种子
random_seed = 42
torch.manual_seed(random_seed)


# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--save_dir", type=str, default="./output", help="Directory to save models")
    return parser.parse_args()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = preprocessing_text(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, accumulation_steps, save_dir, writer):
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}',ncols=100)
        for batch in progress_bar:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            # 混合精度训练上下文
            with autocast():
                _, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            
                loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1), labels)
                 # 梯度累积
                loss = loss / accumulation_steps
            # 缩放梯度并反向传播
            loss.backward()
            if (global_step + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 记录学习率和损失
                writer.add_scalar('train/loss', loss.item(), (global_step + 1) //accumulation_steps)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], (global_step + 1) //accumulation_steps)
                # 更新参数
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})
            global_step += 1
            
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                _, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1), labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(logits.squeeze()) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print('Best model saved!')
        

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    print('Training complete! Final model saved!')
    writer.close()

def main():
    args = parse_args()
    
    root_dir = "C:/AI_NLP/tianchi_LMTextDetect/"
    src_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(src_dir, "output", "logs", "runs")
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard日志目录
    mode_save_dir = os.path.join(src_dir,"output","saved_models")
    # 检查并创建保存目录
    os.makedirs(mode_save_dir, exist_ok=True)
    model_path = os.path.join(root_dir,"ai-detector")
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model = RobertaClassifier.from_pretrained(model_path, local_files_only=True)
    model = model.bfloat16() # 使用bfloat16精度
    # 检查模型是否加载成功
    if model is None:
        raise ValueError("Failed to load the model. Please check the model path and ensure it exists.")
    # 设置设备
    if torch.cuda.is_available():
        # 指定使用的GPU设备
        if args.gpu_num > 1:
            # 检查实际可用的GPU数量
            available_gpus = torch.cuda.device_count()
            if args.gpu_num > available_gpus:
                print(f"Warning: Only {available_gpus} GPUs available, using {available_gpus} instead of {args.gpu_num}")
                args.gpu_num = available_gpus
            
            print(f"Using {args.gpu_num} GPUs!")
            device_ids = list(range(args.gpu_num))  # 使用前N个GPU
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        if args.gpu_num > 0:
            print("Warning: CUDA not available, using CPU instead")

    model.to(device)

    # 读取数据文件
    train_file = os.path.join(root_dir, "CCKS2025_LLM-Generated_Text_Detection/dataset/train.jsonl")
    df_train = pd.read_json(train_file, lines=True)
    texts = df_train['text'].tolist()
    labels = df_train['label'].tolist()
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # 创建数据集和数据加载器
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, args.max_length)
    full_data = ConcatDataset([train_dataset, val_dataset])
    
    full_loader = DataLoader(full_data, batch_size=args.batch_size, shuffle=True) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 准备优化器和学习率调度器
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = (len(train_loader)) * args.epochs // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,  # 10% of total steps for warmup
        num_training_steps=total_steps
    )

    # 开始训练
    train_model(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, args.accumulation_steps, mode_save_dir, writer)

if __name__ == "__main__":
    main()