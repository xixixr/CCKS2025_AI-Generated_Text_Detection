
import datasets
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/train_processed.jsonl"
train_data = datasets.load_dataset("json", data_files=train_path, split="train",num_proc=8)

model_name = "/data/workspace/xiarui/workspace/huggingface/Qwen/Qwen2.5-0.5B-Instruct"
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only = True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only = True)

# 随机采样恰好20条数据
train_data_select = train_data.shuffle(seed=114514).select(range(20))

# 准备所有样本的输入文本
batch_texts = []
for item in train_data_select:
    messages = [
        {"role": "user", "content": item["instruction"] + "text:" + item["input"]+"\n"+
         "Reply only \"AI-generated\" or \"Human-generated\""+ "\n"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    batch_texts.append(text)

# 批量编码所有输入文本
model_inputs = tokenizer(
    batch_texts, 
    return_tensors="pt", 
    padding=True, 
    truncation=True, 
    max_length=1024 
).to(model.device)

# 使用模型进行推理
with torch.no_grad():
    # 批量生成回复
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        num_return_sequences=1,  # 每个输入只生成一个回复
        pad_token_id=tokenizer.eos_token_id  # 确保填充标记正确设置
    )

    # 使用batch_decode一次性解码所有生成的回复
    input_lengths = [len(ids) for ids in model_inputs.input_ids]
    generated_ids = [ids[input_len:] for ids, input_len in zip(generated_ids, input_lengths)]
    response_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# 打印对比结果
for i, (item, response) in enumerate(zip(train_data_select, response_batch)):
    print(f"Sampled Data{i+1}: {item['output']}")
    print(f"Model Response{i+1}: {response}")
    print("-" * 50)

