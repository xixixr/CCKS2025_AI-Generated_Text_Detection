#%%
import datasets
import json
from tqdm import tqdm
test_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/dataset/test.jsonl"
test_data = datasets.load_dataset("json", data_files={"test":test_path}, num_proc=8)

from util_xr import make_promt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
# from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_name = "/data/shared_workspace/xiarui/huggingface/Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code =True,
    local_files_only = True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only = True,trust_remote_code =True,)


# 加载lora模型参数
# model = PeftModel.from_pretrained(model,"/data/workspace/xiarui/workspace/huggingface/custom/Qwen/Qwen2.5-0.5B-Instruct/lora/sft")
model.load_adapter("/data/shared_workspace/xiarui/huggingface/custom/Qwen/Qwen2.5-0.5B-Instruct/lora/sft_2")
#%%

batch_size = 1

# 推理结果
infernce_results = []
responses = []
# 按照批次大小进行推理
for start in tqdm(range(0, len(test_data["test"]), batch_size),ncols=100, desc="Inference batches"):
    end = min(start + batch_size, len(test_data["test"]))
    batch_data = test_data["test"].select(range(start,end))
    # 准备所有样本的输入文本
    batch_texts = []
    for item in batch_data:
        messages = [
            {"role": "user", "content": make_promt() + "\n" + item["text"]}
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
    ).to(model.device)

    # 使用模型进行推理
    with torch.no_grad():
        # 批量生成回复
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            num_return_sequences=1,  # 每个输入只生成一个回复
            pad_token_id=tokenizer.eos_token_id # 确保填充标记正确设置
        )

        # 使用batch_decode一次性解码所有生成的回复
        input_lengths = [len(ids) for ids in model_inputs.input_ids]
        generated_ids = [ids[input_len:] for ids, input_len in zip(generated_ids, input_lengths)]
        response_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # 保留模型回答进行查看
    for response in response_batch:
        responses.append(response)
    
    # 解析模型回答
    for response in response_batch:
        if response == "AI":
            infernce_results.append(1)
        else :
            infernce_results.append(0)
# 保存回复到文件中去
response_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/response.txt"
with open(response_path,"w") as f:
    for response in responses:
        f.write(f"{response}\n")
# 观察输出的具体情况
set_response= set(responses)
print(set_response)
print(len(set_response))

# # 保存推理结果
output_path = "/data/workspace/xiarui/tianchi_LMTextDetect/CCKS2025_LLM-Generated_Text_Detection/output/submit_A/submit.txt"
# 写入文本文件,每行一个结果
with open(output_path,"w") as f:
    for result in infernce_results:
        f.write(f"{result}\n")