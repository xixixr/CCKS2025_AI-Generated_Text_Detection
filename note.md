# 不同方法测试记录
# 一. SuperAnnotate/ai-detector (bert_based finetuning)
## 环境配置
```shell
# 创建conda 环境
conda create -n xxx python=3.11
conda init
conda activate xxx
# 安装superAnnotate所需环境(换源阿里云)
pip install git+https://github.com/superannotateai/generated_text_detector.git@v1.1.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt
# 测试torch
print(torch.cuda.is_available())
# 安装torch(可选)
pip install torch==2.2.1+cu118  torchaudio==2.2.1+cu118  torchvision==0.17.1+cu118 -i https://mirrors.aliyun.com/pytorch-wheels/

```
## 解释
使用hf模型SuperAnnotate/ai-detector进行微调得到检测结果
### config&&result
|idx |batch_size |lr  |scheduler|epoch |accumulation_steps|loss|acc_local_val |threshold| acc
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|1|5e-5|linear_schedule_with_warmup|5|1|sigmoid+交叉熵|---|0.5|0.5084|
|2|1|1e-5|linear_schedule_with_warmup|1|16|sigmoid+交叉熵|0.9546|0.5195|---|
|3|2|2e-5|linear_schedule_with_warmup|1|8|sigmoid+交叉熵|0.9832|0.5195|0.6541|
|4|16|1e-5|None|1|1|sigmoid+交叉熵|0.9743|0.5195|0.6419|
|5|16|1e-5|None|1|1|MSE|0.9703|0.7995|0.7619|
|6|16|1e-5|None|1|1|MSE|0.9703|0.82|---|
# 二.使用大模型进行微调
## llama-factory的使用方法
### llama-factory环境配置
[llama_factory项目链接](https://github.com/hiyouga/LLaMA-Factory)
```shell
conda create -n llama_factory python=3.11
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
conda activate llama_factory
pip install -e ".[torch,metrics]"
```
### llama-factory数据预处理
可以参照[说明文档](https://llamafactory.readthedocs.io/zh-cn/latest/index.html)的预处理方法进行自制数据集，同时将数据集注册到data_info.json文件中，同时按照指定的训练方式，配置训练的配置文件



## sft + lora微调  
### 实验配置及记录
|method|promt| acc
|:---:|:---:|:---:|
|llama factory|output:AI-generated/Human-generated|0.7194|
|llama factory|output:AI/Human|0.7647|
## 将模型作为特征提取器然后做回归任务
### 使用qwen0.5-0.5B-instruct
#### A榜
default lora config
```python
 lora_config = LoraConfig(r=8, 
 lora_alpha=8, 
 target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
  bias = "none", 
  task_type= TaskType.FEATURE_EXTRACTION
)
 ```
|idx |batch_size |lr  |lora config|epoch |scheduler|loss|threshold| acc|note
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|8|5e-5|default|1|cosine_schedule_with_warmup|MSE|0.513|0.8682|使用last token的hidden_state|
|2|8|5e-5|all|1|cosine_schedule_with_warmup|MSE|0.5|0.7086||
|3|8|5e-5|default|1|cosine_schedule_with_warmup|MSE|0.6|0.8864|使用last token的hidden_state|
#### B榜

|idx |batch_size |lr  |lora config|epoch |scheduler|loss|threshold|train/val loss| acc|note
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|8|5e-5|default|1|cosine_schedule_with_warmup|MSE|0.4905|-|0.6958|使用last token的hidden_state 数据加入prompt|
|2|8|5e-5|default|1|cosine_schedule_with_warmup|MSE|0.513|-|0.7463|使用last token的hidden_state|
|3|8|5e-5|default|1|cosine_schedule_with_warmup|MSE|0.513|-|0.7343|使用最后一层数据的mean_pooling|
|4|8|5e-5|default|1|cosine_schedule_with_warmup|MSE|0.612|0.025/0.003|0.7776|将regression_head 增加一层提升表达能力|
|5|8|5e-5|dropput = 0.1|1|cosine_schedule_with_warmup|MSE|0.401|0.0622/0.0052|0.8185|regression_head update + concat_layers|
|6|8|5e-5|dropput = 0.1 a = 16|1|-|BCEwithlogit|-|0.0407/0.0131|0.6731|regression_head update + concat_layers|
|7|8|5e-5|dropput = 0.1|1|-|BCEwithlogit|-|0.0426/0.0151|-|regression_head update + concat_layers|
|8|8|5e-5|dropput = 0.1 a = 16|1|cosine_schedule_with_warmup|MSE|0.5355|0.0536/0.0044|0.8253|regression_head update + concat_layers2|
|9|8|5e-5|dropput = 0.1 a = 32|1|cosine_schedule_with_warmup|MSE|0.606|0.0466/0.0039|0.8114|regression_head update + concat_layers3|
|10|8|5e-5|dropput = 0.1 r =16 ,a = 32|1|cosine_schedule_with_warmup|MSE|0.5435|0.0495/0.0041|0.8312|regression_head update + concat_layers4|
|11|8|5e-5|dropput = 0.1 a = 16,qkvo|1|cosine_schedule_with_warmup|MSE|-|0.0699/0.0075|-|regression_head update + concat_layers|
|12|8|5e-5|dropput = 0.1 r =16 ,a = 32|1|cosine_schedule_with_warmup|MSE|0.565|0.0952/0.0040|0.8139|regression_head update + concat_layers +fgm对抗训练|
### 使用qwen2.5-7B-instruct
default:concat_layers:[8,16,-1]，使用last_token的hidden_state,使用两层线性层的regression_head 
```python
 lora_config = LoraConfig(
  r=8, 
 lora_alpha=16, 
 target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
  bias = "none", 
  lora_dropout=0.1,
  task_type= TaskType.FEATURE_EXTRACTION
)
 ```
```

 CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file accelerate_config.yaml finetune_lora_accelerate.py 
 ```
|idx |batch_size |lr  |lora config|epoch |scheduler|loss|threshold|train/val loss| acc|note
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|2*4|2e-5|default|1|cosine_schedule_with_warmup|MSE|0.4|0.0198/0.0089|0.7871|default:过拟合|
|2|2*4|2e-5|dropout = 0.2|1|cosine_schedule_with_warmup|MSE|0.563|0.0256/0.0027|0.8410|dropout|
|3|2*4|2e-5|dropout = 0.2 r=16,a=32|1|cosine_schedule_with_warmup|MSE|0.2235|0.0211/0.0094|-|dropout2:过拟合|