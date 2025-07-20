#%%
from transformers import AutoModel,AutoTokenizer,AutoConfig

model_name = "/data/shared_workspace/LLM_weights/meta-llama/Meta-Llama-3-8B"
# model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
# print(model)
# config = AutoConfig.from_pretrained(model_name,trust_remote_code = True)
# print(config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
text = ["hello wold","i love you,you are my sunshine"]
tokenizer.pad_token = tokenizer.eos_token
model_input = tokenizer(text, add_special_tokens =True, truncation=True, max_length=512)
print(model_input)
model_input = tokenizer.pad(model_input,return_tensors="pt", padding=True,)
print(model_input)
# %%
