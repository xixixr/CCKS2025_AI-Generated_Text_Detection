#%%
from transformers import AutoModel,AutoTokenizer

model_name = "/data/shared_workspace/xiarui/huggingface/Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModel.from_pretrained(model_name,local_files_only=True)

#%%
for name, module in model.named_modules():
    print(name)
# %%
tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
texts = ["Hello world!", "This is a longer sentence."]
inputs = [tokenizer(text) for text in texts]


inputs_pad = tokenizer.pad(inputs,padding = True,return_tensors="pt")
print(inputs,inputs_pad)
# %%
import torch
x = torch.randn(3,9,2)
z = [1,2,5]
y = x[torch.arange(x.shape[0]),z]
print(x,y.shape)
# %%
print(torch.arange(12).shape)
# %%
