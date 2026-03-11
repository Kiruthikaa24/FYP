from transformers import AutoModel
import torch
model=AutoModel.from_pretrained('../save_dir',torch_dtype=torch.float16,device_map="cuda:0")
print(model)
w = model.layers[0].self_attn.q_proj.weight

print(w.min(), w.max())
print(torch.unique(w[:100]))