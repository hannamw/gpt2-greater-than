#%%
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dataset import YearDataset
from utils import get_valid_years

DEVICE='cuda:0'
#%%
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
N = 400  
ds = YearDataset(years_to_sample_from, N, Path("cache/potential_nouns.txt"), tokenizer, balanced=False, device=DEVICE, eos=True)

# %%
with torch.inference_mode():
    logits = model(ds.good_toks).logits
# %%
year_probs = torch.softmax(logits[:, -2], dim=-1)
topk = torch.topk(year_probs, k=100)
topk_tokens = [[tokenizer._convert_id_to_token(top) for top in ex] for ex in topk.indices]
topk_numbers = torch.tensor([[int(tok[1:]) if tok[1:].isnumeric() else 0 for tok in ex] for ex in topk_tokens], device=DEVICE)
zeros = torch.zeros_like(topk.values, device=DEVICE)
# %%
valid_prob = torch.where(topk_numbers >= ds.years_XX.view(-1,1).to(DEVICE), topk.values, zeros)
print(valid_prob.sum(-1).mean(), topk.values.sum(-1).mean())

# %%
year_prob = torch.where(topk_numbers == ds.years_XX.view(-1,1).to(DEVICE), topk.values, zeros)
print(year_prob.sum(-1).mean(), topk.values.sum(-1).mean())
# %%

year_probs = torch.softmax(logits[:, -1], dim=-1)
topk = torch.topk(year_probs, k=5)
topk_tokens = [[tokenizer._convert_id_to_token(top) for top in ex] for ex in topk.indices]
topk_numbers = torch.tensor([[int(tok) if tok.isnumeric() else 0 for tok in ex] for ex in topk_tokens], device=DEVICE)
zeros = torch.zeros_like(topk.values, device=DEVICE)
valid_prob = torch.where(topk_numbers >= ds.years_YY.view(-1,1).to(DEVICE), topk.values, zeros)
print(valid_prob.sum(-1).mean(), topk.values.sum(-1).mean())
# %%
print((topk_numbers >= ds.years_YY.view(-1,1).to(DEVICE)).float().sum(-1).mean())