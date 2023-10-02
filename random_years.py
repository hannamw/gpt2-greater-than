#%%
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dataset import YearDataset
from utils import year_indices, get_valid_years

DEVICE='cuda:0'
#%%
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
N = 400  
ds = YearDataset(years_to_sample_from, N, Path("cache/potential_nouns.txt"), tokenizer, balanced=False, device=DEVICE, eos=True)

# %%
N_prior = 200
start_years = years_to_sample_from[torch.randint(years_to_sample_from.size(0), (N_prior,))]
years_XX = start_years // 100
years_XX00 = (start_years // 100) * 100
years = []
for XX00 in years_XX00:
    sample_space = years_to_sample_from[(years_to_sample_from >= XX00) & (years_to_sample_from < XX00+100)]
    years.append(sample_space[torch.randint(sample_space.size(0), (5,))])
years = torch.stack(years)
years_YY = years % 100

years_strings = [f'{str(y.tolist())[1:-1]}, {XX}' for y, XX in zip(years, years_XX)]
years_tokens = tokenizer(years_strings, return_tensors="pt")['input_ids'].to(DEVICE)

with torch.inference_mode():
    logits = model(years_tokens).logits[:, -1]
probs = torch.softmax(logits, dim=-1)
year_probs = probs[:, year_indices]
topk = torch.topk(probs, k=5)
topk_tokens = [[tokenizer._convert_id_to_token(top) for top in ex] for ex in topk.indices]

def comp_prob(probs, years_YY, gt=True):
    comps = []
    for prob, year in zip(probs, years_YY[:, -1]):
        if gt:
            comps.append(prob[year+1:].sum())
        else:
            comps.append(prob[:year+1].sum())
    return torch.stack(comps)
print(year_probs[torch.arange(year_probs.size(0)), years_YY[:, -1]+1].mean())
print(comp_prob(year_probs, years_YY).mean())
print(comp_prob(year_probs, years_YY, False).mean())
# %%
i = 4
plt.plot(year_probs[i].cpu())
plt.title(f"GPT-2 Probabilities when YY={years_YY[i].tolist()}")
plt.xlabel(f"Predicted Year")
plt.ylabel(f"probability")
plt.show()
# %%
