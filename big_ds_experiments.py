#%%
from typing import cast, List
from pathlib import Path

import torch
import rust_circuit as rc
import matplotlib.pyplot as plt
from tqdm import tqdm

from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher

from dataset import YearDataset
from utils import (
    collate,
    HeadOrMlpType, 
    MLPHeadAndPosSpec,
    load_diff_model,
    load_gpt2_small_circuit,
    path_patching,
    to_device,
    load_and_split_gpt2,
    make_extender_factory,
    show_diffs,
    get_valid_years,
    prob_diff,
    cutoff_sharpness,
    make_all_nodes_names,
)

#%%
# Loading our base model
DEVICE = "cuda:0"
_, tokenizer, _ = load_gpt2_small_circuit()

#%%
# Creating our dataset
years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
N = 10000  
ds = YearDataset(
    years_to_sample_from,
    N,
    Path("cache/potential_nouns.txt"),
    tokenizer,
    balanced=False,
    eos=True,
    device=DEVICE,
)

MAX_LEN = ds.good_toks.size(-1)
END_POS = MAX_LEN - 1
XX1_POS = ds.good_prompt.index("XX1")
YY_POS = ds.good_prompt.index("YY")
last_two_digits = ds.years_YY

#%%
# Splitting our model to make it pretty
metric = "prob"
circuit = load_and_split_gpt2(MAX_LEN)
year_indices = torch.load("cache/logit_indices.pt")


def se(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    c = c.update("year_logits", lambda x: rc.batch_to_concat(x, axis=0, batch_size=100))
    return transform.sample(c).evaluate().cpu()

#%%
# manual batching
batch_size = 200
logits = []
for i in tqdm(range(N // batch_size)):
    # Let's make a copy of the circuit that actually has inputs!
    ld_circuit, group = load_diff_model(
        circuit, year_indices, ds.good_mask[i * batch_size : (i + 1) * batch_size], device=DEVICE
    )
    c = ld_circuit.update(
        "tokens",
        lambda _: rc.DiscreteVar(
            to_device(rc.Array(ds.good_toks[i * batch_size : (i + 1) * batch_size], name="tokens"), DEVICE),
            probs_and_group=group,
        ),
    )
    logits.append(se(c.get_unique("logits"))[:, -1].cpu())
logits = torch.stack(logits).view(N, -1)
#%%
probs = torch.softmax(logits, dim=-1)[:, year_indices]
torch.save(collate(probs, ds.years_YY), "paper-cache/probs.pt")
diffs = prob_diff(probs, ds.years_YY)
print(diffs.mean(), diffs.std())

sharpness = cutoff_sharpness(probs, ds.years_YY)
print(sharpness.mean(), sharpness.std())
#%%
yearwise_probs = collate(probs, ds.years_YY)

fig = show_diffs(
    yearwise_probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="amp",
)
fig.show()
#%%
# Defining the matchers


extender_factory = make_extender_factory(MAX_LEN)
end_pos_matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), END_POS), qkv=None)
    for l in range(12)
    for h in list(range(12)) + ["mlp"]
]

mlp_set_extender = extender_factory(
    {MLPHeadAndPosSpec(i, cast(HeadOrMlpType, "mlp"), END_POS) for i in range(8, 12)}, qkv=None
)
attention_set_extenders = extender_factory(
    {
        MLPHeadAndPosSpec(layer, head, END_POS)
        for layer, head in [(9, 1), (8, 11), (7, 10), (6, 9), (5, 5), (8, 8), (5, 1)]
    }
)
running = corr_root_matcher
ms = attention_set_extenders(corr_root_matcher)
for i in range(4):
    running = mlp_set_extender(running)
    ms = ms | attention_set_extenders(running)

#%%
batch_size = 200
patched_probs = []
for i in tqdm(range(N // batch_size)):
    # Let's make a copy of the circuit that actually has inputs!
    ld_circuit, group = load_diff_model(
        circuit, year_indices, ds.good_mask[i * batch_size : (i + 1) * batch_size], device=DEVICE
    )
    patched_circuit = path_patching(
        ld_circuit,
        ds.bad_toks[i * batch_size : (i + 1) * batch_size],  # unpatched nodes get bad data
        ds.good_toks[i * batch_size : (i + 1) * batch_size],  # patched nodes get good data
        ms,
        group,
        "tokens",
    )
    patched_probs.append(torch.softmax(se(patched_circuit.get_unique("logits"))[:, -1], dim=-1)[:, year_indices].cpu())
patched_probs = torch.stack(patched_probs).view(N, -1)


torch.save(collate(patched_probs, ds.years_YY), "paper-cache/patched_probs.pt")
#%%
patched_diffs = prob_diff(patched_probs, ds.years_YY)
mean_diffs = 0.817
print(patched_diffs.mean())
print(patched_diffs.mean() / mean_diffs)

patched_sharpness = cutoff_sharpness(patched_probs, ds.years_YY)
mean_sharpness = 0.059
print(patched_sharpness.mean())
print(patched_sharpness.mean() / mean_sharpness)

#%%
yearwise_patched_probs = []
for year in range(2, 99):
    yearwise_patched_probs.append(patched_probs[ds.years_YY == year].mean(0))
yearwise_patched_probs = torch.stack(yearwise_patched_probs)

fig = show_diffs(
    yearwise_patched_probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="Patched GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="Blues",
)
fig.update_layout(margin=dict(l=0, r=0, b=10, t=30), title_x=0.5, height=500, width=600)
fig.show()
fig.write_image("paper-plots/patched-probability-heatmap.pdf")
# %%

reverse_patched_circuit = path_patching(
    ld_circuit,
    ds.good_toks,  # unpatched nodes get bad data
    ds.bad_toks,  # patched ndoes get good data
    ms,
    group,
    "tokens",
)

reverse_patched_results = se(reverse_patched_circuit)
reverse_patched_mean = reverse_patched_results.mean()
print(reverse_patched_mean)
# %%
reverse_patched_probs = torch.softmax(se(reverse_patched_circuit.get_unique("logits")), dim=-1)[:, -1, year_indices]
reverse_patched_diffs = prob_diff(reverse_patched_probs, ds.years_YY)
print(reverse_patched_diffs.mean())
print(reverse_patched_diffs.mean() / diffs.mean())

# %%
def embed_extender(m: rc.IterativeMatcher) -> rc.IterativeMatcher:
    return m.chain(rc.new_traversal(start_depth=0, end_depth=1)).chain(
        rc.new_traversal(start_depth=1, end_depth=2).chain(
            rc.restrict(
                rc.Matcher("embeds"),
                term_early_at=rc.Matcher(make_all_nodes_names(MAX_LEN)),
                term_if_matches=True,
            )
        )
    )


lower_extenders = extender_factory(
    {
        MLPHeadAndPosSpec(layer, head, YY_POS)
        for layer, head in [(3, "mlp"), (2, "mlp"), (1, "mlp"), (0, "mlp"), (0, 5), (0, 3), (0, 1)]
    }
)
lower_extenders2 = extender_factory(
    {
        MLPHeadAndPosSpec(layer, head, YY_POS)
        for layer, head in [(3, "mlp"), (2, "mlp"), (1, "mlp"), (0, "mlp"), (0, 1)]
    }
)
running = lower_extenders(ms)
lms = embed_extender(running)  # running.chain('embeds')
for i in range(4):
    running = lower_extenders2(running)
    lms = lms | embed_extender(running)  # running.chain('embeds')
lms = lms | ms.chain(rc.restrict({"a.q"}, term_if_matches=True, end_depth=8))

#%%
batch_size = 400
patched_probs = []
for i in tqdm(range(N // batch_size)):
    # Let's make a copy of the circuit that actually has inputs!
    ld_circuit, group = load_diff_model(
        circuit, year_indices, ds.good_mask[i * batch_size : (i + 1) * batch_size], device=DEVICE
    )
    patched_circuit = path_patching(
        ld_circuit,
        ds.bad_toks[i * batch_size : (i + 1) * batch_size],  # unpatched nodes get bad data
        ds.good_toks[i * batch_size : (i + 1) * batch_size],  # patched nodes get good data
        lms,
        group,
        "tokens",
    )
    patched_probs.append(torch.softmax(se(patched_circuit.get_unique("logits"))[:, -1], dim=-1)[:, year_indices].cpu())
patched_probs = torch.stack(patched_probs).view(N, -1)


torch.save(collate(patched_probs, ds.years_YY), "paper-cache/full_patched_probs.pt")