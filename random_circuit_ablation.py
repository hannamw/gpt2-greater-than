#%%
from typing import cast
from pathlib import Path

import torch
import rust_circuit as rc
from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher

from dataset import YearDataset
from utils import (
    collate,
    HeadOrMlpType, 
    MLPHeadAndPosSpec,
    load_gpt2_small_circuit,
    load_diff_model,
    iterative_path_patching_nocorr,
    path_patching,
    to_device,
    load_and_split_gpt2,
    show_diffs,
    get_valid_years,
    make_extender_factory,
    get_valid_years,
)

#%%
# Loading our base model
DEVICE = "cuda:0"
MODEL_ID = "gelu_12_tied"  # aka gpt2 small
_, tokenizer, _ = load_gpt2_small_circuit() 

#%%
# Creating our dataset
years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
N = 200  
ds = YearDataset(years_to_sample_from, N, Path("cache/potential_nouns.txt"), balanced=True, device=DEVICE, eos=True)

MAX_LEN = ds.good_toks.size(-1)
END_POS = MAX_LEN - 1
XX1_POS = ds.good_prompt.index("XX1")
YY_POS = ds.good_prompt.index("YY")
last_two_digits = ds.years_YY

#%%
# Splitting our model to make it pretty
circuit = load_and_split_gpt2(MAX_LEN)
year_indices = torch.load("cache/logit_indices.pt")
ld_circuit, group = load_diff_model(circuit, year_indices, ds.good_mask, device=DEVICE)

#%%
def se(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return transform.sample(c).evaluate()


def sec(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return collate(transform.sample(c).evaluate(), ds.years_YY)


# Let's make a copy of the circuit that actually has inputs!
c = ld_circuit.update(
    "tokens",
    lambda _: rc.DiscreteVar(to_device(rc.Array(ds.good_toks, name="tokens"), DEVICE), probs_and_group=group),
)
baseline_mean = se(c).mean()


#%%
# We need to make an extender factory, and then some matcher extenders to iteratively path patch with
extender_factory = make_extender_factory(MAX_LEN)
end_pos_matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), END_POS), qkv=None)
    for l in range(12)
    for h in list(range(12)) + ["mlp"]
]


def iterative_path_patch(matchers_to_extend, matcher_extenders, patch_data):
    """Calls iterative path patching, keeping the baseline / patch data, group, input_name, and output_name constant"""
    return (
        iterative_path_patching_nocorr(
            circuit=ld_circuit,
            matchers_to_extend=matchers_to_extend,
            baseline_data=ds.good_toks,
            patch_data=patch_data,
            group=group,
            matcher_extenders=matcher_extenders,
            input_name="tokens",
            output_shape=(12, 13, -1),
        ).mean(-1)
    ) - baseline_mean


#%%
m11_extender = extender_factory(MLPHeadAndPosSpec(11, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m11_matcher = m11_extender(corr_root_matcher)
m10_extender = extender_factory(MLPHeadAndPosSpec(10, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m10_matcher = m10_extender(corr_root_matcher | m11_matcher)
m9_extender = extender_factory(MLPHeadAndPosSpec(9, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m9_matcher = m9_extender(corr_root_matcher | m11_matcher | m10_matcher)
m8_extender = extender_factory(MLPHeadAndPosSpec(8, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m8_matcher = m8_extender(corr_root_matcher | m11_matcher | m10_matcher | m9_matcher)


mlp_set_extender = extender_factory(
    {MLPHeadAndPosSpec(i, cast(HeadOrMlpType, "mlp"), END_POS) for i in range(10,11)}, qkv=None
)

heads2 = [(9, 2), (8, 0), (7, 11), (6, 10), (5, 6), (8, 9), (5, 2)]
heads_orig = [(9, 1), (8, 11), (7, 10), (6, 9), (5, 5), (8, 8), (5, 1)]
heads = heads_orig

attention_set_extenders = extender_factory(
    {
        MLPHeadAndPosSpec(layer, head, END_POS)
        for layer, head in heads
    }
)
running = corr_root_matcher
ms = attention_set_extenders(corr_root_matcher)
for i in range(4):
    running = mlp_set_extender(running)
    ms = ms | attention_set_extenders(running)


patched_circuit = path_patching(
    ld_circuit,
    ds.bad_toks,  # unpatched nodes get bad data
    ds.good_toks,  # patched ndoes get good data
    ms,
    group,
    "tokens",
)

patched_results = se(patched_circuit).mean()
ms_patched_results = patched_results
print(patched_results, baseline_mean, patched_results / baseline_mean)


probs = torch.softmax(sec(patched_circuit.get_unique("logits")), dim=-1)[:, -1, year_indices]
fig = show_diffs(
    probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="GPT-2 Small Probability Heatmap (Patched)",
    zlabel="probability",
    color_continuous_scale="amp",
)
fig.show()


# %%
