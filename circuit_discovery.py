#%%
from typing import cast
from pathlib import Path

import torch
import rust_circuit as rc
import matplotlib.pyplot as plt

from rust_circuit.ui import cui
from rust_circuit.ui.very_named_tensor import VeryNamedTensor
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
    logit_lens_ln,
    load_and_split_gpt2,
    show_mtx,
    get_attention_pattern,
    await_without_await,
    show_diffs,
    mean_logit_diff,
    get_valid_years,
    make_all_nodes_names,
    make_extender_factory,
    cutoff_sharpness,
    make_scrubbed_printer,
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
N = 490  
ds = YearDataset(years_to_sample_from, N, Path("cache/potential_nouns.txt"), tokenizer, balanced=True, device=DEVICE, eos=True)

MAX_LEN = ds.good_toks.size(-1)
END_POS = MAX_LEN - 1
XX1_POS = ds.good_prompt.index("XX1")
YY_POS = ds.good_prompt.index("YY")

#%%
# Splitting our model to make it pretty
metric = "prob"
circuit = load_and_split_gpt2(MAX_LEN)
year_indices = torch.load("cache/logit_indices.pt")
ld_circuit, group = load_diff_model(circuit, year_indices, ds.good_mask, device=DEVICE)

#%%
def se(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return transform.sample(c).evaluate()


def sec(c):
    """Short function for Sample,  Evaluate, and collate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return collate(transform.sample(c).evaluate(), ds.years_YY)


# Let's make a copy of the circuit that actually has inputs!
c = ld_circuit.update(
    "tokens",
    lambda _: rc.DiscreteVar(to_device(rc.Array(ds.good_toks, name="tokens"), DEVICE), probs_and_group=group),
)
baseline_mean = se(c).mean()

#%%
# Let's visualize normal model behavior!
probs = torch.softmax(sec(c.get_unique("logits")), dim=-1)[:, -1, year_indices]
# torch.save(probs, "paper-cache/probs.pt")
fig = show_diffs(
    probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="amp",
)
fig.show()
#%%
# let's look at an individual one
i = 39
plt.plot(probs[i].cpu())
plt.title(f"GPT-2 Probabilities when YY={i + 2}")
plt.xlabel(f"Predicted Year")
plt.ylabel(f"probability")
plt.show()

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


# %%
#  Let's see what nodes are important, starting from the root, and looking at all MLPs / attention heads

results = iterative_path_patch([corr_root_matcher], end_pos_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/ipp_logits.pt")
show_mtx(
    results.cpu(),
    title=f"logits",
    color_map_label=f"{metric} diff variation",
)

# %%
#  Let's see what nodes are important, starting from root->m11, and looking at all MLPs / attention heads
m11_extender = extender_factory(MLPHeadAndPosSpec(11, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m11_matcher = m11_extender(corr_root_matcher)
results = iterative_path_patch([m11_matcher], end_pos_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/ipp_mlp11.pt")
show_mtx(
    results.cpu(),
    title=f"m11",  # f"{metric} diff variation m11 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
)

#%%
#  Let's see what nodes are important, starting from root->m11->m10, and looking at all MLPs / attention heads
m10_extender = extender_factory(MLPHeadAndPosSpec(10, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m10_matcher = m10_extender(corr_root_matcher | m11_matcher)
results = iterative_path_patch([m10_matcher], end_pos_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/ipp_mlp10.pt")
show_mtx(
    results.cpu(),
    title=f"m10",  # f"{metric} diff variation m10 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
)
# %%
#  Let's see what nodes are important, starting from root->m11->m10->m9, and looking at all MLPs / attention heads
m9_extender = extender_factory(MLPHeadAndPosSpec(9, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m9_matcher = m9_extender(corr_root_matcher | m11_matcher | m10_matcher)
results = iterative_path_patch([m9_matcher], end_pos_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/ipp_mlp9.pt")
show_mtx(
    results.cpu(),
    title=f"m9",  # f"{metric} diff variation m9 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
)

#%%
#  Let's see what nodes are important, starting from root->m11->m10->m9->m8, and looking at all MLPs / attention heads
m8_extender = extender_factory(MLPHeadAndPosSpec(8, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
m8_matcher = m8_extender(corr_root_matcher | m11_matcher | m10_matcher | m9_matcher)
results = iterative_path_patch([m8_matcher], end_pos_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/ipp_mlp8.pt")
show_mtx(
    results.cpu(),
    title=f"m8",  # f"{metric} diff variation m8 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
)

# %%
# What other heads could be important? Let's look at the attention patterns of all heads to find out
# First get the attention patterns for 20 examples
heads = [(i, j) for i in range(12) for j in range(12)]
n_examples = 20
attn_patterns = get_attention_pattern(to_device(circuit, DEVICE), heads, ds.good_toks[:n_examples]).cpu()

# Then visualize them (maybe take the mean over sentences)
await_without_await(lambda: cui.init(port=6781))
attn_pattern_vnt = VeryNamedTensor(
    attn_patterns,
    dim_names="head sentence queries keys".split(),
    dim_types="example example axis axis".split(),
    dim_idx_names=[
        heads,
        [f"seq {i}" for i in range(n_examples)],
        ds.good_prompt,
        ds.good_prompt,
    ],
    title="Attention patterns",
)
await_without_await(lambda: cui.show_tensors(attn_pattern_vnt))

#%%
attn_patterns_7_10 = get_attention_pattern(to_device(circuit, DEVICE), [(7, 10)], ds.good_toks).cpu()[0]
attention_collated = collate(attn_patterns_7_10, ds.years_YY)
torch.save(attention_collated, "paper-cache/attn_collated.pt")
mean_attn_patterns_7_10 = attn_patterns_7_10.mean(0)
attn_patterns_8_11 = get_attention_pattern(to_device(circuit, DEVICE), [(8, 11)], ds.good_toks).cpu()[0]
mean_attn_patterns_8_11 = attn_patterns_8_11.mean(0)
torch.save(attn_patterns_7_10, "paper-cache/attn_patterns_a7.h10.pt")
torch.save(attn_patterns_8_11, "paper-cache/attn_patterns_a8.h1.pt")

#%%
# So what do these heads do? We can examine this question with logit lens (or DPP)
module = "a7.h10"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
torch.save(logits, "paper-cache/logit_lens_a7.h10.pt")
show_diffs(logits, title=f"Logit lens of {module}")
#%%
module = "a8.h11"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
torch.save(logits, "paper-cache/logit_lens_a8.h11.pt")
show_diffs(logits, title=f"Logit lens of {module}")
#%%
module = "a9.h1"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
show_diffs(logits, title=f"Logit lens of {module}")

#%%
module = "a11.h0"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
show_diffs(logits - logits[:, 0:1], title=f"Logit lens of {module}")


#%%
# So what do these heads do? We can examine this question with logit lens (or DPP)
module = "m8"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
torch.save(logits, "paper-cache/logit_lens_mlp8.pt")
show_diffs(logits - logits[:, 0:1], title=f"Logit lens of {module}").show()
print(module, "logit mean diff", mean_logit_diff(logits), "cutoff sharpness", cutoff_sharpness(logits).mean())
#%%
module = "m9"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
torch.save(logits, "paper-cache/logit_lens_mlp9.pt")
show_diffs(logits - logits[:, 0:1], title=f"Logit lens of {module}").show()
print(module, "logit mean diff", mean_logit_diff(logits), "cutoff sharpness", cutoff_sharpness(logits).mean())
#%%
module = "m10"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
torch.save(logits, "paper-cache/logit_lens_mlp10.pt")
show_diffs(logits - logits[:, 0:1], title=f"Logit lens of {module}").show()
print(module, "logit mean diff", mean_logit_diff(logits), "cutoff sharpness", cutoff_sharpness(logits).mean())
#%%
module = "m11"
logits = sec(logit_lens_ln(c, module, device=DEVICE))
torch.save(logits, "paper-cache/logit_lens_mlp11.pt")
sd = show_diffs(logits - logits[:, 0:1], title=f"Logit lens of {module}").show()
print(module, "logit mean diff", mean_logit_diff(logits), "cutoff sharpness", cutoff_sharpness(logits).mean())

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

#%%
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
# Now, let's find the rest of the circuit!
# what's important to the values of our attention heads
yy_pos_q_matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), YY_POS), qkv="q")
    for l in range(12)
    for h in list(range(12)) + ["mlp"]
]

results = iterative_path_patch([ms], yy_pos_q_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/attn_q.pt")
#%%
show_mtx(
    results.cpu(),
    title=f"nodes important to attention heads' query vectors",  # f"{metric} diff variation m8 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
).show()

# %%
# Now, let's find the rest of the circuit!
# what's important to the values of our attention heads
yy_pos_k_matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), YY_POS), qkv="k")
    for l in range(12)
    for h in list(range(12)) + ["mlp"]
]

results = iterative_path_patch([ms], yy_pos_k_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/attn_k.pt")
#%%
show_mtx(
    results.cpu(),
    title=f"nodes important to attention heads' key vectors",  # f"{metric} diff variation m8 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
).show()
# %%
# Now, let's find the rest of the circuit!
# what's important to the values of our attention heads
yy_pos_v_matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), YY_POS), qkv="v")
    for l in range(12)
    for h in list(range(12)) + ["mlp"]
]

results = iterative_path_patch([ms], yy_pos_v_matcher_extenders, ds.bad_toks)
torch.save(results, "paper-cache/attn_v.pt")
#%%
show_mtx(
    results.cpu(),
    title=f"nodes important to attention heads' value vectors",  # f"{metric} diff variation m8 (patch data: {alt_tok_name}-dataset)",
    color_map_label=f"{metric} diff variation",
).show()

# %%
yy_pos_matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), YY_POS), qkv=None)
    for l in range(12)
    for h in list(range(12)) + ["mlp"]
]

for i in range(4):
    mlp_matcher = extender_factory(MLPHeadAndPosSpec(i, "mlp", YY_POS), qkv="v")(ms)
    mlp_results = iterative_path_patch([mlp_matcher], yy_pos_matcher_extenders, ds.bad_toks)
    torch.save(results, f"paper-cache/results_mlp{i}.pt")
    show_mtx(
        mlp_results.cpu(),
        title=f"nodes important to m{i}",  # f"{metric} diff variation m8 (patch data: {alt_tok_name}-dataset)",
        color_map_label=f"{metric} diff variation",
    ).show()
#%%
def embed_extender(m: rc.IterativeMatcher):
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
lms = embed_extender(running) #| embed_extender(ms)
for i in range(4):
    running = lower_extenders2(running)
    lms = lms | embed_extender(running)  # running.chain('embeds')
lms = lms | ms.chain(rc.restrict({"a.q"}, term_if_matches=True, end_depth=8))
#%%
patched_circuit = path_patching(
    ld_circuit,
    ds.bad_toks,  # unpatched nodes get bad data
    ds.good_toks,  # patched nodes get good data
    lms,#whole_circuit_matchers,
    group,
    "tokens",
)
printer = make_scrubbed_printer(*patched_circuit.get("tokens"))

whole_circuit_results = se(patched_circuit)
print(whole_circuit_results.mean())
print(whole_circuit_results.mean() / se(c).mean())
print(whole_circuit_results.mean() / ms_patched_results)
# %%
probs = torch.softmax(sec(patched_circuit.get_unique("logits")), dim=-1)[:, -1, year_indices]
fig = show_diffs(
    probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="amp",
)
fig.show()
# %%
