#%%
from pathlib import Path

import torch
import plotly.express as px
import rust_circuit as rc

from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher

from dataset import YearDataset
from utils import (
    load_gpt2_small_circuit,
    load_diff_model,
    MLPHeadAndPosSpec,
    path_patching,
    load_and_split_gpt2,
    make_extender_factory,
    show_diffs,
    split_mlp,
    replace_inputs,
    get_valid_years,
)

# %%
DEVICE = "cuda:0"
_, tokenizer, _ = load_gpt2_small_circuit()

#%%
years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
N = 490  
ds = YearDataset(years_to_sample_from, N, Path("cache/potential_nouns.txt"), ordered=True, device=DEVICE, eos=True)

MAX_LEN = ds.good_toks.size(-1)
extender_factory = make_extender_factory(MAX_LEN)
END_POS = MAX_LEN - 1
XX1_POS = ds.good_prompt.index("XX1")
YY_POS = ds.good_prompt.index("YY")
last_two_digits = ds.years_YY

#%%
logit_diff = True
metric = "logit mean" if logit_diff else "prob"
circuit = load_and_split_gpt2(MAX_LEN)
year_indices = torch.load("cache/logit_indices.pt")
ld_circuit, group = load_diff_model(circuit, year_indices, ds.good_mask, logit_diff=logit_diff, device=DEVICE)
# %%
c = replace_inputs(ld_circuit, ds.good_toks, "tokens", corr_root_matcher, group)
d = replace_inputs(ld_circuit, ds.bad_toks, "tokens", corr_root_matcher, group)

def se(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return transform.sample(c).evaluate()


#%%
extender_factory = make_extender_factory(MAX_LEN)
layer = "m10"
direct_layer_extender = extender_factory(MLPHeadAndPosSpec(int(layer[1:]), "mlp", END_POS))
baseline_logits = se(c.get_unique("year_logits"))
weights = c.get_unique(f"{layer}.w.proj_out").value
unembed_matrix = c.get_unique("t.w.tok_embeds").value
activations = se(c.get_unique(rc.IterativeMatcher(layer).chain(rc.restrict("m.act", term_if_matches=True))))[:, -1]
order = torch.load('cache/order.pt')
direct_effects = []
outer_products = []

# We save the top 11 instead of top 10, because the plot looks nicer with one extra neuron
for i, neuron in enumerate(order[:11]):
    # getting effects the outer product way
    acts = activations[:, neuron]
    logit_lens_weights = torch.einsum("h,lh->l", weights[:, neuron], unembed_matrix[year_indices])
    outer_product = torch.einsum("y,l -> yl", acts, logit_lens_weights)
    outer_products.append(outer_product.cpu())

    # getting results the direct effects way
    split_circuit = c.update(layer, lambda node: split_mlp(node, torch.tensor([neuron])))
    patched_circuit = path_patching(
        split_circuit,
        ds.good_toks,
        ds.bad_toks,
        direct_layer_extender(corr_root_matcher).chain("m.pre_important"),
        group,
        "tokens",
    )
    patched_logits = se(patched_circuit.get_unique("year_logits"))
    logit_diff = baseline_logits - patched_logits - baseline_logits[:, 0:1] + patched_logits[:, 0:1]
    direct_effects.append(logit_diff.cpu())

outer_products = torch.stack(outer_products)
direct_effects = torch.stack(direct_effects)
torch.save(outer_products,'paper-cache/t10-logitlens.pt')
torch.save(direct_effects,'paper-cache/t10-direct_effects.pt')
#%%
top = 10
extender_factory = make_extender_factory(MAX_LEN)
layer = "m10"
direct_layer_extender = extender_factory(MLPHeadAndPosSpec(int(layer[1:]), "mlp", END_POS))
baseline_logits = se(c.get_unique("year_logits"))
split_circuit = c.update(layer, lambda node: split_mlp(node, torch.tensor(order[:top])))
patched_circuit = path_patching(
    split_circuit,
    ds.good_toks,
    ds.bad_toks,
    direct_layer_extender(corr_root_matcher).chain("m.pre_important"),
    group,
    "tokens",
)
patched_logits = se(patched_circuit.get_unique("year_logits"))
patched_logit_diff = baseline_logits - patched_logits - baseline_logits[:, 0:1] + patched_logits[:, 0:1]
show_diffs(
        patched_logit_diff.cpu(), title=f"Top {top} patched direct contributions", center_zero=True
    )

#%%
torch.save(patched_logit_diff, 'paper-cache/t10-t10patched.pt')

#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

top2_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"MLP 10 Neuron {i} Contributions" for i in order[:2]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
top2_fig.add_trace(go.Heatmap(z=direct_effects[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
top2_fig.add_trace(go.Heatmap(z=direct_effects[0].cpu(), coloraxis="coloraxis1"), row=1, col=2)

top2_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Logit Change",
        colorbar_title_side="right",
    ),
)
top2_fig.update_yaxes(autorange="reversed")
top2_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
top2_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
top2_fig.show()
top2_fig.write_image("paper-plots/top2-neurons.pdf")

#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

summed_logit_diff = outer_products[:10].sum(0)

top10_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"MLP 10 Neuron {i} Contributions" for i in order[:2]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
top10_fig.add_trace(go.Heatmap(z=patched_logit_diff.cpu(), coloraxis="coloraxis1"), row=1, col=1)
top10_fig.add_trace(go.Heatmap(z=summed_logit_diff.cpu(), coloraxis="coloraxis1"), row=1, col=2)

top10_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Logit Change",
        colorbar_title_side="right",
        cmin=-3,
        cmax=3,
    ),
)
top10_fig.update_yaxes(autorange="reversed")
top10_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
top10_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
top10_fig.show()
top10_fig.write_image("paper-plots/top10-patched-summed.pdf")
#%%
y_labels = list(range(2,99))
title = None  # "MLP Iterative Path Patching"
full_fig = px.imshow(
    outer_products[3:11].cpu(),
    aspect="equal",
    facet_col_spacing=0,
    facet_row_spacing=0,
    facet_col=0,
    facet_col_wrap=4,
    labels=dict(x="Predicted Year", y="YY"),
    title=title,
#    x=x_labels,
    y=y_labels,
    range_color=(-5.0, 5.0),
    zmin=-1.0,
    zmax=1.0,
    color_continuous_scale="RdBu",
)
full_fig.update_layout(
    margin=dict(l=0, r=0, b=30, t=20),
    coloraxis_colorbar_x=1.0,
)

full_fig.update_layout(title_x=0.5)
for i, label in enumerate([f"MLP 10 Neuron {i}" for i in (order[7:11].tolist() + order[3:7].tolist())]):
    full_fig.layout.annotations[i]["text"] = label
full_fig.show()
full_fig.write_image("paper-plots/appendix/neurons.pdf")
