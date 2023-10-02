#%%
from typing import cast, List
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from utils import show_diffs

# Loading our base model
logit_diff = False
metric = "logit mean" if logit_diff else "prob"

#%%
probs = torch.load("paper-cache/probs.pt")
fig = show_diffs(
    probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="Blues",
)

fig.update_layout(margin=dict(l=0, r=0, b=10, t=30), title_x=0.5, height=500, width=600)
fig.show()
fig.write_image("paper-plots/probability-heatmap.pdf")

i = 39
plt.plot(probs[i].cpu())
plt.title(f"GPT-2 Probabilities when YY={i + 2}")
plt.xlabel(f"Predicted Year")
plt.ylabel(f"probability")
plt.savefig("paper-plots/probability-slice.pdf", bbox_inches="tight")


#%%
def make_ipp_plot(results, title, cmin=-2.0, cmax=2.0):
    fig = make_subplots(
        subplot_titles=[title],
    )

    fig.add_trace(go.Heatmap(z=results, coloraxis="coloraxis1"), row=1, col=1)
    fig.update_layout(
        width=450,
        height=350,
        coloraxis1=dict(
            colorscale="RdBu",
            colorbar_x=1.007,
            colorbar_thickness=23,
            colorbar_title=f"{metric} diff variation",
            colorbar_title_side="right",
            cmin=cmin,
            cmax=cmax,
        ),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
        yaxis=dict(title="Layer"),
    )
    fig.update_layout(margin=dict(l=30, r=50, b=30, t=30), title_x=0.5)
    return fig


results_logits = torch.load("paper-cache/ipp_logits.pt").cpu()
results_11, results_10, results_9, results_8 = (torch.load(f"paper-cache/ipp_mlp{i}.pt").cpu() for i in [11, 10, 9, 8])

fig_logits = make_ipp_plot(results_logits, "IPP: Direct Contributions to the Logits", -0.5, 0.5)
fig_logits.show()
fig_logits.write_image("paper-plots/logits-ipp.pdf")


fig_11 = make_ipp_plot(results_11, "m11", cmin=-0.1, cmax=0.1)
fig_11.show()
fig_11.write_image("paper-plots/appendix/m11-ipp.pdf")

fig_10 = make_ipp_plot(results_10, "m10", cmin=-0.2, cmax=0.2)
fig_10.show()
fig_10.write_image("paper-plots/appendix/m10-ipp.pdf")

fig_9 = make_ipp_plot(results_9, "m9", cmin=-0.2, cmax=0.2)
fig_9.show()
fig_9.write_image("paper-plots/appendix/m9-ipp.pdf")

fig_8 = make_ipp_plot(results_8, "m8", cmin=-0.1, cmax=0.1)
fig_8.show()
fig_8.write_image("paper-plots/appendix/m8-ipp.pdf")

x_labels = [f"h{i}" for i in range(12)] + ["mlp"]
y_labels = list(range(12))
title = "IPP: Direct Contributions via MLPs"  # "MLP Iterative Path Patching"
full_fig = px.imshow(
    torch.stack([results_11, results_10, results_9, results_8]).cpu(),
    aspect="equal",
    facet_col_spacing=0,
    facet_row_spacing=0,
    facet_col=0,
    facet_col_wrap=2,
    labels=dict(x="Head", y="Layer"),
    title=title,
    x=x_labels,
    y=y_labels,
    range_color=(-0.2, 0.2),
    zmin=-1.0,
    zmax=1.0,
    color_continuous_scale="RdBu",
)
full_fig.update_layout(
    margin=dict(l=0, r=0, b=30, t=35),
    coloraxis_colorbar_title=f"{metric} diff variation",
    coloraxis_colorbar_title_side="right",
    coloraxis_colorbar_thickness=23,
)

full_fig.update_layout(title_x=0.5)
for i, label in enumerate([f"MLP {i}" for i in [9, 8, 11, 10]]):
    full_fig.layout.annotations[i]["text"] = label
full_fig.show()
full_fig.write_image("paper-plots/mlps-ipp.pdf")


#%%
attn_patterns_7_10 = torch.load("paper-cache/attn_patterns_a7.h10.pt")
attn_patterns_8_11 = torch.load("paper-cache/attn_patterns_a8.h1.pt")
attn_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=False,
    subplot_titles=["Mean Attention Pattern for a7.h10", "Mean Attention Pattern for a8.h11"],
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
)
attn_fig.add_trace(go.Heatmap(z=attn_patterns_7_10, coloraxis="coloraxis1"), row=1, col=1)
attn_fig.add_trace(go.Heatmap(z=attn_patterns_8_11, coloraxis="coloraxis1"), row=1, col=2)

attn_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="Blues",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Attention",
        colorbar_title_side="right",
    ),
)
attn_fig.update_yaxes(autorange="reversed")
prompt = ["<bos>", "The", "NOUN", "lasted", "from", "the", "year", "XX1", "YY", "to", "the", "year", "XX2"]
attn_fig.update_layout(
    xaxis=dict(title="Key", position=0.5, tickmode="array", tickvals=list(range(len(prompt))), ticktext=prompt),
    xaxis2=dict(title="Key", position=0.5, tickmode="array", tickvals=list(range(len(prompt))), ticktext=prompt),
    yaxis=dict(title="Query", tickmode="array", tickvals=list(range(len(prompt))), ticktext=prompt),
    yaxis2=dict(title=None, tickmode="array", tickvals=list(range(len(prompt))), ticktext=prompt),
)

attn_fig.update_layout(
    margin=dict(l=30, r=30, b=30, t=30),
    title_x=0.5,
)
attn_fig.show()
attn_fig.write_image("paper-plots/attn-patterns.pdf")

#%%
### Logit Lens Attention Heads
logits_a7 = torch.load("paper-cache/logit_lens_a7.h10.pt")
logits_a8 = torch.load("paper-cache/logit_lens_a8.h11.pt")
attn_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"Logit Lens of a7.h10", "Logit Lens of a8.h11"],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
attn_fig.add_trace(go.Heatmap(z=logits_a7, coloraxis="coloraxis1"), row=1, col=1)
attn_fig.add_trace(go.Heatmap(z=logits_a8, coloraxis="coloraxis1"), row=1, col=2)

attn_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Magnitude in unembedding space",
        colorbar_title_side="right",
    ),
)
attn_fig.update_yaxes(autorange="reversed")
attn_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
attn_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
attn_fig.show()
attn_fig.write_image("paper-plots/attn-logitlens.pdf")


#%%
### MLP Logit Lens
logit_lens_mlp = [torch.load(f"paper-cache/logit_lens_mlp{i}.pt").cpu() for i in range(8, 12)]
logit_lens_fig = make_subplots(
    rows=1,
    cols=4,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"Logit Lens of MLP {i}" for i in [11, 10, 9, 8]],
    horizontal_spacing=0.05,
    vertical_spacing=0.15,
)
logit_lens_fig.add_trace(
    go.Heatmap(z=logit_lens_mlp[3] - logit_lens_mlp[3][:, 0:1], coloraxis="coloraxis1"), row=1, col=1
)
logit_lens_fig.add_trace(
    go.Heatmap(z=logit_lens_mlp[2] - logit_lens_mlp[2][:, 0:1], coloraxis="coloraxis2"), row=1, col=2
)
logit_lens_fig.add_trace(
    go.Heatmap(z=logit_lens_mlp[1] - logit_lens_mlp[1][:, 0:1], coloraxis="coloraxis3"), row=1, col=3
)
logit_lens_fig.add_trace(
    go.Heatmap(z=logit_lens_mlp[0] - logit_lens_mlp[0][:, 0:1], coloraxis="coloraxis4"), row=1, col=4
)

logit_lens_fig.update_layout(
    width=2000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=0.215,
        colorbar_thickness=23,
        colorbar_title="Magnitude in unembedding space",
        colorbar_title_side="right",
        cmin=-5,
        cmax=4,
    ),
    coloraxis2=dict(
        colorscale="RdBu",
        colorbar_x=0.48,
        colorbar_thickness=23,
        colorbar_title="Magnitude in unembedding space",
        colorbar_title_side="right",
        cmin=-15,
        cmax=15,
    ),
    coloraxis3=dict(
        colorscale="RdBu",
        colorbar_x=0.74,
        colorbar_thickness=23,
        colorbar_title="Magnitude in unembedding space",
        colorbar_title_side="right",
    ),
    coloraxis4=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Magnitude in unembedding space",
        colorbar_title_side="right",
        cmin=-15,
        cmax=15,
    ),
)

logit_lens_fig.update_layout(
    margin=dict(l=30, r=30, b=30, t=30), title_x=0.5, xaxis_title="Predicted Year", yaxis_title="YY"
)
logit_lens_fig.update_yaxes(autorange="reversed")
logit_lens_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    xaxis3=dict(title="Predicted Year"),
    xaxis4=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
logit_lens_fig.show()
logit_lens_fig.write_image("paper-plots/mlps-logitlens.pdf")


#%%
### Patched prob plot
patched_probs = torch.load("paper-cache/patched_probs.pt").cpu()
fig = show_diffs(
    patched_probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="Patched GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="Blues",
)
fig.update_layout(margin=dict(l=0, r=0, b=10, t=30), title_x=0.5, height=500, width=600)
fig.show()
fig.write_image("paper-plots/patched-probability-heatmap.pdf")
#%%
### Full Patched prob plot
patched_probs = torch.load("paper-cache/full_patched_probs.pt").cpu()
fig = show_diffs(
    patched_probs,
    center_zero=False,
    zrange=(0.0, 0.25),
    title="Full-Circuit Patched GPT-2 Small Probability Heatmap",
    zlabel="probability",
    color_continuous_scale="Blues",
)
fig.update_layout(margin=dict(l=0, r=0, b=10, t=30), title_x=0.5, height=500, width=600)
fig.show()
fig.write_image("paper-plots/appendix/full-patched-probability-heatmap.pdf")
#%%
attn_v = torch.load("paper-cache/attn_v.pt").cpu()
attn_v_fig = make_ipp_plot(attn_v, "Attention Value", cmin=-0.5, cmax=0.5)
attn_v_fig.show()
attn_v_fig.write_image("paper-plots/appendix/attn_v.pdf")
#%%
ipp_low_mlp = [torch.load(f"paper-cache/results_mlp{i}.pt").cpu() for i in range(0, 4)]
logit_lens_fig = make_subplots(
    rows=1,
    cols=4,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"Iterative Path Patching MLP {i}" for i in [3, 2, 1, 0]],
    horizontal_spacing=0.01,
    vertical_spacing=0.15,
)
logit_lens_fig.add_trace(go.Heatmap(z=ipp_low_mlp[3], coloraxis="coloraxis1"), row=1, col=1)
logit_lens_fig.add_trace(go.Heatmap(z=ipp_low_mlp[2], coloraxis="coloraxis1"), row=1, col=2)
logit_lens_fig.add_trace(go.Heatmap(z=ipp_low_mlp[1], coloraxis="coloraxis1"), row=1, col=3)
logit_lens_fig.add_trace(go.Heatmap(z=ipp_low_mlp[0], coloraxis="coloraxis1"), row=1, col=4)

logit_lens_fig.update_layout(
    width=2000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title=f"{metric} diff variation",
        colorbar_title_side="right",
        cmin=-0.5,
        cmax=0.5,
    ),
)

logit_lens_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5, xaxis_title="Head", yaxis_title="Layer")
x_labels = [f"h{i}" for i in range(12)] + ["mlp"]
y_labels = list(range(12))
logit_lens_fig.update_yaxes(autorange="reversed")
logit_lens_fig.update_layout(
    xaxis=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=x_labels),
    xaxis2=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=x_labels),
    xaxis3=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=x_labels),
    xaxis4=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=x_labels),
    yaxis=dict(
        title="Layer",
    ),
)
logit_lens_fig.show()
logit_lens_fig.write_image("paper-plots/appendix/low-mlps-ipp.pdf")
# %%
attention_collated = torch.load('paper-cache/attn_collated.pt')[:, -1, 8]
plt.plot(torch.arange(2, 99), attention_collated)
plt.ylabel("Attention")
plt.xlabel("Input Year (YY)")
plt.xlim(0,100)
plt.title("Attention of a7.h10 (end position) to the YY position")
plt.savefig('paper-plots/appendix/attn_proportion.pdf')
plt.show()
# %%
