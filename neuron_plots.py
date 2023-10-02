#%%
import torch
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %%
outer_products = torch.load("paper-cache/t10-logitlens.pt")
direct_effects = torch.load("paper-cache/t10-direct_effects.pt")
patched_logit_diff = torch.load("paper-cache/t10-t10patched.pt")
order = torch.load("cache/order.pt")
#%%
top3_fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"MLP 10 Neuron {i}" for i in order[:3]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
top3_fig.add_trace(go.Heatmap(z=outer_products[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
top3_fig.add_trace(go.Heatmap(z=outer_products[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)
top3_fig.add_trace(go.Heatmap(z=outer_products[2].cpu(), coloraxis="coloraxis1"), row=1, col=3)

top3_fig.update_layout(
    width=1000,
    height=350,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Logit Lens Magnitude",
        colorbar_title_side="right",
    ),
)
top3_fig.update_yaxes(autorange="reversed")
top3_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    xaxis3=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
top3_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
top3_fig.show()
top3_fig.write_image("paper-plots/top3-neurons.pdf")
# %%
top3_fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"MLP 10 Neuron {i}" for i in order[:3]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
top3_fig.add_trace(go.Heatmap(z=direct_effects[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
top3_fig.add_trace(go.Heatmap(z=direct_effects[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)
top3_fig.add_trace(go.Heatmap(z=direct_effects[2].cpu(), coloraxis="coloraxis1"), row=1, col=3)

top3_fig.update_layout(
    width=1000,
    height=350,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Logit Change",
        colorbar_title_side="right",
    ),
)
top3_fig.update_yaxes(autorange="reversed")
top3_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    xaxis3=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
top3_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
top3_fig.show()
top3_fig.write_image("paper-plots/top3-neurons-directeffects.pdf")
# %%
ll_sum_fig = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"Logit Lens of Top-10 MLP 10 Neurons"],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
ll_10 = outer_products[:10].sum(0).cpu()
ll_sum_fig.add_trace(go.Heatmap(z=ll_10 - ll_10[0:1], coloraxis="coloraxis1"), row=1, col=1)
ll_sum_fig.update_layout(
    width=450,
    height=350,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Logit Change",
        colorbar_title_side="right",
        cmin=-30,
        cmax=30,
    ),
)
ll_sum_fig.update_yaxes(autorange="reversed")
ll_sum_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
ll_sum_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
ll_sum_fig.show()
ll_sum_fig.write_image("paper-plots/top10-neurons-logitlenssum.pdf")
# %%
de_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"{t} Direct Effects" for t in ["Summed", "Patched"]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
de_fig.add_trace(go.Heatmap(z=direct_effects[:10].sum(0).cpu(), coloraxis="coloraxis1"), row=1, col=1)
de_fig.add_trace(go.Heatmap(z=patched_logit_diff.cpu(), coloraxis="coloraxis1"), row=1, col=2)

de_fig.update_layout(
    width=750,
    height=350,
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
de_fig.update_yaxes(autorange="reversed")
de_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
de_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
de_fig.show()
de_fig.write_image("paper-plots/top10-neurons-directeffects.pdf")
#%%
top100_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"Top-{i} MLP 10 Neurons Logit Lens" for i in [100, 200]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
ll_100 = outer_products[:100].sum(0).cpu()
ll_200 = outer_products[:200].sum(0).cpu()
top100_fig.add_trace(go.Heatmap(z=ll_100 - ll_100[0:1], coloraxis="coloraxis1"), row=1, col=1)
top100_fig.add_trace(go.Heatmap(z=ll_200 - ll_200[0:1], coloraxis="coloraxis1"), row=1, col=2)

top100_fig.update_layout(
    width=750,
    height=350,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Logit Lens Magnitude",
        colorbar_title_side="right",
        cmin=-45,
        cmax=50,
    ),
)
top100_fig.update_yaxes(autorange="reversed")
top100_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    xaxis3=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
top100_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
top100_fig.show()
top100_fig.write_image("paper-plots/top100-neurons.pdf")
# %%
y_labels = list(range(2, 99))
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
    coloraxis_colorbar_title=f"Logit Lens Magnitude",
    coloraxis_colorbar_title_side="right",
)


full_fig.update_layout(title_x=0.5)
for i, label in enumerate([f"MLP 10 Neuron {i}" for i in (order[7:11].tolist() + order[3:7].tolist())]):
    full_fig.layout.annotations[i]["text"] = label
full_fig.show()
full_fig.write_image("paper-plots/appendix/neurons.pdf")
