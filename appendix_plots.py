#%%
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#%%
failure_probs = [torch.load(f"paper-cache/generalization/probs_{option}.pt") for option in range(3, 6)]

failure_prob_fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=["Probability Heatmap"] * 3,
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
failure_prob_fig.add_trace(go.Heatmap(z=failure_probs[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
failure_prob_fig.add_trace(go.Heatmap(z=failure_probs[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)
failure_prob_fig.add_trace(go.Heatmap(z=failure_probs[2].cpu(), coloraxis="coloraxis1"), row=1, col=3)

failure_prob_fig.update_layout(
    width=1400,
    height=400,
    coloraxis1=dict(
        colorscale="Blues",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Probability",
        colorbar_title_side="right",
        cmin=0.0,
        cmax=0.25,
    ),
)
failure_prob_fig.update_yaxes(autorange="reversed")
failure_prob_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    xaxis3=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
failure_prob_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
failure_prob_fig.show()
failure_prob_fig.write_image("paper-plots/appendix/failure-probs.pdf")
# %%
success_probs = [torch.load(f"paper-cache/generalization/probs_{option}.pt") for option in [0,8,2]]

success_prob_fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=["Probability Heatmap"] * 3,
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
success_prob_fig.add_trace(go.Heatmap(z=success_probs[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
success_prob_fig.add_trace(go.Heatmap(z=success_probs[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)
success_prob_fig.add_trace(go.Heatmap(z=success_probs[2].cpu(), coloraxis="coloraxis1"), row=1, col=3)

success_prob_fig.update_layout(
    width=1400,
    height=400,
    coloraxis1=dict(
        colorscale="Blues",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Probability",
        colorbar_title_side="right",
        cmin=0.0,
        cmax=0.2,
    ),
)
success_prob_fig.update_yaxes(autorange="reversed")
success_prob_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    xaxis3=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
success_prob_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
success_prob_fig.show()
success_prob_fig.write_image("paper-plots/appendix/success-probs.pdf")
# %%
bc_probs = [torch.load(f"paper-cache/generalization/probs_{option}.pt") for option in range(6, 8)]

bc_prob_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=["Probability Heatmap"] * 2,
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
bc_prob_fig.add_trace(go.Heatmap(z=bc_probs[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
bc_prob_fig.add_trace(go.Heatmap(z=bc_probs[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)

bc_prob_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="Blues",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title="Probability",
        colorbar_title_side="right",
        cmin=0.0,
        cmax=0.2,
    ),
)
bc_prob_fig.update_yaxes(autorange="reversed")
bc_prob_fig.update_layout(
    xaxis=dict(title="Predicted Year"),
    xaxis2=dict(title="Predicted Year"),
    yaxis=dict(title="YY", tickmode="array", tickvals=list(range(0, 98, 5)), ticktext=list(range(2, 99, 5))),
)
bc_prob_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
bc_prob_fig.show()
bc_prob_fig.write_image("paper-plots/appendix/bc-probs.pdf")
# %%
success_ipps = [torch.load(f"paper-cache/generalization/ipp_{option}.pt") for option in [0,8,2]]

success_ipp_fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=["Iterative Path Patching: Logits"] * 3,
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
success_ipp_fig.add_trace(go.Heatmap(z=success_ipps[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
success_ipp_fig.add_trace(go.Heatmap(z=success_ipps[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)
success_ipp_fig.add_trace(go.Heatmap(z=success_ipps[2].cpu(), coloraxis="coloraxis1"), row=1, col=3)

success_ipp_fig.update_layout(
    width=1400,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title=f"prob diff variation",
        colorbar_title_side="right",
        cmin=-0.5,
        cmax=0.5,
    ),
)


success_ipp_fig.update_yaxes(autorange="reversed")
success_ipp_fig.update_layout(
    xaxis=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    xaxis2=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    xaxis3=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    yaxis=dict(title="Layer"),
)
success_ipp_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
success_ipp_fig.show()
success_ipp_fig.write_image("paper-plots/appendix/success-ipps.pdf")

# %%
bc_ipps = [torch.load(f"paper-cache/generalization/ipp_{option}.pt") for option in range(6, 8)]

bc_ipp_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=["Iterative Path Patching: Logits"] * 2,
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
bc_ipp_fig.add_trace(go.Heatmap(z=bc_ipps[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
bc_ipp_fig.add_trace(go.Heatmap(z=bc_ipps[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)

bc_ipp_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title=f"prob diff variation",
        colorbar_title_side="right",
        cmin=-0.25,
        cmax=0.25,
    ),
)


bc_ipp_fig.update_yaxes(autorange="reversed")
bc_ipp_fig.update_layout(
    xaxis=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    xaxis2=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    yaxis=dict(title="Layer"),
)
bc_ipp_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
bc_ipp_fig.show()
bc_ipp_fig.write_image("paper-plots/appendix/bc-ipps.pdf")
# %%
m7m8_ipps = [torch.load(f"paper-cache/generalization/m{i}_2.pt") for i in range(7, 9)]

m7m8_ipp_fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=True,
    subplot_titles=[f"Iterative Path Patching: MLP {i}" for i in [7, 8]],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)
m7m8_ipp_fig.add_trace(go.Heatmap(z=m7m8_ipps[0].cpu(), coloraxis="coloraxis1"), row=1, col=1)
m7m8_ipp_fig.add_trace(go.Heatmap(z=m7m8_ipps[1].cpu(), coloraxis="coloraxis1"), row=1, col=2)

m7m8_ipp_fig.update_layout(
    width=1000,
    height=400,
    coloraxis1=dict(
        colorscale="RdBu",
        colorbar_x=1.007,
        colorbar_thickness=23,
        colorbar_title=f"prob diff variation",
        colorbar_title_side="right",
        cmin=-0.05,
        cmax=0.05,
    ),
)


m7m8_ipp_fig.update_yaxes(autorange="reversed")
m7m8_ipp_fig.update_layout(
    xaxis=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    xaxis2=dict(title="Head", tickmode="array", tickvals=list(range(13)), ticktext=list(range(12)) + ["mlp"]),
    yaxis=dict(title="Layer"),
)
m7m8_ipp_fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), title_x=0.5)
m7m8_ipp_fig.show()
m7m8_ipp_fig.write_image("paper-plots/appendix/m7m8-ipps.pdf")
# %%
