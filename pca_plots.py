#%%
from pathlib import Path

import torch
import rust_circuit as rc
import matplotlib.pyplot as plt

from dataset import YearDataset
from utils import (
    load_gpt2_small_circuit,
    load_diff_model,
    to_device,
    load_and_split_gpt2,
)

from color_utils import RGB_to_hex, polylinear_gradient
from sklearn.decomposition import PCA
import numpy as np

#%%
# Loading our base model
DEVICE = "cuda"
_, tokenizer, _ = load_gpt2_small_circuit()

#%%
# Creating our dataset
years_to_sample_from = torch.arange(1702, 1799)
N = len(years_to_sample_from)
ds = YearDataset(years_to_sample_from, N, Path("cache/potential_nouns.txt"), ordered=True, device=DEVICE)

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
ld_circuit, group = load_diff_model(circuit, year_indices, ds.good_mask, logit_diff=False, device=DEVICE)

#%%
def se(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return transform.sample(c).evaluate()


# Let's make a copy of the circuit that actually has inputs!
c = ld_circuit.update(
    "tokens",
    lambda _: rc.DiscreteVar(to_device(rc.Array(ds.good_toks, name="tokens"), DEVICE), probs_and_group=group),
)

#%%
BASE_PALETTE = [
    [255, 128, 0],
    [128, 255, 0],
    [128, 0, 255],
    [0, 128, 255],
    [0, 255, 128],
    [255, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [0, 76, 153],
    [76, 0, 153],
]
color_gradient = polylinear_gradient([RGB_to_hex(color) for color in BASE_PALETTE], 110)["hex"][2:99]
print(len(color_gradient))


def plot(reps, title="PCA for Embeddings"):
    reducer = PCA(n_components=2)
    fits = np.array(reducer.fit_transform(reps))
    print(reducer.explained_variance_ratio_)
    return raw_plot(fits, title)


def raw_plot(fits, title="PCA for Embeddings"):
    fig, ax = plt.subplots()
    ax.scatter(fits[:, 0], fits[:, 1], c=color_gradient)

    texts = [str(i) for i in range(2, 99)]
    for i, txt in enumerate(texts):
        ax.annotate(txt, (fits[i, 0], fits[i, 1]), xytext=(2, 2), textcoords="offset points")

    ax.set_title(title)
    return fig
# %%
a7h10_reps = se(c.get_unique("a7_h10_t11")).squeeze().cpu().numpy()
fig = plot(a7h10_reps, title="PCA of a7.h10 outputs")
# %%
a7h8_reps = se(c.get_unique("a7_h8_t11")).squeeze().cpu().numpy()
fig = plot(a7h8_reps, title="PCA of a7.h8 outputs")
#%%
m8_input = se(c.get_unique("b8.m.input"))[:, 11].squeeze().cpu().numpy()
fig = plot(m8_input, title="PCA of MLP 8 input")
# %%
embeds = se(c.get_unique("embeds"))[:, 7].squeeze().cpu().numpy()
fig = plot(embeds, title="PCA of static embeddings")
#%%
fig, axs = plt.subplots(1, 4, sharey=True)
remove_ticks = False
for reps, ax, name in zip(
    [m8_input, a7h10_reps, a7h8_reps, embeds],
    axs,
    ["MLP 8 Input", "a7.h10 Output", "a7.h8 Output", "Static Embeddings"],
):
    reducer = PCA(n_components=2)
    fits = np.array(reducer.fit_transform(reps))
    fits[:, 0] = (fits[:, 0] - fits[:, 0].mean()) / (fits[:, 0].std())
    fits[:, 1] = (fits[:, 1] - fits[:, 1].mean()) / (fits[:, 1].std())
    ax.scatter(fits[:, 0], fits[:, 1], c=color_gradient)

    texts = [f"{i:02d}" for i in range(2, 99)]
    for i, txt in enumerate(texts):
        ax.annotate(txt, (fits[i, 0], fits[i, 1]), xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"PCA of {name}")
    # ax.set_ylim(-2,2)
    if remove_ticks:
        ax.tick_params(
            axis="y",
            left=False,
        )
    else:
        remove_ticks = True
fig.set_size_inches([20, 6])
fig.set_dpi(200)
fig.tight_layout()
fig.savefig("paper-plots/pca-analysis.pdf")
fig.show()

