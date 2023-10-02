# Code Release 
Here is a code release for the 2023 NeurIPS paper "How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model". 

The code release is structured as follows:
- `circuit_discovery.py` reproduces the circuit discovery and semantics assignment process
- `big_ds_experiments.py` reproduces the experiments run on the larger, 10,000 element dataset
- `neuron_investigations.py` reproduces the neuron-level experiments
- `sequence_generalization.py` reproduces the generalization experiments

The aforementioned files will cache files (in `paper-cache`) that can be used to generate plots using these scripts: 
- `circuit_discovery_plotting.py`
- `pca_plots.py`
- `neuron_plots.py`
- `appendix_plots.py`

In addition, we include three useful files in the `cache` folder (indices of the relevant logits, the nouns used in our template, and the order of MLP10 neurons, which otherwise takes a long time to compute). Finally, we include two utility files, `utils.py` and `color_utils.py` (for plotting).

Most of these experiments started as exploratory VSCode notebooks, but can be run just as easily as Python scripts, and will produce all necessary output. We also include a few smaller notebooks that correspond to discussions with reviewers, and didn't fit in neatly with the rest of our experiments:
- `random_circuit_ablation.py`: allows you to try ablating random circuits, as opposed to the one we found
- `random_years.py`: tests GPT-2's responses to random sequences of years from the same century
- `topk_years.py`: tests the degree to which GPT-2's top-k YY predictions are correct.

## Running the code
Unfortunately, using the `rust-circuit` library to work with `gpt2-small` is not easy. To run the code, follow these steps:

1. Compile [rust-circuit](https://github.com/redwoodresearch/rust_circuit_public), following the instructions there given; note that this requires clang and rust. The repo instructs you to install maturin; be sure to install 0.14.x (we used 0.14.7), as newer versions do not work.
2. Install this project's requirements via the provided requirements file `pip install -r requirements.txt`
3. Download the `gpt2-small` model files from [this link](https://rrserve.s3.us-west-2.amazonaws.com/remix/remix_tensors.zip). Extract them to a folder called `~/tensors_by_hash_cache/`. If this doesn't work, try extracting them instead to `../rrfs/tensor_db`.

# The paper
Our paper is available on [ArXiv](https://arxiv.org/abs/2305.00586) and, hopefully sometime soon, on the NeurIPS website. You can cite it like so:

```
@inproceedings{
hanna2023how,
title={How does {GPT}-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model},
author={Michael Hanna and Ollie Liu and Alexandre Variengien},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=p4PckNQR8k}
}
```
