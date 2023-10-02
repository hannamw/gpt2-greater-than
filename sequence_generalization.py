#%%
from typing import cast
import random

import torch
import rust_circuit as rc

from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher

from utils import (
    load_gpt2_small_circuit,
    HeadOrMlpType, 
    MLPHeadAndPosSpec,
    load_diff_model,
    iterative_path_patching_nocorr,
    path_patching,
    to_device,
    load_and_split_gpt2,
    show_diffs,
    make_extender_factory,
    get_valid_years
)

#%%
# Loading our base model
DEVICE = "cuda:0"
MODEL_ID = "gelu_12_tied"  # aka gpt2 small
_, tokenizer, _ = load_gpt2_small_circuit()

#%%
"""
Here are the tasks that we discuss:
0 “The <noun> started in the year 17YY and ended in the year 17”
1 “It was 17YY then. Some years later, it was the year 17” 
2 “1599, 1607, 1633, 1679, 17YY, 17”

3 “1799, 1753, 1733, 1701, 16YY, 16”
4 Exact-answer tasks, e.g. “1599, 1607, 1633, 1679, 17YY, 17”
5 “17YY is smaller than 17”

6 The <noun> ended in the year 17YY and started in the year 17”
7 “The <noun> lasted from the year 7YY BC to the year 7”
8 "The price of that <item> ranges from $ 17YY to $ 17"

9 "XXY1, XXY2, XXY3, XXY4, XXY5, XX", where Y1,...,Y5 are randomly sampled
"""

# Creating our dataset
years = torch.arange(1702, 1799)
last_two_digits = years % 100

with open("cache/potential_nouns.txt", "r") as f:
    noun_list = [line.strip() for line in f]
nouns = random.choices(noun_list, k=len(years))

for option in range(10):
    gt = True
    if option == 0:
        sentences = [
            f"<|endoftext|> The {noun} started in the year 17{y:02d} and ended in the year 17"
            for noun, y in zip(nouns, last_two_digits)
        ]
        sentences_01 = [
            f"<|endoftext|> The {noun} started in the year 1701 and ended in the year 17"
            for noun, _ in zip(nouns, last_two_digits)
        ]
    elif option == 1:
        sentences = [
            f"<|endoftext|> The {noun} happened in 17{y:02d}. Some years later, it is now the year 17"
            for noun, y in zip(nouns, last_two_digits)
        ]
        sentences_01 = [
            f"<|endoftext|> The {noun} happened in 1701. Some years later and it is now the year 17"
            for noun, _ in zip(nouns, last_two_digits)
        ]
    elif option == 2:
        sentences = [f"<|endoftext|> 1599, 1607, 1633, 1679, 17{y:02d}, 17" for y in last_two_digits]
        sentences_01 = [f"<|endoftext|> 1599, 1607, 1633, 1679, 1701, 17" for _ in last_two_digits]
    elif option == 3:
        sentences = [f"<|endoftext|> 1799, 1753, 1733, 1701, 16{y:02d}, 16" for y in last_two_digits]
        sentences_01 = ["<|endoftext|> 1799, 1753, 1733, 1701, 1699, 16" for _ in last_two_digits]
        gt = False
    elif option == 4:
        sentences = []
        corrects = []
        for y in years:
            i = 2
            while (y % 100) % i == 0:
                i += 1
            sentences.append(f"<|endoftext|> {y-4*i:04d}, {y-3*i:04d}, {y-2*i:04d}, {y-1*i:04d}, {y:04d}, 17")
            corrects.append(y + i)
        sentences_01 = [f"<|endoftext|> 1693, 1695, 1697, 1699, 1701, 17" for _ in years]
    elif option == 5:
        sentences = [f"<|endoftext|> 17{y:02d} is smaller than 17" for y in last_two_digits]
        sentences_01 = [f"<|endoftext|> 1701 is smaller than 17" for _ in last_two_digits]
    elif option == 6:
        sentences = [
            f"<|endoftext|> The {noun} ended in the year 17{y:02d} and started in the year 17"
            for noun, y in zip(nouns, last_two_digits)
        ]
        sentences_01 = [
            f"<|endoftext|> The {noun} ended in the year 1799 and started in the year 17"
            for noun, _ in zip(nouns, last_two_digits)
        ]
        gt = False
    elif option == 7:
        sentences = [
            f"<|endoftext|> The {noun} lasted from the year 7{y:02d} BC to the year 7"
            for noun, y in zip(nouns, last_two_digits)
        ]
        sentences_01 = [
            f"<|endoftext|> The {noun} lasted from the year 799 BC to the year 7"
            for noun, _ in zip(nouns, last_two_digits)
        ]
        for i in [0, 18, 35, 45, 48, 58, 66, 68, 75, 78]:
            sentences[i] = sentences[i + 1]
        gt = False
    elif option == 8:
        items = [
            "gem",
            "necklace",
            "watch",
            "ring",
            "suitcase",
            "scarf",
            "suit",
            "shirt",
            "sweater",
            "dress",
            "fridge",
            "TV",
            "bed",
            "bike",
            "lamp",
            "table",
            "chair",
            "painting",
            "sculpture",
            "plant",
        ]
        sentences = [
            f"<|endoftext|> The price of that {item} ranges from $ 17{y:02} to $ 17"
            for y, item in zip(last_two_digits, items * 5)
        ]
        sentences_01 = [
            f"<|endoftext|> The price of that {item} ranges from $ 1701 to $ 17"
            for y, item in zip(last_two_digits, items * 5)
        ]
    elif option == 9:
        years = []
        n = len(last_two_digits)
        centuries = torch.arange(10,19) * 100
        years_XX00 = centuries[torch.randint(len(centuries), (n,))]
        years_XX = years_XX00 // 100
        years_to_sample_from = get_valid_years(tokenizer, 1000, 1900)
        for XX00 in years_XX00:
            sample_space = years_to_sample_from[(years_to_sample_from >= XX00) & (years_to_sample_from < XX00+100)]
            years.append(sample_space[torch.randint(sample_space.size(0), (5,))])
        years = torch.stack(years)
        years_YY = years % 100
        last_two_digits = years_YY[:, -1]
        sentences = [f'{str(y.tolist())[1:-1]}, {XX}' for y, XX in zip(years, years_XX)]
        sentences_01 = [f'{str(y.tolist())[1:-3]}01, {XX}' for y, XX in zip(years, years_XX)]
    else:
        raise ValueError(f"Bad option given (should be 0 - 10): {option}")

    toks = [tokenizer(sentence, return_tensors="pt")["input_ids"].squeeze() for sentence in sentences]
    toks = torch.stack(toks).cuda()
    toks_01 = torch.stack(
        [tokenizer(sentence, return_tensors="pt")["input_ids"].squeeze() for sentence in sentences_01]
    ).cuda()

    MAX_LEN = toks.size(-1)
    END_POS = MAX_LEN - 1

    masks = []
    for year in last_two_digits:
        if gt:
            mask = torch.arange(100) > year
        else:
            mask = torch.arange(100) < year
        masks.append(mask)

    masks = torch.stack(masks)

    # Splitting our model to make it pretty
    logit_diff = False
    metric =  "prob"
    circuit = load_and_split_gpt2(MAX_LEN)
    year_indices = torch.load("cache/logit_indices.pt")
    ld_circuit, group = load_diff_model(circuit, year_indices, masks, logit_diff=logit_diff, device=DEVICE)

    def se(c):
        """Short function for Sample and Evaluate along the global variable `group`"""
        transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
        return transform.sample(c).evaluate()

    # Let's make a copy of the circuit that actually has inputs!
    c = ld_circuit.update(
        "tokens",
        lambda _: rc.DiscreteVar(to_device(rc.Array(toks, name="tokens"), DEVICE), probs_and_group=group),
    )
    baseline_mean = se(c).mean()

    probs = torch.softmax(se(c.get_unique("logits")), dim=-1)[:, -1, year_indices]
    torch.save(probs, f"paper-cache/generalization/probs_{option}.pt")

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
                baseline_data=toks,
                patch_data=patch_data,
                group=group,
                matcher_extenders=matcher_extenders,
                input_name="tokens",
                output_shape=(12, 13, -1),
            ).mean(-1)
        ) - baseline_mean

    #  Let's see what nodes are important, starting from the root, and looking at all MLPs / attention heads
    alt_tok_name = "01"
    results = iterative_path_patch([corr_root_matcher], end_pos_matcher_extenders, toks_01)
    torch.save(results, f"paper-cache/generalization/ipp_{option}.pt")

    if option in {0,2,8,9}:
        #  Let's see what nodes are important, starting from root->m11, and looking at all MLPs / attention heads
        alt_tok_name = "01"
        m11_extender = extender_factory(MLPHeadAndPosSpec(11, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
        m11_matcher = m11_extender(corr_root_matcher)


        #  Let's see what nodes are important, starting from root->m11->m10, and looking at all MLPs / attention heads
        alt_tok_name = "01"
        m10_extender = extender_factory(MLPHeadAndPosSpec(10, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
        m10_matcher = m10_extender(corr_root_matcher | m11_matcher)
    
        #  Let's see what nodes are important, starting from root->m11->m10->m9, and looking at all MLPs / attention heads
        alt_tok_name = "01"
        m9_extender = extender_factory(MLPHeadAndPosSpec(9, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
        m9_matcher = m9_extender(corr_root_matcher | m11_matcher | m10_matcher)


        #  Let's see what nodes are important, starting from root->m11->m10->m9->m8, and looking at all MLPs / attention heads
        alt_tok_name = "01"
        m8_extender = extender_factory(MLPHeadAndPosSpec(8, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
        m8_matcher = m8_extender(corr_root_matcher | m11_matcher | m10_matcher | m9_matcher)
        results = iterative_path_patch([m8_matcher], end_pos_matcher_extenders, toks_01)
        torch.save(results, f"paper-cache/generalization/m8_{option}.pt")

        if option == 2:
            #  Let's see what nodes are important, starting from root->m11->m10->m9->m8, and looking at all MLPs / attention heads
            alt_tok_name = "01"
            m7_extender = extender_factory(MLPHeadAndPosSpec(7, cast(HeadOrMlpType, "mlp"), END_POS), qkv=None)
            m7_matcher = m7_extender(corr_root_matcher | m11_matcher | m10_matcher | m9_matcher | m8_matcher)
            results = iterative_path_patch([m7_matcher], end_pos_matcher_extenders, toks_01)
            torch.save(results, f"paper-cache/generalization/m7_{option}.pt")

        extender_factory = make_extender_factory(MAX_LEN)
        end_pos_matcher_extenders = [
            extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), END_POS), qkv=None)
            for l in range(12)
            for h in list(range(12)) + ["mlp"]
        ]

        extra_mlps = [7] if option == 2 else []
        extra_heads = [(7, 11), (6, 1)] if option == 2 else []
        mlp_set_extender = extender_factory(
            {MLPHeadAndPosSpec(i, cast(HeadOrMlpType, "mlp"), END_POS) for i in [8, 9, 10, 11] + extra_mlps}, qkv=None
        )
        attention_set_extenders = extender_factory(
            {
                MLPHeadAndPosSpec(layer, head, END_POS)
                for layer, head in [(9, 1), (8, 11), (7, 10), (6, 9), (5, 5), (8, 8), (5, 1)] + extra_heads
            }
        )
        running = corr_root_matcher
        ms = attention_set_extenders(corr_root_matcher)
        for i in range(4):
            running = mlp_set_extender(running)
            ms = ms | attention_set_extenders(running)

        patched_circuit = path_patching(
            ld_circuit,
            toks_01,  # unpatched nodes get bad data
            toks,  # patched ndoes get good data
            ms,
            group,
            "tokens",
        )

        patched_results = se(patched_circuit)
        patched_mean = patched_results.mean()
        print(patched_mean, baseline_mean, patched_mean / baseline_mean)
        probs = torch.softmax(se(patched_circuit.get_unique("logits")), dim=-1)[:, -1, year_indices]
        show_diffs(probs, center_zero=False, title="Probability heatmap", color_continuous_scale="Blues").show()

        probs = torch.softmax(se(patched_circuit.get_unique("logits")), dim=-1)[:, -1, year_indices]
        show_diffs(probs, center_zero=False, title="Probability heatmap", color_continuous_scale="Blues").show()
