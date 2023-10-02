import time
from typing import Optional, Callable, List, Tuple, cast, Any, Set, Union, Literal, NamedTuple

import torch
import plotly.express as px
import rust_circuit as rc
import numpy as np
from transformers import GPT2TokenizerFast


from rust_circuit.py_utils import Slicer as S
from rust_circuit.causal_scrubbing.hypothesis import corr_root_matcher
from rust_circuit.algebric_rewrite import split_to_concat
from rust_circuit.model_rewrites import To, configure_transformer

year_indices = torch.load("cache/logit_indices.pt")

HeadOrMlpType = Union[int, Literal["mlp"]]
AttnSuffixForGpt = Union[Literal[""], Literal[".out"]]

def make_arr(
    tokens: torch.Tensor,
    name: str,
    device_dtype: rc.TorchDeviceDtype = rc.TorchDeviceDtype("cuda:0", "float32"),
) -> rc.Array:
    return rc.cast_circuit(rc.Array(tokens, name=name), device_dtype.op()).cast_array()

class MLPHeadAndPosSpec(NamedTuple):
    layer: int
    head_or_mlp: HeadOrMlpType
    pos: int

    def to_name(self, attn_suffix_for_bias: str) -> str:
        if self.head_or_mlp == "mlp":
            return f"m{self.layer}_t{self.pos}"
        else:
            return f"a{self.layer}{attn_suffix_for_bias}_h{self.head_or_mlp}_t{self.pos}"

def load_model_path(path: str):
    """Load a .circ file.
    """
    from rust_circuit.module_library import load_transformer_model_string

    with open(path) as f:
        return load_transformer_model_string(f.read())


def load_gpt2_small_circuit():
    return load_model_path("cache/gelu_12_tied.circ")


def get_valid_years(
    tokenizer: GPT2TokenizerFast,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(year_toks) for year_toks in tokens]
    valid = torch.tensor([(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized])
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid]


def to_device(c, device):
    return rc.cast_circuit(c, device_dtype=rc.TorchDeviceDtypeOp(device=device))


def add_year_mask_to_circuit(c: rc.Circuit, good_mask: torch.Tensor, device: str = "cpu"):
    """Run the circuit on all elements of tokens. Assumes the 'tokens' module exists in the circuit."""
    assert good_mask.ndim == 2 and good_mask.shape[1] == 100
    batch_size = good_mask.shape[0]
    group = rc.DiscreteVar.uniform_probs_and_group(batch_size)
    c = c.update(
        "good_mask",
        lambda _: rc.cast_circuit(
            rc.DiscreteVar(rc.Array(good_mask, name="good_mask"), probs_and_group=group),
            device_dtype=rc.TorchDeviceDtypeOp(device=device),
        ),
    )
    return c, group


def load_diff_model(
    split_circuit: rc.Circuit,
    number_indices: torch.Tensor,
    good_logits_masks: torch.Tensor,
    device="cpu",
):
    """Take GPT2 split by head and position and create a new circuit that is only computing the logit difference. The labels will be embedded in the circuit as a DiscreteVar. The function return the logit diff circuit and the group used by the DiscreteVar to sample the labels."""
    device_dtype = rc.TorchDeviceDtype(dtype="float32", device=device)

    good_logits_masks = good_logits_masks.float()

    good_mask = make_arr(
        torch.zeros(
            100,
        ).to(device),
        "good_mask",
        device_dtype=device_dtype,
    )
    split_circuit = rc.cast_circuit(split_circuit, device_dtype=rc.TorchDeviceDtypeOp(device=device))
    bad_mask = rc.Add.from_weighted_nodes((rc.Scalar(1.0), 1), (good_mask, -1))
    indices = [-1, number_indices.to(device)]

    probs = rc.softmax(split_circuit)
    year_probs = rc.Index(probs, indices, name="year_probs")  # rc.softmax(year_logits, name="year_probs")
    good_probs = rc.Einsum.from_einsum_string("l->", year_probs.mul(good_mask), name="good_probs")
    bad_probs = rc.Einsum.from_einsum_string("l->", year_probs.mul(bad_mask), name="bad_probs")
    prob_diff_circuit = to_device(
        rc.Add.from_weighted_nodes((good_probs, 1), (bad_probs, -1), name="prob_diff"), device
    )
    diff_circuit, group = add_year_mask_to_circuit(prob_diff_circuit, good_logits_masks, device=device)
    return rc.cast_circuit(diff_circuit, device_dtype=rc.TorchDeviceDtypeOp(device=device)), group


def replace_inputs(
    c: rc.Circuit,
    x: torch.Tensor,
    input_name: str,
    m: rc.IterativeMatcher,
    group: rc.Circuit,
    array_suffix: str = "_array",
):
    """
    Replace the input on the model branch define by the matcher `m` with a DiscreteVar.
    The input in the circuit `c` are expected non batched.
    """
    assert x.ndim >= 1
    c = c.update(
        m.chain(input_name),
        lambda _: rc.DiscreteVar(
            rc.Array(x, name=input_name + array_suffix),
            name=input_name,
            probs_and_group=group,
        ),
    )
    return c


def path_patching(
    circuit: rc.Circuit,
    baseline_data: torch.Tensor,
    patch_data: torch.Tensor,
    matcher: rc.IterativeMatcher,
    group: rc.Circuit,
    input_name: str,
) -> rc.Circuit:
    baseline_circuit = replace_inputs(
        circuit,
        baseline_data,
        input_name,
        corr_root_matcher,
        group,
        array_suffix="_baseline",
    )
    if len(matcher.get(circuit)) == 0:
        return baseline_circuit
    patched_circuit = replace_inputs(
        baseline_circuit,
        patch_data,
        input_name,
        matcher,
        group,
        array_suffix="_patched",
    )
    return patched_circuit


def iterative_path_patching_nocorr(
    circuit: rc.Circuit,
    matchers_to_extend: List[rc.IterativeMatcher],
    baseline_data: torch.Tensor,
    patch_data: torch.Tensor,
    group: rc.Circuit,
    matcher_extenders: List[Callable[[rc.IterativeMatcher], rc.IterativeMatcher]],
    input_name: str,
    output_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    t1 = time.time()
    circuits = []
    sampler = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    nb_not_found = 0
    for matcher_extender in matcher_extenders:
        matchers_to_h = []
        for matcher in matchers_to_extend:
            matchers_to_h.append(matcher_extender(matcher))
        union_matcher = matchers_to_h[0]

        for matcher in matchers_to_h[1:]:
            union_matcher = union_matcher | matcher

        if len(union_matcher.get(circuit)) == 0:
            nb_not_found += 1
        patched_circuit = path_patching(circuit, baseline_data, patch_data, union_matcher, group, input_name)
        patched_circuit = sampler(patched_circuit)  # we replace discrete vars by the real arrays
        circuits.append(patched_circuit)

    if nb_not_found > 0:
        print(f"Warning: No match found for {nb_not_found} matcher extenders")

    # a fancy function to evaluate fast many circuit that share tensors in common
    results = rc.optimize_and_evaluate_many(
        circuits,
        rc.OptimizationSettings(scheduling_simplify=False, scheduling_naive=True),
    )
    t2 = time.time()
    print(f"Time for path patching :{t2 - t1:.2f} s")
    if output_shape is None:
        return torch.cat([x.unsqueeze(0) for x in results], dim=0)

    return torch.cat(results).reshape(output_shape)

def collate(results: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    return torch.stack([results[years == y].mean(0) for y in range(2, 99)])

def logit_lens_ln_all(circuit: rc.Circuit, matcher: rc.IterativeMatcherIn, device="cpu"):
    component = circuit.get_unique(matcher)
    logits = circuit.get_unique("logits")
    logits_new_input = logits.update("final.input", lambda _: component)
    return rc.Index(logits_new_input, [-1], name=f"{circuit.name}_logit_lens_all")

def logit_lens_ln(circuit: rc.Circuit, matcher: rc.IterativeMatcherIn, device="cpu"):
    component = circuit.get_unique(matcher)
    logits = circuit.get_unique("logits")
    logits_new_input = logits.update("final.input", lambda _: component)
    return rc.Index(logits_new_input, [-1, year_indices.to(device)], name=f"{circuit.name}_logit_lens_years")


def make_scrubbed_printer(a, b):
    def scrub_colorer(c):
        getting_scrubbed = c.are_any_found(a)
        getting_unscrubbed = c.are_any_found(b)
        if getting_scrubbed and getting_unscrubbed:
            return "purple"
        elif getting_scrubbed:
            return "red"
        elif getting_unscrubbed:
            return "cyan"
        else:
            return "lightgrey"

    scrubbed_printer = rc.PrintHtmlOptions(shape_only_when_necessary=False, colorer=scrub_colorer)
    return scrubbed_printer


def load_and_split_gpt2(max_len: int):
    circ_dict, tokenizer, model_info = load_gpt2_small_circuit()
    unbound_circuit = circ_dict["t.bind_w"]

    tokens_arr = rc.Array(torch.zeros(max_len).to(torch.long), name="tokens")
    # We use this to index into the tok_embeds to get the proper embeddings
    token_embeds = rc.GeneralFunction.gen_index(circ_dict["t.w.tok_embeds"], tokens_arr, 0, name="tok_embeds")
    bound_circuit = model_info.bind_to_input(unbound_circuit, token_embeds, circ_dict["t.w.pos_embeds"])

    transformed_circuit = bound_circuit.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
        ),
    )
    transformed_circuit = rc.conform_all_modules(transformed_circuit)

    subbed_circuit = transformed_circuit.cast_module().substitute()
    subbed_circuit = subbed_circuit.rename("logits")

    def module_but_norm(circuit: rc.Circuit):
        if isinstance(circuit, rc.Module):
            if "norm" in circuit.name or "ln" in circuit.name or "final" in circuit.name:
                return False
            else:
                return True
        return False

    for i in range(100):
        subbed_circuit = subbed_circuit.update(module_but_norm, lambda c: c.cast_module().substitute())

    renamed_circuit = subbed_circuit.update(rc.Regex(r"[am]\d(.h\d)?$"), lambda c: c.rename(c.name + ".inner"))
    renamed_circuit = renamed_circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))

    for l in range(model_info.params.num_layers):
        # b0 -> a1.input, ... b11 -> final.input
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l+1}"
        renamed_circuit = renamed_circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

        # b0.m -> m0, etc.
        renamed_circuit = renamed_circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.m.p_bias", lambda c: c.rename(f"m{l}.p_bias"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a", lambda c: c.rename(f"a{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a.p_bias", lambda c: c.rename(f"a{l}.p_bias"))

        for h in range(model_info.params.num_layers):
            # b0.a.h0 -> a0.h0, etc.
            renamed_circuit = renamed_circuit.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))

    head_and_mlp_matcher = rc.IterativeMatcher(rc.Regex(r"^(a\d\d?.h\d\d?|m\d\d?)$"))
    partition = range(max_len)
    split_circuit = renamed_circuit.update(
        head_and_mlp_matcher,
        lambda c: split_to_concat(c, axis=0, partitioning_idxs=partition).rename(c.name + "_by_pos"),
    )

    new_names_dict = {}
    for l in range(model_info.params.num_layers):
        for i in range(max_len):
            for h in range(model_info.params.num_layers):
                # b0.a.h0 -> a0.h0, etc.
                new_names_dict[f"a{l}.h{h}_at_idx_{i}"] = f"a{l}_h{h}_t{i}"
            new_names_dict[f"m{l}_at_idx_{i}"] = f"m{l}_t{i}"

    split_circuit = split_circuit.update(
        rc.Matcher(*list(new_names_dict.keys())), lambda c: c.rename(new_names_dict[c.name])
    )

    return split_circuit


def show_mtx(mtx, title="NO TITLE :(", color_map_label="Logit diff variation", **kwargs):
    """Show a plotly matrix with a centered color map. Designed to display results of path patching experiments."""
    # we center the color scale on zero by defining the range (-max_abs, +max_abs)
    max_val = float(max(abs(mtx.min()), abs(mtx.max())))
    x_labels = [f"h{i}" for i in range(12)] + ["mlp"]
    fig = px.imshow(
        mtx,
        title=title,
        labels=dict(x="Head", y="Layer", color=color_map_label),
        color_continuous_scale="RdBu",
        range_color=(-max_val, max_val),
        x=x_labels,
        y=[str(i) for i in range(mtx.shape[0])],
        aspect="equal",
        **kwargs
    )
    fig.update_coloraxes(colorbar_title_side="right")
    return fig


def make_all_nodes_names(max_len: int):
    ALL_NODES_NAMES = set(
        [
            MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), pos).to_name("")
            for l in range(12)
            for h in (list(range(12)) + ["mlp"])  # type: ignore
            for pos in range(max_len)
        ]
    )
    return ALL_NODES_NAMES


def make_extender_factory(max_len: int):
    ALL_NODES_NAMES = make_all_nodes_names(max_len)

    def extender_factory(node: Union[MLPHeadAndPosSpec,Set[MLPHeadAndPosSpec]], qkv: Optional[str] = None):
        """
        `qkv` define the input of the attention block we want to reach.
        """
        assert qkv in ["q", "k", "v", None]

        if isinstance(node, set):
            node_name = {n.to_name("") for n in node}
            nodes_to_ban = ALL_NODES_NAMES.difference(node_name)
        else:
            node_name = node.to_name("")
            nodes_to_ban = ALL_NODES_NAMES.difference(set(node_name))

        if qkv is None:
            attn_block_input = rc.new_traversal(start_depth=0, end_depth=1)
        else:
            attn_block_input = rc.restrict(f"a.{qkv}", term_if_matches=True, end_depth=8)

        def matcher_extender(m: rc.IterativeMatcher):
            return m.chain(attn_block_input).chain(
                rc.new_traversal(start_depth=1, end_depth=2).chain(
                    rc.restrict(
                        rc.Matcher(node_name),
                        term_early_at=rc.Matcher(nodes_to_ban),
                        term_if_matches=True,
                    )
                )
            )

        return matcher_extender

    return extender_factory


def eval_on_toks(c: rc.Circuit, toks: torch.Tensor):
    group = rc.DiscreteVar.uniform_probs_and_group(len(toks))
    c = c.update(
        "tokens",
        lambda _: rc.DiscreteVar(rc.Array(toks, name="tokens"), probs_and_group=group),
    )
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    results = transform.sample(c).evaluate()
    return results


def get_attention_pattern(
    c: rc.Circuit,
    heads: List[Tuple[int, int]],
    toks: torch.Tensor,
    add_value_weighted=False,
):
    assert toks.ndim == 2
    seq_len = toks.shape[1]
    attn_patterns = torch.zeros((len(heads), len(toks), seq_len, seq_len))

    for i, (l, h) in enumerate(heads):
        a = rc.Matcher(f"a{l}.h{h}").chain(rc.restrict("a.attn_probs", term_if_matches=True, end_depth=3))
        pattern_circ = a.get_unique(c)
        attn = eval_on_toks(pattern_circ, toks)

        if add_value_weighted:
            v = rc.Matcher(f"a{l}.h{h}").chain(rc.restrict("a.v_p_bias", term_if_matches=True, end_depth=3))
            values = v.get_unique(c)
            vals = eval_on_toks(values, toks)
            vals = torch.linalg.norm(vals, dim=-1)
            attn_patterns[i] = torch.einsum("bKQ,bK->bKQ", attn, vals)
        else:
            attn_patterns[i] = attn

    return attn_patterns


def await_without_await(func: Callable[[], Any]):
    """We want solution files to be usable when run as a script from the command line (where a top level await would
    cause a SyntaxError), so we can do CI on the files. Avoiding top-level awaits also lets us use the normal Python
    debugger.
    Usage: instead of `await cui.init(port=6789)`, write `await_without_await(lambda: cui.init(port=6789))`
    """
    try:
        while True:
            func().send(None)
    except StopIteration:
        pass


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def imshow(tensor, center_zero=True, zrange=None, color_continuous_scale="RdBu", **kwargs):
    if center_zero:
        return px.imshow(
            to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale=color_continuous_scale, **kwargs
        )
    elif zrange is not None:
        zmin, zmax = zrange
        return px.imshow(
            to_numpy(tensor), zmin=zmin, zmax=zmax, color_continuous_scale=color_continuous_scale, **kwargs
        )
    else:
        return px.imshow(to_numpy(tensor), color_continuous_scale=color_continuous_scale, **kwargs)


def show_diffs(
    diffs, center_zero=True, zrange=None, title="", xlabel="predicted year", zlabel="logit change", dim=500, **kwargs
):
    return imshow(
        diffs,
        center_zero=center_zero,
        zrange=zrange,
        height=dim,
        width=dim,
        title=title,
        labels={"x": xlabel, "y": "YY", "color": zlabel},
        y=[str(i) for i in range(2, 99)],
        **kwargs,
    )


def split_mlp(mlp_node: rc.Circuit, important_neurons: torch.Tensor) -> rc.Circuit:
    activations, weights = mlp_node.children
    (pre_activations,) = activations.children
    device = "cpu" if mlp_node.device is None else mlp_node.device

    unimportant_neurons = torch.tensor(list(set(range(3072)) - set(important_neurons.tolist())))
    if len(important_neurons) == 0:
        return mlp_node.update(
            rc.restrict("m.pre", end_depth=3), lambda node: rc.Index(node, [S[:], S[:]], name="m.pre_unimportant")
        )

    if len(unimportant_neurons) == 0:
        return mlp_node.update(
            rc.restrict("m.pre", end_depth=3), lambda node: rc.Index(node, [S[:], S[:]], name="m.pre_important")
        )

    important_neurons = important_neurons.to(device)
    unimportant_neurons = unimportant_neurons.to(device)

    important_pre_activations = to_device(
        rc.Index(pre_activations, [S[:], important_neurons], name="m.pre_important"), device
    )
    unimportant_pre_activations = to_device(
        rc.Index(pre_activations, [S[:], unimportant_neurons], name="m.pre_unimportant"), device
    )

    important_activations = rc.gelu(important_pre_activations, name="m.act_important")

    unimportant_activations = rc.gelu(unimportant_pre_activations, name="m.act_unimportant")

    important_weights = to_device(
        rc.Index(weights, [S[:], important_neurons], name=f"{weights.name}_important"), device
    )
    unimportant_weights = to_device(
        rc.Index(weights, [S[:], unimportant_neurons], name=f"{weights.name}_unimportant"), device
    )

    mlp_important = rc.Einsum.from_einsum_string(
        "pi,oi->po", important_activations, important_weights, name=f"{mlp_node.name}_important"
    )
    mlp_unimportant = rc.Einsum.from_einsum_string(
        "pu,ou->po", unimportant_activations, unimportant_weights, name=f"{mlp_node.name}_unimportant"
    )

    mlp_reconstructed = rc.Add(mlp_important, mlp_unimportant, name=mlp_node.name)
    return mlp_reconstructed

#%%
def mean_logit_diff(logits: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    diffs = []
    for logit, year in zip(logits, years):
        diffs.append(logit[year + 1 :].sum() - logit[: year + 1].sum())
    return torch.tensor(diffs)

def cutoff_sharpness(logits: torch.Tensor, years: torch.Tensor = torch.arange(2, 99)) -> torch.Tensor:
    sharpness = logits[torch.arange(len(logits)), years + 1] - logits[torch.arange(len(logits)), years - 1]
    return sharpness

def prob_diff(probs: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    diffs = []
    for prob, year in zip(probs, years):
        diffs.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
    return torch.tensor(diffs)