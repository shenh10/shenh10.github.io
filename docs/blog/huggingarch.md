---
date: 2026-08-04
title: "HuggingArch: Automating Model Architecture Analysis"
description: A harness that makes LLM inference cost analysis automatic, verifiable, and reusable — as long as the model is open on HuggingFace.
---

# HuggingArch: Automating Model Architecture Analysis

## Intro

> This is not a polished project launch, but the scaffolding works well enough to be worth sharing. If you find the project interesting or useful, stars are welcome, and so are developers — though I haven't yet worked out what healthy co-development looks like for a vibe-coded project. That's a question I'd genuinely like to discuss with anyone who has ideas.

HuggingArch takes a harness approach to a specific problem: automating the arithmetic of LLM inference cost — for any model that is open-sourced on HuggingFace.

The motivation is straightforward. Models keep multiplying, and anyone working in this field constantly needs to analyze and compare them. It's a critical question for model labs, MaaS providers, and silicon vendors alike. Doing the arithmetic is laborious, and the barrier to doing it *correctly* is not low. Over the past year, labs have gone all-in on hybrid architectures, and structural diversity has grown sharply — which makes the arithmetic harder, not easier. Which is cheaper at long context, DSA or KDA? How much does CSA save over DSA compared to HSA? How many more TPM can MiniMax and StepFun support by deploying at the 200B scale? How much does K3, at 3T, cost relative to its predecessors? Just enumerating these questions suggests weeks of work. It's 2026 — to keep our own working memory from OOM-ing, we should let agents do the arithmetic.

The project currently covers computation-graph visualization of model architectures, plus theoretical cost analysis for LLM inference. Ahead lies operator benchmarking at specific shapes and cross-framework operator performance reporting, so that deployment performance for a given configuration can be obtained quickly. In principle HuggingArch could go further: turn each model's modules into pluggable units so that an architecture designer can freely change hyperparameters, recombine module units into a new model, and immediately compute what that model would cost across GPU SKUs and deployment strategies. That's where the project's imagination really lies.

This is a pure side project. From first idea to now it has taken the better part of six months, stitched together in commute gaps and weekends — during which model capability itself advanced dramatically. Today, getting a frontier model to analyze one model's KV cache and theoretical FLOPs is no longer hard. So what is the point of this project? In short: I believe the value is in being a **trustworthy**, **self-correcting**, **reusable**, **lightweight** piece of infrastructure.

- **Trustworthy.** Pure vibe analysis of a model is ad hoc. An agent's analysis has to be traced through its derivation before you can judge whether the result is credible — and agents are, by disposition, better at telling you the answer than the intermediate steps. Without feedback, expecting a single turn to land on the correct result is unrealistic, and the probability drops as the model gets more complex. Human feedback is an inefficient loop: is it wrong, where is it wrong, why is it wrong — interrogating an agent repeatedly only lengthens the conversation, and the repeated answers feel less trustworthy each round.

- **Self-correcting.** HuggingArch is built around a validation system. Using ground truth — the model's own forward source, its config, its weights — the model can correct itself, and elementary errors are almost entirely eliminated. Beyond that, the whole system computes on top of sympy's symbolic algebra, so every value traces back to a specific formula, passed straight through from the backend engine to the frontend. Any formula that is wrong can be caught by eye.

- **Reusable.** Every model enters the library as a spec. Specs are abstracted into three levels — primitive, component, model — so shared building blocks like attention and MoE are reusable across models. That lowers the difficulty of generating a spec, reduces ambiguity in the computation, and keeps entropy growth in the repository within manageable bounds. Once models are assembled from components in the repository, one can in principle grant that structure nearly the full capability of a graph compiler — shape propagation, quantization, activation liveness analysis, device mapping — the only difference being that the bottom layer computes FLOPs and memory traffic rather than actual tensors. From a simple, agent-authored spec grows a sophisticated model analysis system.

- **Lightweight.** None of this analysis requires the model to run. No kernel implementations, no framework runtime, no GPU. Models of any size can be analyzed quickly, and every result stays traceable indefinitely.

An LLM inference cost system needs two things.

**1. A representation of the model architecture.**

You *can* build one by hand, but reconstructing a model manually is tedious, and it is exactly the kind of work an agent can do. To build this system, an agent needs: weight shapes, computation-graph structure, and model parameters. HuggingFace happens to have all three — model weights, the transformers library, and inference source code. So the web system is built on HuggingFace, though there is also a backend CLI for DIY use (vLLM / SGLang source plus model weights work too).

**2. A methodology for the inference arithmetic.**

Speed-of-Light (SOL) estimation is the computational foundation. LLM inference consists of prefill and decode. The compute pattern of each phase is determined by the operators that dominate its forward time. Prefill is generally compute-bound, because GEMM and attention operators both have favorable roofline arithmetic intensity, so GPU TFLOPS bound the lower limit of execution time. Decode's GEMM and attention operators are generally memory-bound, so bandwidth sets the floor. A good reference for the definition is [SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits](https://arxiv.org/pdf/2603.19173).

$$T_{\mathrm{SOL}} =
\max\left(
\frac{\text{Total FLOPs}}{\text{Compute Throughput}},
\frac{\text{Total Fused Bytes}}{\text{Memory Bandwidth}}
\right)$$

From each operator's $T_{\mathrm{SOL}}$, propagating bottom-up through the computation graph yields the SOL time of the prefill and decode phases, and from there an estimate of inference cost.

---

## The Wrong Paths

Before the main story, a word about the detours.

The idea was appealing; the execution was hard. The goal was a lightweight, GPU-free, cross-model general analyzer:

- Downloading model weights was out of the question — multi-terabyte downloads simply don't parallelize.
- Actually running the models was out too. GPU cost is more than I can pay out of pocket, and the combination of CUDA versions and engine versions is far too heavy to stand up.

The intuitive approach was to use transformers' accelerate and the meta-device mechanism: hook each layer before execution to count FLOPs, and recover model structure through graph analysis. But any approach that depends on the runtime simply doesn't work.

### 1) The meta-device construction mechanism can't handle instantiation inside `forward`

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

`from_config` runs to completion and gives you a fully structured `nn.Module`. But the approach has an inherent limitation: any attempt to forward or actually compute raises an error — some models call `.cpu()` inside `forward`, which fails immediately.

### 2) Depending on the PyTorch computation graph is not a good idea

Model structure can only be inferred from the `forward` function, but extracting the computation graph from PyTorch means retracing the whole path of PyTorch compilation:

- `torch.export`-based approaches can't handle data-dependent control flow, and the operators they emit are too fine-grained to interpret.
- trace / dynamo approaches need mock inputs, but signatures aren't guaranteed to match, so mocking is hard. When running attention with a KV cache, for instance, different models have different forward signatures, and constructing the data is difficult:

```python
class MiMoV2Attention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class GlmMoeDsaAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ...
```

- AST analysis is excessively complex and drags in a large amount of Python-primitive handling.

### 3) transformers has poor backward compatibility, so model runtimes often can't start at all

Take MiMo-V2-Flash. transformers 5.x refactored RoPE configuration from a flat `rope_scaling` field into a nested `rope_parameters: {default: {...}}` dictionary. MiMo-V2-Flash's modeling code was written against 4.40.1, so it errors:

```
modeling_rope_utils.py:1021: FutureWarning: `rope_config_validation` is deprecated
and has been removed. Its functionality has been moved to
RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this
class, so please call self.validate_rope() instead. Also, make sure to use the
new rope_parameters syntax. You can call self.standardize_rope_params() in
the meantime.
```

Each model pins a transformers version in `config.json`. But even with dynamic multi-version routing to the right transformers release, a model like Kimi will fail in a CPU environment because its `modeling.py` imports flash_attention directly.

---

## I. Building the Model Spec

HuggingArch's core architecture is a bottom-up DAG DSL, plus a SpecTree IR built on that DAG. The DAG is written by Claude Code; the SpecTree IR is built-in framework code. This keeps the agent exercising its real strength — turning unstructured data into structured data — but always under constraint. It's worth restating that the computation graph is the core abstraction of modern deep learning frameworks; transformers is just one particular structure built on that abstraction. Construct the computation graph well and you can, in principle, represent arbitrarily complex model structures. Hang a sympy symbolic system on that IR and you can express structural relationships, shapes, and arithmetic simultaneously. Given such a description, higher-level task-specific computation — KV cache capacity derivation, parallelism derivation — is merely an application layer on top of the spec system.

### Core: The Spec System

HuggingArch constructs model structure from transformers, or from the model's own repository inference source. This is an open-ended code-reading problem, which requires a well-specified DSL so that an agent (Claude Code, say) knows how to write the structure.

#### 1) A layered op system: Primitive / Component / Model

HuggingArch organizes models into a **three-tier op system** — primitive op, component (compound op), and model — each tier with its own syntax and responsibility boundary.

**Primitive ops** are the lowest-level operator units, defined in `backend/arch_spec/ops.py`. Each primitive registers via `@register_op()` and carries three core attributes: a parameter-count formula (`params_fn`), a FLOPs formula (`flops_fn`), and a shape template. These formulas aren't strings — they're lambdas over sympy Symbols. `linear`, for instance, is the symbolic expression `2·B·S·In·Out`. There are currently 60-odd primitives, covering `linear`, `embedding`, `rmsnorm` / `layernorm`, `attention_score`, `softmax`, `attention_apply`, the activations (silu / gelu / relu² / sigmoid …), `rope`, `add` / `mul`, `split_heads` / `merge_heads`, and so on. This tier is closed — callers may not extend it — because the lower the operator, the more its FLOPs and parameter formulas must be trustworthy.

**Components (compound ops)** are the tier that actually carries architectural diversity. Each component is a YAML file, physically split across **three directories** — directory location *is* the tier declaration, with no extra field and no tag:

- `backend/arch_spec/components/library/`: genuinely distinct topological building blocks. Anything that encodes a new op sequence, new connectivity, or a variant that existing structures can't express through binding/rename alone lives here. GQA / MLA / SwiGLU / MoE / `pre_norm` / `gemma4_4norm_scaled_block` / `dsv4_csa_attention` — all at this tier.
- `backend/arch_spec/components/adapters/`: thin shims of pure binding + rename — single-node wrappers that bind a library component to a specific checkpoint's naming conventions (`wq` vs `q_proj`, stacked-Parameter experts vs ModuleList experts, etc.). An adapter introduces no new graph nodes; if you need a second node, that work belongs promoted back to library.
- `backend/arch_spec/components/drafts/`: the staging area for the spec-gen agent. Drafts are parsed at startup but **do not enter the global namespace** — they're visible only to model specs that explicitly declare `drafts: [<name>]`. A bad V4-Pro draft can't contaminate the next spec's manifest. Once reviewed, `python -m backend.arch_spec.promote <name>` does a plain `git mv` into `library/`.

Each component describes a **DAG**:

- `inputs` / `outputs`: the tensors this component exposes to the tier above.
- `graph`: a dict of DAG nodes. A node's `op` field may point at a primitive, or recursively at another component, giving unbounded nesting.
- `in:` per node: declares which upstream nodes to pull data from — **the engine genuinely propagates shapes along `in:`** (details in the next section).
- `dims`: the node's local dim bindings (e.g. `out: "N * D"`), which drive sympy shape inference together with upstream shapes.
- `repeat` / `active`: describe sparse topologies like MoE, where you declare one expert and activate k of them.
- `params`: parameterized slots, letting attention type, FFN type, norm type and so on become injectable roles, so the tier above can reuse a single block template.
- `parallelism.shard_symbols`: declares which dimensions are shardable under TP; the inference analyzer reads this field directly for weight sharding.
- `kv_label` / `kv_window` / `attn_variant` (attention components only): tag KV cache topology / window / variant. KV cache modeling and the frontend badges are all driven by these — adding a new linear attention requires only one `kv_label:` line in the component YAML, and every downstream tool recognizes it automatically.

For example, `gqa_attention.yaml` looks like this:

```yaml
name: gqa_attention
role: attention
description: "Plain Grouped-Query Attention with no QK-norm. Llama, Qwen2, Mistral."
inputs: [x]
parallelism:
  shard_symbols: [N, Nk]                       # TP shard axes: Q along N, K/V along Nk
kv_cached: [k_split_heads, v_split_heads]      # KV cache takes shape_out of these two nodes
kv_label: GQA
attn_variant: GQA
graph:
  q_proj:          { op: linear, in: x, dims: { in: H, out: "N * D",  bias: Qb } }
  k_proj:          { op: linear, in: x, dims: { in: H, out: "Nk * D", bias: Qb } }
  v_proj:          { op: linear, in: x, dims: { in: H, out: "Nk * D", bias: Qb } }
  q_split_heads:   { op: head_split, in: q_proj, dims: { N: N,  D: D } }
  k_split_heads:   { op: head_split, in: k_proj, dims: { N: Nk, D: D } }
  v_split_heads:   { op: head_split, in: v_proj, dims: { N: Nk, D: D } }
  q_rope:          { op: rope, in: q_split_heads, dim: D }
  k_rope:          { op: rope, in: k_split_heads, dim: D }
  qk_score:        { op: attention_score, in: [q_rope, k_rope], dims: { N: N, D: D } }
  softmax:         { op: softmax, in: qk_score, dims: { N: N } }
  attn_apply:      { op: attention_apply, in: [softmax, v_split_heads], dims: { N: N, D: D } }
  o_merge_heads:   { op: head_merge, in: attn_apply, dims: { N: N, D: D } }
  o_proj:          { op: linear, in: o_merge_heads, dims: { in: "N * D", out: H } }
outputs: [o_proj]
```

A few things worth noting:

- **`role: attention`** declares that this component can fill the `attention` slot in a block template above. Permitted roles are `attention` / `ffn_dense` / `ffn_moe` / `router` / `block` / `aux_block`; omitting the field marks a sub-component or utility (compressors, indexers, `head_split` — things that can't be slotted in directly).
- **`graph` is a dict, not a list.** Each key is a node name and the value describes the op type and its dependencies (`in:`). Nodes reference each other by name, forming the DAG. The `dims` field takes sympy expression strings directly (`out: "N * D"` parses into the symbolic expression `N·D`).
- **`parallelism.shard_symbols`** declares which dimensions shard under TP (`N` for Q heads, `Nk` for KV heads); the inference analyzer reads this directly for weight sharding.
- **`kv_label` / `kv_cached` / `attn_variant`** are the tags attention components provide for KV cache modeling — adding a new linear attention needs only a `kv_label` declaration, and every downstream tool knows which KV topology family it belongs to.
- **`bias: Qb`** turns "does this have QKV bias" into a parameterized switch via an optional dim — the model spec above sets `Qb=0` or `Qb=1` and reuses the same component.

The whole graph is 12 nodes threading the standard GQA flow: `q_proj → head_split → rope → attention_score → softmax → attention_apply → head_merge → o_proj`. This component system is open, which is why new models can be built compositionally: MQA sets `Nk` to 1, MHA sets `Nk` equal to `N`, partial-RoPE makes one more variant replacing the `rope` node.

> **Encapsulation parity.** When an agent writes topology, it easily loses semantic information — an unfamiliar semantic module that is a standalone layer in the source gets inlined and flattened, and the human-friendly semantic structure is destroyed. In the prompt we constrain spec encapsulation granularity to be no finer than that of the source `forward()`. Take DeepSeek-V4-Pro: the upstream author wrote `def hc_pre(...)`, which means they drew a line saying "this is an atomic operation, callable from multiple places, independently testable." Inlining it produces a pile of poorly readable atomic modules, which badly hurts our ability to read and analyze the structure. A spec may be **more abstract** than the source (encapsulating a repeated implementation into one component), but never **more fragmented**. This gives the agent a principled decomposition granularity and produces a better architectural drawing style.

**The model tier** is defined in `ArchSpec` and describes: how many block templates does this model have? Which block does each layer instantiate? Which component fills each attention/FFN slot inside a block? `qwen2.yaml`, for instance, uses one `block` template plus `params: {attention: gqa_attention, ffn: swiglu_ffn}` and expands the layer assignment across `L` layers; a hybrid model (MLA on some layers, SWA on others) writes different templates per layer index in the `model` node. The analyzable scope is currently fixed to decoder-only (multimodal) LLMs; for the schema boundary and design trade-offs, see [`docs/design.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/design.md) §1.4.

#### 2) A bottom-up, three-tier symbolic system

I want HuggingArch to produce not just numbers but **explainable formulas** — a capability a cost-accounting system has to have, or interpretability simply doesn't exist. I find it hard to trust a black-box planner, because any assumption it makes may be buried in implementation detail, and large-model infrastructure changes constantly. I don't demand that a cost system model everything, but I must know which assumptions it makes; only then can I judge the credibility of the numbers it emits. Hovering over any value in the frontend and seeing the symbolic expression behind it is, to me, the ideal presentation. That requires a symbolic algebra threaded through all three tiers of the op system.

Each primitive's `flops_fn` / `params_fn` accepts a uniform set of dimension symbols (`B` / `S` / `H` / `N` / `Nk` / `D` / `I` / `V` / `L` / `E` / `R` / `P` / `Dv` …) and returns a sympy `Expr`. When components nest, the outer tier pushes dim bindings into scope and the inner tier picks them up and continues expanding — the whole graph is ultimately one sympy expression tree.

**Dual-track evaluation is the biggest payoff of this design.** The same `Expr`, given a `Symbol`, returns an algebraic expression (`2·B·S·H²`); given an `int`, it returns a number. The engine has exactly one implementation — symbolic versus numeric depends purely on input type. No two engines, no mid-stream "is this symbolic?" branch. What the frontend gets on hover is two synchronized render trees, formula and concrete value side by side, with no awkward gap between the number and the formula that supposedly produced it.

**Aggregation at the model tier deliberately preserves heterogeneous structure.** A hybrid model ends up with a sum like `Ld·MLA_FLOPs + (L − Ld)·MHA_FLOPs` rather than being flattened into a single formula — so the inference analyzer can correctly add up heterogeneous attention when computing KV cache, weights, and traffic.

But the symbolic system only solves "how the numbers are computed"; it doesn't solve "whether the graph itself is right." That's the job of the next two sections. (As we'll see, the real harness grows further guards on top of these two skeletal ground truths.)

### Where Ground Truth Comes From

To make spec generation both precise and cheap, HuggingArch pulls facts from several sources on the HF model — while **downloading no weights at all**:

- **`config.json`.** Via `_fetch_config()` in `backend/analyzer.py`, trying in order: the local custom-model cache, the built-in snapshot (`backend/arch_spec/snapshots/`), and online HF Hub download. This is the scalar source for model dimensions (H / L / N / I / V, etc.).
- **The model skeleton printout.** `_fetch_model_str()` uses `init_empty_weights()` to construct an empty model on the meta device, then `print(model)` to get the module hierarchy. This downloads no weights and runs no forward — it exists only to obtain the nested `nn.Module` name structure and shape annotations. Results are cached in `~/.cache/huggingarch/model_info/`.
- **Forward source.** `backend/hf/forward_source.py` uses `inspect.getsource()` to pull the forward implementations of key classes — `DecoderLayer`, `Attention`, `MLP`. This is the only credible factual source describing operator connectivity, and it goes to the spec-generation agent to prevent hallucination.
- **The safetensors index.** `backend/hf/metadata.py` (a fairly large module) uses only 2–3 HTTP range requests to fetch `model.safetensors.index.json` and each shard's metadata block, yielding **every tensor's name, shape, dtype, and storage bytes** — without downloading a single weight byte. This is the ground truth for parameter count and byte validation.

For **private repositories that aren't in the transformers mainline and are only provided via trust_remote_code in the model card** (typically Kimi, MiMo, GLM-MoE and similar), HuggingArch fetches `modeling_*.py` and `configuration_*.py` from the HF Hub repo and feeds the forward source to the agent. Native dependencies like `flash_attn` are never actually imported — they're read as text. That sidesteps the "runtime won't run" problem entirely.

With the factual sources in place, the next two sections describe how each becomes a check: forward source plus model skeleton covers the forward data structure; the safetensors index covers parameter count and quantization.

### Shape Inference: Ground Truth for the Forward Data Structure

When an agent writes a spec, the easiest mistake isn't a wrong parameter count — that kind of error is caught downstream by tensor comparison. The easiest mistake is **geometric**: MLA's `k_pe` gets concatenated without being broadcast to all heads; a view_split happens on a flat dim that isn't actually `N·D`; Q and K head_dims don't line up. These errors can **cancel out exactly** in parameter count — one linear over-counts, another under-counts, and the byte diff is still 0 — while the forward dataflow is wrong. The spec appears to pass validation, yet the inference analyzer produces an absurd KV cache, the wrong attention type, and shape badges that don't agree with each other.

**The fundamental difficulty: HuggingArch doesn't run the model, so how can it verify the forward topology is correct?**

The first version took a detour: have each component declare a `shape:` per node and let the engine check consistency. But that pushes the burden onto the spec author — for a V4-Pro block with 60 nodes, the author has to work out every node's shape and write it into YAML, which is more error-prone than writing the forward itself.

**The final path makes shape inference a first-class mechanism:**

1. **`in:` isn't a comment — it's a real dataflow edge.** The engine follows each node's `in:` to look up the upstream NodeResult in the evaluated table and feeds its `shape_out` into the current op's shape function.
2. **Every primitive declares a `shape_kind`** — currently 7 of them (`preserve` / `axis_replace` / `axis_concat` / `contract` / `permute` / `source` / `explicit`) — and the engine applies the corresponding rule for shape inference. Adding an op is **declarative**: we don't let each op write its own mini shape solver, which would make errors hard to localize and raise the cost of onboarding new ops.
3. **Unified across scales.** Between nodes inside a component, across nested component calls, and across the model tier (each layer's `shape_out` feeding the next layer's `inputs[0]`) — all three scales share one dataflow propagation. V4's `[B, S, Hc, H]` hyper-connection input flows through the entire graph, and what the engine sees is the real tensor form, with nothing for downstream to guess backwards.

Only after shape becomes first-class does the key capability appear: **the validate phase can perform geometric consistency checks without evaluating any numerics**. Every multi-input primitive (`axis_concat` / `attention_score` / `contract` / multi-input `add` / `mul`) has its upstream shapes verified — is the rank consistent, are the non-axis dims equal, do Q and K head_dims align. Geometric errors are caught as dataflow advances, rather than inferred backwards from "the bytes match but the attention looks strange."

**The real value of this path** is turning "the forward data structure" from an implicit property that can't be checked directly into a first-class artifact that can be validated independently. The ground truth is the connectivity described by upstream forward source; once the spec is written, shape inference plus geometric consistency checking is the local reproduction of that ground truth.

### Tensor Weights: Ground Truth for Parameter Count and Quantization

LLMs writing specs that look plausible but are wrong somewhere is a daily occurrence. But as long as the weights are still in the checkpoint, **the truth is right there** — safetensors doesn't lie, and its tensor names and byte counts are facts the upstream model team put there by hand. The question is how to turn that fact into a check.

#### One fact, two layers of information

HuggingFace tensor naming follows a fairly stable set of conventions:

```
model.layers.X.self_attn.q_proj.weight                         ← standard LLM
model.layers.X.mlp.experts.X.gate_proj.weight                  ← MoE
model.vision_tower.encoder.blocks.X.attn.q_proj.weight         ← vision tower
model.language_model.layers.X.self_attn.q_proj.weight          ← multimodal nesting
```

This naming encodes **two kinds of information**, and the validation mechanism handles each along its own path.

**Topological information** (which layer, which attention slot, which expert) is expressed by the hierarchy of the name prefix. The validator extracts a "block-internal suffix" from these patterns (`self_attn.q_proj` / `mlp.experts.X.gate_proj` / `attn.q_proj`) and then suffix-matches against the hierarchical paths of spec nodes, longest-suffix-first. In a heterogeneous stack the same module path may be claimed by several spec nodes at once (`input_layernorm` appears Ld times in `dense_block` and L−Ld times in `moe_block`), so the matcher sums all hits before reconciling.

**Quantization packing information** (is this GPTQ qweight, MXFP4 `_blocks`, or FP8 `.weight` + `scale_inv`) is expressed jointly by the name suffix and the dtype. Here HuggingArch's strategy is to write **each quant scheme's on-disk representation as YAML**: the main weight's packing factor, storage dtype and suffix; the metadata siblings' suffixes and cardinality formulas; the rule for determining logical element format; and the scale granularity type.

That sentence sounds unremarkable, but it's a **deliberate boundary**. Tensor naming conventions and packing details — the kind of scattered knowledge where upstream transformers, autoawq, compressed-tensors and various training frameworks each do their own thing — were previously spread across several Python libraries, so each new scheme meant an invasive change somewhere. Once unified into YAML, **adding a quant scheme is a YAML declaration, not a Python change**, and all downstream consumers share one factual source.

#### Decoupling quantization: architecture vs storage

The complication with quantized models is that model architecture and quantization strategy are **two independently composable things**. The same DSv3 may have a BF16 checkpoint, an FP8 checkpoint, and an INT4 checkpoint; the same checkpoint may also be asked "what if it were requantized to W4A16?" HuggingArch's answer:

- **The architecture tier is dtype-agnostic.** The `linear` op in a spec cares about parameter count and FLOPs, which are geometric facts; the formula's free symbols contain no dtype.
- **The storage tier is dtype-dependent and expressed separately.** A spec carries a `quant_context` section declaring quantization schemes along the role dimension, plus a set of overrides for edge cases that per-role can't express.

For instance, a mixed deployment like gpt-oss — quantize only MoE, leave the rest BF16 — is declared concisely:

```yaml
quant_context:
  per_role:
    ffn_moe: { method: mxfp4, group_size: 32 }
    default: { method: none }
  overrides:
    - tensors_re: 'layers\.X\.mlp\.router\.weight$'
      dtype: BF16
  default_dtype: BF16
```

Mixed-precision models like DSv3 and Kimi-K2.5, which pick separate quantization strategies for attention / FFN / router / vision_tower / lm_head, take the same form — `per_role` expresses the overall scheme, `overrides` handles the boundaries. `materialize.py` is the **single path** that lowers `quant_context` onto concrete tensors: it takes the logical weight inventory parsed from the spec, plus `quant_context`, plus the scheme YAML, and emits each tensor's physical dtype and byte count. The validator, weight breakdown, KV cache, and frontend badges all consume that one output — there is no fast path around it. This is a deliberate restraint: the moment quantization is implemented in several places, mixed-precision models make the two sides disagree immediately.

A useful by-product is that a spec can answer two questions at once: **what it actually is** (reconcile the spec's own `quant_context` against the real checkpoint), and **what deploying at a different dtype would look like** (the Inference page detaches from `quant_context` and lets the user pick any deployment dtype; the sympy formulas simply re-evaluate). Both paths share one spec and one symbolic algebra.

#### The backstop: bytes-vs-bytes diff

The final hard criterion is whether **storage_bytes** match. On the spec side, materialize gives each leaf its true on-disk byte count; on the HF side, the per-tensor byte count comes from the safetensors header. The reason for choosing bytes over numel: in a mixed-precision checkpoint (FP4 routed experts, FP8 attention, and BF16 norms in one file), **physical bytes are the only unit that needs no conversion**. Any mismatch is written back to the agent in structured form — which module, how many bytes HF has, how many the spec has, what the diff is. The self-correction loop rests on feedback at exactly that granularity. This backstop also indirectly covers coefficient-level shape errors: get an `out_features` coefficient wrong and the parameter count is immediately off, so the feedback the agent receives is "I claimed `self_attn.o_proj` has X, it actually has 1.5X" — far more actionable than a solver reporting "shape mismatch."

### Two Primary Ground Truths Form a Guard Harness

The two most fundamental factual paths, stated plainly — they are the skeleton of the whole harness:

|  | Ground truth source | How HuggingArch reproduces it | Error class caught |
|--|------|------|------|
| **Forward topology** | Connectivity described by upstream `modeling_*.py` | Shape inference + geometric consistency checks on multi-input ops | Topological errors (rank mismatch, Q/K head_dim mismatch, missing broadcast) |
| **Parameter count + quantization** | Tensor names + byte counts in `safetensors.index.json` | Tensor name matching + materialize → bytes diff | Quantity errors (wrong coefficient, missing module, mismatched quant scheme) |

The two are complementary: geometric errors are caught by the shape consistency check as dataflow advances, **independent** of whether parameter counts happen to cancel; quantity errors are caught by the bytes diff before ship, **independent** of whether the geometry happens to be self-consistent. The agent doesn't need to "guess right" — it only needs to correct from feedback and re-run, and the loop converges on its own. This mechanism is the core of the project as harness engineering: **validation is more expensive and harder than generation, but once validation works, generation can be handed to a probabilistic model with confidence.**

One clarification, though: as models grew more complex, more checks grew on top of these two primary ground truths. When we later ran the guard-ablation experiment (see the postscript), we split the harness into **5 independently toggleable guards**:

- **`weight`** — the bytes / packing / tensor matching above (factual source: safetensors).
- **`shape`** — the geometric consistency above (factual source: forward topology).
- **`source`** — diff the spec's call structure directly against the forward source: call order, and whether the number of residual adds is right. **Correct shapes don't imply correct connection order** (factual source: forward source).
- **`axiom`** — treat the per-layer invariants pinned down by the forward source (typically KV growth: how much a layer caches per token, whether SWA actually caps the window) as checkable assertions, targeting semantic errors bytes can't reveal. V4's "sliding window not capped → cache blowing up to the 128TB range" was caught by this one (factual source: forward source).
- **`structural`** — per-module executability plus one **independent factual source**: the per-layer config lists in `config.json` (`moe_layer_freq` / `mlp_layer_types` / `sliding_window` patterns…). A spec's per-layer templates must **exhaust** every value those lists actually take; a `rest:` fallback may not silently swallow an unmodeled layer type (factual source: config.json).

In other words, **"two ground truths" is the skeleton but not the whole story**: the forward source alone underpins three distinct checks (shape / source diff / axiom), and `config.json` is not merely the source of scalar dims — its per-layer lists are themselves an independent gating fact. The "94% with no guards → 6% with all guards" curve in the postscript measures exactly how agent output credibility changes as these 5 guards are reinstalled one notch at a time.

The edge cases those guards don't cover are handled by a light layer of schema validation: `template:` references must resolve, model-tier `inputs:` must form a forward DAG (no referencing later layers, no cycles), roles must be on the whitelist. **But the schema doesn't block new topology** — when an agent hits a structure the library lacks, it inlines a new block in its own spec's `blocks:` section, and the schema passes once it sees a local definition. The cost of novel topology should be "write a new YAML," not "PR the library first, then write the model." The schema rejects bad references, not innovation.

### The Generation Pipeline for a New Model Spec

Back to the original goal: input an HF model ID, output a verified spec. `backend/spec_worker/` orchestrates this as an agentic pipeline. Most new models are topologically a recombination of existing components (a hybrid GQA + MLA, SwiGLU + 256-expert MoE, add sliding window, swap the RoPE), so the agent only fills the attention / ffn / norm role slots in the model spec with existing components, while the thin shims in `adapters/` absorb naming differences between checkpoints of the same topology. When there genuinely is novel topology (DSv4's hyper-connection, CSA's sparse indexer, SSM), the agent first writes a new component in `components/drafts/` and declares it within this spec; after review, `promote` merges it into library with a single `git mv`. Extending the system and using the system are protected by the same validation, and an unreviewed draft is visible only to the spec that declares it, never polluting the global manifest.

The pipeline assembles the agent's prompt automatically: op registry, component manifest, schema and verified examples, plus this model's ground truth (forward source, `config.json`, safetensors tensor inventory). It then spawns an agent backend (`claude` / `codex` / `kimi`, abstracted in `backend/agents/base.py`) that writes YAML in a sandbox, calls `validate`, and corrects from unmatched-tensor / shape-mismatch / bytes-diff feedback until `bytes_diff == 0` or the round limit is reached — a fully automatic refinement loop. Once validation works, generation can be handed to a probabilistic model with confidence.

---

### Are the Guards Necessary? A From-Zero Ablation

How much this pipeline's output can be trusted comes down entirely to whether that set of guards actually stops errors. The natural objection is that the guards are redundant — surely a strong enough agent writes a correct spec without them? We answered it as a controlled ablation (`experiments/guard_ablation/`; full report in [`docs/guard_ablation/report.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/guard_ablation/report.md)), from two complementary angles.

**First, deterministic unique-coverage analysis.** For each guard, inject the class of error it specifically defends against into an already-committed correct spec, then test with the **full** guard set — and see whether it is the only one that catches it. Across 14/14 live units, every gating guard is the **unique catcher** on the fault it guards; not one is covered by another. On a model that declares no axioms at all (MiMo), the SWA `kv_window` contract is the **only** backstop against sliding-window KV blowing up as dense storage.

**Second, a fair from-zero agentic ablation.** The spec-writer agent (Claude Opus 4.8) rewrites each model's spec **from scratch** in a **decontaminated** sandbox: the target's own component, every same-family component in the library, all other model specs, other snapshots, fusion plans, and the primitives for novel operators are all stripped away — the agent is left with generic building blocks, primitive ops, and the target's ground truth (forward source, `config.json`, tensor names), and has to **derive** the architecture itself, re-declaring novel operators as in-spec `custom_ops` (an escape scan confirms no run left the sandbox). Guards are added back one tier at a time — `bare → +weight → +shape → +struct/source → +axiom (full)` — with K=3 seeds per tier, and the oracle is the full-guard validator.

The result is a clean severity ladder: each guard added back removes exactly its own class of error.

![Guard-necessity ablation: violation severity for three models (MiMo-V2.5-Pro / DeepSeek-V4-Pro / GLM-5.2) under opus-4.8, as guards are added back tier by tier from bare to full (mean over K seeds). Taller bars mean more wrong; %✓ is the share of seeds that fully pass at that tier.](/blog/huggingarch/error_distribution.png)

At the bare tier V4-Pro is a single red bar of 32.7 STORAGE violations and 1.8 GB of byte error, collapsing to zero as weight / shape / schedule / axiom come back; each guard eats only its own colour, never another's:

| Model | Bare (no guards) | Full (all guards) |
|---|---|---|
| **DeepSeek-V4-Pro** (1T, Hyper-Connections) | 0% valid — 32.7 STORAGE + 7.3 NAMING + 4.3 SHAPE + 1.3 KV, 1.8 GB byte diff | **100% valid** |
| **GLM-5.2** (MLA+DSA+MoE) | 0% valid — STORAGE | **100% valid** |
| **MiMo-V2.5-Pro** (SWA + fused-QKV) | 0% valid — STORAGE + SCHEDULE | **100% valid** |

weight clears STORAGE, shape clears SHAPE, schedule clears SCHEDULE, and axiom / SWA `kv_window` clears the last residual KV-semantics errors — which on V4-Pro and GLM-5.2 survive all the way to the full tier.

**The failure mode is precisely "confidently wrong."** At the bare and weight-only tiers the agent exits with `success=True` and **its own (weakened) validation green** — it believes it is done — yet the full oracle finds the spec broken. The gap between what the agent sees and what is true **is** that guard's contribution: without it the agent receives no signal that it is wrong, so it confidently declares success and hands over a bad spec. Guard necessity also scales with how much the agent has to derive — the 1T Hyper-Connections model is the most spectacularly wrong when bare (32.7 STORAGE violations, 1.8 GB), and needs the full set to turn a confident but wrong draft into a correct spec.

The oracle here is not self-certifying: `validate()` anchors on **real model artifacts** (checkpoint tensor names / bytes / dtypes, `config.json` values, forward source), not on any spec. A structural-equivalence cross-check against committed goldens (name-independent params / bytes / per-block profile / KV-semantics comparison) agrees with the oracle on clear cases, and where they diverge the oracle catches **more** — SCHEDULE's per-layer assignment coverage and NAMING's tensor mapping are exactly what a pure params/structure comparison cannot see. (On cost: the hardest model, V4-Pro, takes 62–87 agent turns, 13–25 M tokens, 44–63 minutes and $14–22 per from-zero derivation — 3–4× the other two.)

The conclusion is one sentence: **every guard is necessary** — deterministically each is the unique catcher for the fault it guards, and empirically, remove it and a from-zero agent will ship that class of error with false confidence. What the guards stop is not "getting the arithmetic wrong," but "getting it wrong and believing it is right."

---

### Visualizing the Spec: Gallery and Inspector

Once a spec is trustworthy, the most immediate use is visualization and interaction. **Gallery** (`GET /gallery/overview`) aggregates the whole snapshot library into a global comparison view — release timeline, total-vs-active parameter scatter, and each model's prefill/decode SOL curves at long context in its native dtype. All values reuse the same computation path as Inference, so architectural evolution across eras can be compared side by side in a single view. **Inspector** (`backend/rendering/builders.py` + `frontend/inspector.html`) expands a spec recursively into a layer-by-layer expandable DAG, each node carrying shape / params / FLOPs; hovering shows the symbolic formula and numeric value as two renderings from the same source, so every value's origin is traceable. Switching from single-model to comparison mode places two models side by side for block-by-block comparison of attention type, KV cache, and parameter distribution.

## II. Building the Inference Cost System

Given this trustworthy spec and the dual-track sympy evaluation threaded through all three tiers, "can this model be deployed on this GPU, and how" becomes an application layer on top of the spec — no modeling code needs to change, and every value derives from the same expression tree. The five sections below correspond to the five mechanism layers of the cost estimation system: how inference frameworks fuse several ops into one kernel (Fusion), how a deployment plan shards tensors across devices (parallel sharding), how the shapes each device actually executes are derived after sharding (the shape system), how SOL-based prefill/decode throughput and capacity are computed (SOL throughput prediction), and how measurement corrects the SOL upper bound to attainable throughput (measured-operator-library calibration).

### 2.1 Fusion

A spec describes the computation graph in the mathematical sense, but a real inference framework fuses a run of ops into a single kernel at runtime — the `q_proj` / `k_proj` / `v_proj` that are separate in the spec may be one fused QKV GEMM in vLLM. The Fusion Patterns page (`frontend/fusion.html`, backend `GET /fusion/registry` + `/fusion/fired`) models this layer explicitly as a registry: each fusion plan declares a foldable op topology, and each framework binding (vLLM / SGLang, by version) declares which plans that framework actually enables. Pick a model plus framework plus version and `/fusion/fired` runs one `apply_fusion` over that model's resolved spec, reporting which plans fired, which leaf nodes they anchored to, and which siblings were folded away. Because fusion matching is purely topological (independent of TP / EP / GPU), this analysis runs at batch=1 / seq=1 and is cheap. Its value is that the spec doesn't stop at "the theoretical operator graph" but can also match "how a specific version of a specific framework will actually run" — fusion plans and bindings are both derived from real vLLM / SGLang source, not inferred from blog posts.

### 2.2 Parallel Sharding

When a model is distributed across GPUs, per-device memory and communication depend on how tensors are sharded. The naive approach is uniform division by `cluster_size`, but when a model uses different TP degrees in different regions (attention sharded 8 ways, MoE experts 32) and layers CP and EP on top, global division can express neither which tensors sit in the same communication group nor which edges need communication inserted. HuggingArch therefore models parallelism as a placement algebra (`backend/arch_spec/sharding.py`), an abstraction borrowed from PyTorch [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html) (and its intellectual source GSPMD, [Xu et al. 2021](https://arxiv.org/abs/2105.04663)). DTensor's core idea: a logical tensor's distribution is described by its **placement** on each dimension of a device mesh, taking one of three states — `Shard` / `Replicate` / `Partial`. When an operator executes on distributed tensors, the output placement is derived from input placements by operator semantics, and wherever the two disagree a **redistribute is inserted automatically** (the required all-gather / all-reduce / reduce-scatter / all-to-all), so distributed tensor programming matches single-device programming and communication needn't be hand-written. HuggingArch adopts these placement semantics but performs no actual tensor computation, propagating only shape / bytes / FLOPs over the computation graph: each tensor carries a `TensorShardingState` recording its placement on every parallel world (`tp_attention` / `tp_moe` / `ep` / `cp` …) as `Replicated` / `Sharded(axis, world)` / `Partial(reduce_op, world)`, with multiple worlds composing orthogonally (TP-attention's head shard and CP-Ulysses acting simultaneously, i.e. two worlds each `Sharded` on the `N` axis).

The value of this abstraction is that communication is no longer special-cased per op but derived automatically from producer/consumer state differences: `Sharded → Replicated` derives an all-gather, `Partial → Replicated` derives an all-reduce, and multiple worlds decompose axis by axis, each producing its own collective. EP routing, CP's Ulysses / ring, and decode's KV sharding with LSE combine (aligned with vLLM DCP) are all derived from the same algebra — no communication anywhere is hard-coded for a particular parallelism combination. On the timing side, comm bytes take intra- or inter-link bandwidth by a region-aware mesh determination, while FLOPs are divided by a `placement_divisor` containing only model-parallel axes and excluding DP. The Inference page exposes TP / PP / EP / CP / DP as a set of controls, and each leaf's communication time is presented separately within `max(t_compute, t_memory, t_comm)`. For the full rules of "op-native sharding mode → shard-state propagation → communication injection," the state algebra, and per-rank costs, see [`docs/parallelism.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/parallelism.md).

### 2.3 SOL-Based Prefill/Decode Throughput Prediction

The Intro gave a single operator's SOL time as $T_{\mathrm{SOL}} = \max(\text{FLOPs}/\text{peak compute},\ \text{bytes}/\text{bandwidth})$; aggregating bottom-up along the computation graph gives end-to-end SOL time for the prefill and decode phases, and from there TTFT, TPOT, throughput, and max-batch capacity. This layer closes the loop without any kernel measurement, and its value is providing a unified, reproducible, implementation-independent analytical baseline:

- **Performance upper bound.** SOL is the theoretical minimum time set by the hardware roofline; no real implementation can beat it. It lets you quickly judge whether a model can be deployed on a given GPU under a given plan, and at what order of magnitude.
- **Deployment plan comparison.** The relative merits of one model across different TP / EP / parallel degrees / GPUs are directly comparable under a single SOL convention, with no need to measure each separately.
- **Cross-model comparison.** The relative cost of different architectures (MLA vs GQA, DSA vs KDA, the various hybrids) at long context and high concurrency — SOL provides the one control baseline undisturbed by kernel implementation differences. That is exactly the core demand behind the "arithmetic" described in the Intro.

Here is how each component of the SOL loop is computed.

**Precise memory modeling** runs through the same sympy engine, bringing more variables into the symbolic system:

- **Quantization.** Weight bytes are determined by `weight_dtype × params + quant_overhead` (see the Quantization section above).
- **Parallelism.** `estimate_capacity()` in `backend/inference/capacity.py` takes a `ParallelConfig` — not one generic TP, but parallel axes broken out finely: TP along three paths (`tp_attention` / `tp_dense_ffn` / `tp_moe_expert`, since attention and MoE experts commonly use different TP degrees), EP (`ep_moe`), PP, CP, plus two kinds of DP (`dp_attention` replicating attention, `dp_replicas` replicating whole machines), plus an SP switch and a CP scheme (ulysses / ring). Each device's weight / KV / activation / communication shards along its own parallel axis — how exactly, see the placement algebra above.
- **Attention-type-aware KV cache.** `kv_cache_per_token_per_layer()` in `backend/inference/symbolic.py` gives an independent formula per attention type: MHA / GQA / SWA are `2·Nk·D·dtype_bytes`, MLA is `(R + P)·dtype_bytes` (compressed latent plus RoPE-only key, with no fanout to each head). Linear attention / SSM (GLA, GatedDeltaNet, Mamba2) recurrent states aren't "accumulated per token" but are modeled as a class of **constant-storage KV** — `backend/inference/kv_topology.py` gives them a `recurrent_state` node whose per-token element count is constant and doesn't grow with context, and `kv_cache.py` lowers it to fixed bytes via `constant_floor`. A hybrid model's KV cache is naturally a weighted sum over layer types — the per-token-growing branch and the constant recurrent state are each computed on their own terms, never conflated.
- **GPU SKU.** `backend/inference/gpu_specs.py` is a CSV-driven GPU parameter library (HBM capacity, HBM bandwidth, peak BF16/FP8 TFLOPs), so any SKU plugs straight into the capacity model.

**Theoretical time and MFU.** `backend/inference/phases.py` decomposes a single inference step into phases — weight load, compute, KV store / load, communication. Each phase's theoretical time is bytes / bandwidth or FLOPs / peak_TFLOPs, and the phases take a maximum (the roofline view). MFU is computed directly at the `/inference/estimate` endpoint as `model_FLOPs_per_token / (peak_TFLOPs × measured_latency)`; combined with the phase decomposition, this locates whether a workload is bandwidth-bound or compute-bound.

All of the above rests purely on SOL, with no kernel measurement. Real kernels reach only a fraction of that bound; the next two sections discuss correcting it toward attainable throughput. But that doesn't change SOL's standing as the cross-model, cross-plan comparison baseline — measurement is only a correction to that bound.

### 2.4 The Shape System: Deriving the Real Shapes on Each Device

Correcting the SOL bound toward real implementations starts with knowing the shape each op actually executes on each device — which needs a reliable shape system. It reuses the same mechanism as the shape inference in Part I: sharding is modeled as an **idempotent transformation** on the computation graph, and per-rank shapes are obtained as a **downstream derivation** of the transformed graph, not as an independent recomputation from the full shapes.

Concretely, a valid deployment plan (`ParallelConfig`) only changes the atomic model dims fed into the engine — attention heads `N ÷ tp_attention`, dense intermediate `I ÷ tp_dense_ffn`, expert intermediate `Im ÷ tp_moe_expert`; EP reduces the expert count to `E / ep` without changing any dim — and then `compute_model` from Part I is re-run with that set of dims. Since each op's shape is already derived from input dims by its `shape_kind` rule (`out = N·(D + P)` recomputes automatically with `N`), the per-rank shapes of the entire graph after sharding derive consistently, with no need to write separate derivation logic for arbitrary TP / EP / CP combinations. That's the core value of "one shape substrate plus an idempotent graph transformation": shape correctness is guaranteed by the derivation rules, not by special handling of each parallelism combination.

The shapes obtained this way are the real execution scales on each device — DeepSeek-V2-Lite's grouped_gemm `K` is 1408 under tp_ep8, for example — and can feed the measurement of the next section directly. This complements the shape inference of Part I: there it performs geometric consistency validation while building the spec; here it forward-derives a deployment plan into the real kernel shapes on each device. For details of the placement algebra, see [`docs/parallelism.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/parallelism.md).

A reliable shape system brings one more key benefit. Because it can forward-derive **the real kernel shape on each device under a distributed deployment**, profiling an operator at that shape on **a single GPU** yields kernel performance that would normally require standing up a genuinely large distributed cluster to measure. In other words, the lightest possible single-GPU measurement covers operator costs for distributed deployments — which is the precondition for the next section.

### 2.5 Measured-Operator-Library Calibration

With the per-rank shapes forward-derived in 2.4, the SOL bound of 2.3 can be corrected toward attainable throughput, filling the gap between theoretical bound and real implementation: measured `t_measured` sits **alongside** theoretical `t_sol` and does not overwrite it.

Taking real per-rank shapes as input, measurement runs through **adapters onto open-source operator libraries**: each library (FlashAttention 2/3, FlashInfer, FlashMLA, DeepGEMM, FLA, Mamba-SSM, vLLM, SGLang, …) registers a provider that adapts canonical inputs into its own layout and executes the real kernel. The `kernel_bench` driver uses a torch-eager implementation as reference and cross-checks numerics within a dtype-scaled tolerance; a kernel whose results disagree is not recorded. The fastest correct kernel across libraries becomes that op's measured envelope, and the `op_calibration` overlay backfills it into the `t_measured` track. The corpus has two layers: the **shared corpus** is the deployed public baseline (materialized to disk; core only reads disk and never touches the database), while **private uploads** are a signed-in user's private rows, overlaid at estimation time to calibrate that user's own estimates without being made public. For the "deployment plan → real per-rank shape → adapter measurement" mechanism, the library/kernel support matrix, and details of the two corpus layers, see [`docs/measured_calibration.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/measured_calibration.md); for the placement algebra that shape derivation depends on, see [`docs/parallelism.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/parallelism.md).

---

## Roadmap

HuggingArch is still an evolving scaffold. The directions ahead:

- **More precise inference modeling.** Bring mechanisms with real throughput impact — speculative decoding, EP balance (expert load balancing) — into the model, and replace purely theoretical comm time with measured communication, so estimates track real deployments more closely.
- **A thicker measured corpus.** Extend the measured corpus across GPU SKUs, operator libraries, and models. Every additional measurement tightens the SOL bound a little further toward attainable throughput — the layer best suited to community contribution.
- **From spec analysis to model design.** Extend from "understanding published models" to cost analysis of custom models, supporting cost exploration and comparison during architecture design.
- **Training cost modeling.** Extend cost modeling from inference to training.
- **Evolving the spec abstraction.** As models like DeepSeek-V4-Pro push structural complexity higher, promote side-branches, dynamic axes, and dim-family types — currently handled as special cases — into first-class abstractions (see [`docs/future_work.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/future_work.md)).

## How to Contribute

This is an open project of side-project character. Stars are welcome, and so is joining as a developer. Ways to participate:

- **Contribute a model spec.** Run the agentic generation pipeline for an unsupported model; once the spec passes the full validation suite (`bytes_diff == 0`) it can be nominated, and after review `promote` merges it into the component library. Fill in your GitHub handle in your profile and contributed specs will be credited co-authored-by in the PR.
- **Contribute measured corpus data.** Run `kernel_bench` on your own GPU and upload the measured CSV. That data calibrates your own estimates immediately (no need to publish it) and can also be nominated into the shared corpus to become a public baseline after review.
- **Contribute components and fusion bindings.** New topological components, or new framework fusion plans derived from vLLM / SGLang source, all go through the same draft → review → promote flow.
- **Report issues, join the discussion.** Any number that doesn't reconcile, or any new architecture not yet supported — issues and PRs are welcome.

For the getting-started path and code structure, see the repository's `docs/` (`README.md` is the layered map); for mechanism details, see the individual deep dives.

## Postscript: I Had an Agent Do the Arithmetic for DeepSeek V4-Pro. It Crashed — Which Is Exactly the Point of This Project

After DeepSeek released V4-Pro I ran an experiment. This time I didn't do it myself: I had an agent write a deployment cost analysis of V3 / V3.2 / V4-Pro on 8×B200 — prefill, decode, max-batch, context swept from 4K to 1M. The agent did a creditable job. It laid out architectural differences in fine detail (CSA / HCA interleaving, per-layer SWA branches, shared-KV MQA, FP4 MoE, mHC, …), and the qualitative conclusions held up (V4 cuts cumulative KV to ~10% of V3's; single-node long-context high-concurrency is its design target). Roofline formulas throughout, speedup multiples, sweep table after sweep table — it looked thoroughly professional.

Then I ran the document through HuggingArch to reconcile it. The result is telling: **qualitatively all correct, quantitatively a crash.**

The most representative case is prefill. The document has V3 @1M at 73.48 µs/token; recomputing from first principles in HuggingArch (128 query heads × qk192/v128 × 61 layers × causal ½, aggregate 8×B200 BF16) gives ~150 — **the agent dropped a factor of 2**. Looking at that one cell in isolation, the error is undetectable; 73 and 147 both "look reasonable," and the agent re-reading its own work can't tell which is wrong. But it's the baseline, so one error poisons an entire column of relative values: because V3 was underestimated by half, V4's prefill speedup over V3 was written as 4.7×, when **the true value is 9.4×**. The agent undersold this model's headline selling point by a factor of two.

The subtler case is short context. The document assumes prefill is compute-bound and computes purely from FLOPs/peak, giving ~1 µs/token at 4K. But single-stream (B=1) short-context prefill is actually **weight-bound** — a pile of MoE expert weights has to come off HBM — and that floor was ignored for the whole column. This isn't a transcription error; it's **a modeling assumption quietly failing in a particular regime**. That kind of error is the hardest to catch by "reading it again."

This is a live specimen of the "claiming an N× speedup is easy, but it won't survive scrutiny" problem from the opening — except this time the one making claims was an agent. And that's precisely what makes it dangerous: let an agent do cost analysis freely and its output **looks more professional than a human's** — formulas, tables, multiples, all present, confidence maxed out — while the dropped ×2 and the failed assumption hiding inside are things it is entirely unable to detect. Nor should you expect "have the agent review it again" to catch them: hallucination reviewing hallucination only makes both look more convincing.

And this isn't a one-off impression. After writing that piece we turned it into a controlled experiment: have 8 different coding agents write specs for the same batch of models, then reinstall the ground-truth checks one notch at a time from all-off, and watch how output changes at each step. The result is clean — **running bare, with no guards at all, an agent judges an actually-wrong spec to be "passing" 94% of the time**. Its local validate is green, because the ruler that could catch the error has been taken away. With all guards reinstalled, that "confidently wrong" rate falls to 6%. Failures land almost entirely on checks pinned to physical fact (weight bytes, per-token KV, per-layer scheduling), and cluster heavily on the hardest model, V4-Pro. Which says it again: whether an agent makes mistakes isn't the point; whether there's ground truth to catch them is.

That is the starting point of HuggingArch's whole design: **not to stop agents from doing the arithmetic, but to stop them doing it in a vacuum with no ground truth.** For the same V4-Pro, what an agent writes in HuggingArch is not a freely composed Markdown document but a constrained spec — every op's FLOPs and parameter formulas are built-in and trustworthy, and every number is pinned to `config.json`, safetensors, and forward source. That missing ×2 either can't be written at all, or immediately fails to reconcile against first principles and real tensor byte counts and gets bounced back. The agent doesn't need to "happen to be right"; it only needs to hit ground-truth feedback and fix. That is a completely different reliability tier from letting it write a cost analysis freehand.

> Incidentally, reconciliation runs both ways: this round also flushed out a bug in HuggingArch itself. For V4's topology — one layer caching both a sliding-window KV and a time-compressed KV, with different cardinalities — the tool had previously capped the compressed branch at the sliding window W=128 as well, so max-batch didn't budge from 4K all the way to 1M. Plainly unphysical. Finding it came down to the same discipline: every number must reconcile against the nodes the spec actually declares in `kv_cached` and against the officially reported "~10% KV"; if it doesn't reconcile, something is wrong. The fix respected the project's rules too — no `if` for V4, but a general abstraction in which each cached node carries its own storage/read cardinality (which incidentally corrected the convention for V3.2's full-length indexer scan). **The conclusion: agents, human brains, and the tool itself all make mistakes while modeling; the only difference is whether there's ground truth to catch them.**

So what HuggingArch is trying to solve was never just the manual labor of "rebuilding a model by hand for every new release." It's the trust problem: can the arithmetic an agent produces be believed at all? In the era of agentic coding, getting a model to generate an analysis is easy; getting it to generate a **trustworthy** analysis is hard. A system that lays the formulas open, states the assumptions, and pins every number to a factual source — that's what makes it safe to publish a sentence like "V4 is 9.4× faster than V3," and what gives you the standing to stop an agent, red-faced, the next time it makes an earnest and confident claim.
