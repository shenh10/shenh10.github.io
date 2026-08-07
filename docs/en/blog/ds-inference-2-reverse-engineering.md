---
title: "DeepSeek V3/R1 Inference Efficiency (2): Reverse-Engineering the Production Deployment"
description: Layer-by-layer profiling of DeepSeek's published EP144 setup — prefill and decode FLOPs, per-operator timings, MFU, and the overlap schedule that follows from them.
---

# DeepSeek V3/R1 Inference Efficiency (2): Reverse-Engineering the Production Deployment

> Originally published in Chinese on [Zhihu](https://zhuanlan.zhihu.com/p/29841050824).

::: tip This is part 2 of 3
Part 1 estimated a throughput ceiling from the V3 paper alone. This part rebuilds the estimate against everything DeepSeek has since published. Part 3 generalizes the result to other configurations.
:::

::: info Updates
**Mar 21** — DeepSeek published `decoding.json`. Reading the MoE layers, a single MoE layer across two microbatches runs about 714 × 2 = 1.4 ms, with end-to-end around 87–96 ms; efficiency is slightly above MTP=1. The overlap structure and the layer-by-layer profiling method are the same as below, so the numbers here are left as they are.

**Mar 16** — Updated the decode BMM profiling figures.
:::

## 1. Introduction

The [qualitative estimate](/en/blog/ds-inference-1-throughput-ceiling) in part 1 drew more interest than expected, along with a number of questions. As an exercise in bounding the problem it did its job — it ruled out some wildly optimistic ceilings — but as an estimate of DeepSeek R1's performance, now that the timeline data is public, it was too coarse. Two methodological problems:

**a) Whether the bound is attainable**

> Taking compute and communication together, without MTP, R1 on EP256 H800 in FP8 has a per-GPU ceiling of 3300 tokens/s (BF16 combine) to 5000 tokens/s (FP8 combine); H20 sits around 1600 tokens/s.

Writing per-GPU throughput as $T$, part 1 used the naive estimate $T_\text{overall} = \min(T_\text{net}, T_\text{compute})$. The minimum of the two is certainly an upper bound, but not necessarily an *attainable* one. $T_\text{net}$ is a relatively tight and realistic bound, but $T_\text{compute}$ should not simply be peak compute converted to throughput, $T_\text{compute}^\text{peak}$ — its real value depends on MFU and on the overlap design:

- On H800, if communication cannot be fully hidden behind compute (that is, $T_\text{compute} = T_\text{compute}^\text{peak} \times \text{MFU} < T_\text{net}$), then $T_\text{net}$ is unreachable and the attainable bound is set by $T_\text{compute}$.
- On H20, even though $T_\text{compute}^\text{peak} < T_\text{net}$, MFU losses still apply: attainable throughput is $T_\text{compute}^\text{peak} \times \text{MFU}$, where MFU is governed mostly by the efficiency of the non-communication operators.

**b) Computing the expert saturation point**

Part 1 used $b_\text{mla} \cdot 9 \cdot d / n_e \ge b_\text{ep,sat}$, taking the saturation point from a dense-GEMM batch figure extrapolated from earlier BF16 GEMM experience. Two problems with that:

- No full FP8 GEMM curve had been measured, so the saturation point could be substantially off.
- It ignores what grouped GEMM does for TFLOPS. Assuming $b_\text{ep,sat} = 4096$ presumes a single expert GEMM can saturate the GPU, which overlooks the SM-utilization gain grouped GEMM provides when individual GEMMs are small. That inflates the estimate of the device count $d$, and forecloses the question most practitioners actually care about — whether a smaller EP group achieves the same effect.
- With grouped GEMM, $b_\text{ep,sat}$ should be a function of `group_number` and `m_per_group`, which are themselves tied to the $d/n_e$ on the left-hand side.

This post therefore works from everything DeepSeek has published — [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepEP](https://github.com/deepseek-ai/DeepEP), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [profile-data](https://github.com/deepseek-ai/profile-data), and the [V3/R1 inference system overview](https://zhuanlan.zhihu.com/p/27181462601) — to reverse-engineer the EP144 deployment reasonably completely.

> This post no longer distinguishes V3 from R1, instead using the average distribution from the inference-system overview. Aligning to the official figures means correcting two coarse assumptions from part 1:
>
> 1. The shared expert is replicated on every device, rather than distributed redundantly across separate nodes as under EP320.
> 2. Expert redundancy is accounted for: both prefill and decode use 256 routed experts plus 32 redundant experts.

The key figures DeepSeek published:

- **Prefill** — routed experts EP32, MLA and shared expert DP32; one deployment unit is 4 nodes, with 32 redundant routed experts, 9 routed experts and 1 shared expert per GPU.
- **Decode** — routed experts EP144, MLA and shared expert DP144; one deployment unit is 18 nodes, with 32 redundant routed experts, 2 routed experts and 1 shared expert per GPU.
- 608B input tokens total, of which 342B (56.3%) hit the KV cache on disk.
- 168B output tokens total. Average output rate 20–22 tps; average KV cache length per output token, 4989.
- Average per-H800 throughput: prefill about 73.7k tokens/s input (including cache hits); decode about 14.8k tokens/s output.

## 2.1 Average prefill/decode lengths

Following the method raised in the comments on part 1:

> Let $P$ be the average input length and $D$ the average output length. Then the average KV cache length per output token is roughly $P + D/2 = 4989$; combined with $P/D = 608\text{B}/168\text{B}$, this gives $P \approx 4383$ and $D \approx 1210$.

So $\bar{P} = 4383$, $\bar{D} = 1210$, and the average attention KV cache length is $s' = 4989 \approx 5000$.

## 2.2 Average prefill/decode instance counts

For balanced prefill/decode consumption, consider the ratio of 4-node prefill instances to 18-node decode instances. With $x$ prefill groups and $y$ decode groups, on average $4x + 18y = 226.75$.

- Working back from total input throughput, concurrency is about $(608 - 342) \times 10^9 / 24 / 3600 / (73.7 \times 43.7\% \times 1000) = 96$ nodes.
- Working back from total output throughput, $168\text{B}/24/3600/14.8/1000 = 131$ nodes.

That gives $x \approx 24$, $y \approx 7$ — roughly 24 prefill instances and 7 decode instances to support DeepSeek's online load in a balanced way.

## 2.3 Prefill analysis

From the [prefill timeline](https://github.com/deepseek-ai/profile-data/blob/main/prefill.json) settings, prefill uses a 4k prompt with 16k tokens per GPU across 2 microbatches. So a single microbatch has $b = 2$, $s = 4096$. Because prefill's overlap balances the two microbatches, only one is considered here.

![](/blog/ds-inference-2-reverse-engineering/fig01.jpg)

*DP-32, EP-32 prefill two-microbatch overlap timeline*

### 2.3.1 Per-layer profiling, single prefill microbatch

**Compute**

A rough sketch of prefill MLA — the notation abuses part 1's slightly, but the correspondence should be clear:

$c' = 1536$, $r = 64$, $a/n_\text{head} = 128$, $d = 128$, $h = 7168$

![](/blog/ds-inference-2-reverse-engineering/fig02.jpg)

- **QKV projection**
  - $X @ \text{Concat}(W_Q', W_p, W_{KR})$: $[bs, 7168] @ [7168, 2112] = 2 \times 2 \times 4096 \times 7168 \times 2112 / 10^9 = 248$ GFLOPs
  - $Q' @ \text{Concat}(W_{UQ}, W_{QR})$: $[bs, 1536] @ [1536, 24576] = 2 \times 2 \times 4096 \times 1536 \times 24576 / 10^9 = 618.5$ GFLOPs
  - $KV @ \text{Concat}(W_{UK}, W_{UV})$: $[bs, 512] @ [512, 128 \times 128 \times 2] = 2 \times 2 \times 4096 \times 512 \times 32768 / 10^9 = 274$ GFLOPs
- **MLA (MHA) attention**
  - $Q^T @ K$: $2 b s\, s\, n_\text{head} (d + r) / 2$ (causal) $= 2 \times 2 \times 4096 \times 4096 \times 128 \times 196 / 10^9 / 2 = 841.5$ GFLOPs
  - $P @ V$: $2 b s\, s\, n_\text{head} d / 2$ (causal) $= 2 \times 2 \times 4096 \times 4096 \times 128 \times 128 / 10^9 / 2 = 550$ GFLOPs
- **O projection**
  - $[bs, 128 \times 128] @ [128 \times 128, 7168] = 2 \times 2 \times 4096 \times 16384 \times 7168 / 10^9 = 1924$ GFLOPs
- **Routed expert GEMM** — `group_number` = 9, `m_per_group` = $bs \times 8 \times 32 / (256 + 32) = 7281$
  - Up & gate: $9 \times [7281, 7168] @ [7168, 4096] = 2 \times 9 \times 7281 \times 7168 \times 4096 / 10^9 = 3848$ GFLOPs
  - Down: $9 \times [7281, 2048] @ [2048, 7168] = 2 \times 9 \times 7281 \times 7168 \times 2048 / 10^9 = 1924$ GFLOPs
- **Shared expert GEMM**
  - Up & gate: $[2 \times 4096, 7168] @ [7168, 4096] = 2 \times 8192 \times 7168 \times 4096 / 10^9 = 481$ GFLOPs
  - Down: $[2 \times 4096, 2048] @ [2048, 7168] = 2 \times 8192 \times 7168 \times 2048 / 10^9 = 241$ GFLOPs

**Communication**

With only 4 machines, the network estimate follows the intra-device deduplication scheme discussed earlier: across 4 nodes, each token sends at most 3 copies outward.

- **Dispatch (per layer)**: $2 \times 4096 \times 7168 \times 3 / 1024 / 1024 = 168$ MB
- **Combine (per layer)**: $2 \times 2 \times 4096 \times 7168 \times 3 / 1024 / 1024 = 336$ MB

With FLOPs and traffic in hand, here is the timing analysis of `prefill.json`:

![](/blog/ds-inference-2-reverse-engineering/fig03.jpg)

*Actual prefill timeline overlap*

**Per-layer, per-microbatch duration and TFLOPS**

To enable overlap, 108 SM cores are used for compute and 24 for communication. GEMMs give up 10–20% of MFU relative to running exclusively.

| Compute | Shape | GFLOPs | Duration (µs) | TFLOPS |
| --- | --- | --- | --- | --- |
| $X @ \text{Concat}(W_Q', W_p, W_{KR})$ | [7168, 2112] | 248 | 268 | 925.5 |
| $Q' @ \text{Concat}(W_{UQ}, W_{QR})$ | [1536, 24576] | 618.5 | 922 | 670.8 |
| $KV @ \text{Concat}(W_{UK}, W_{UV})$ | [512, 32768] | 274 | 533 | 515.7 |
| MHA attention | | 1392 | 2683 | 519 |
| O projection | [16384, 7168] | 1924 | 1652 | 1164.7 |
| Shared up & gate | [7168, 4096] | 481.04 | 439 | 1095 |
| Shared down | [2048, 7168] | 240.5 | 306 | 786 |
| Routed up & gate | [7168, 4096] | 3848.3 | 3534 | 1089 |
| Routed down | [2048, 7168] | 1924 | 2381 | 808 |

| Communication | Traffic (MB) | Duration (µs) | Bandwidth (GB/s) |
| --- | --- | --- | --- |
| Dispatch notify | | 743 | |
| Dispatch all-to-all | 168 | 4326 | 38 |
| Cache notify | | 788 | |
| Combine all-to-all | 336 | 8845 | 37 |

| Other (3004 µs in total) | Duration (µs) |
| --- | --- |
| Attention: add & LayerNorm & RoPE | 549 |
| Attention: BF16→FP8 for O projection | 232 |
| Gate: router gate & prepare shared GEMM | 529 |
| Expert: prepare routed GEMM | 728 |
| Expert: SwiGLU | 314 |
| Expert: combine reduce | 594 |

Principal compute time: 12.7 ms.

| | GFLOPs | Duration (µs) | Model TFLOPS | MFU |
| --- | --- | --- | --- | --- |
| GEMMs + attention (SM 108) | 10950 | 12718 | 861 | 44% |
| GEMMs + attention + memory ops (SM 108) | ~10950 | 15722 | 696 | 35% |

### 2.3.2 Per-GPU prefill throughput

**From the timeline.** A full prefill forward pass takes about 2118 ms, i.e. $4 \times 4096 / 2.118 = 7735$ tokens/s.

Against the theoretical figures:

- At 38 GB/s of communication bandwidth, $T_\text{net} = 2 \times 4096 \times 38 / ((168 + 336) \times 58 / 1024) = 10900$ tokens/s.
- At peak compute, $T_\text{compute}^\text{peak} = 2 \times 4096 \times 1978 \times 1000 / 12341 / 61 = 21524$ tokens/s, of which the attained figure is $T_\text{compute} = T_\text{compute}^\text{peak} \times \text{MFU} = 21524 \times 0.35 = 7533$ tokens/s. So when MFU is low, $T_\text{compute}$ becomes the tight bound on throughput.

**From production data.** Per-GPU prefill throughput is roughly $73.7\text{k} \times (1 - 56.3\%) / 8 = 4025$ tokens/s.

**Conclusion.** DeepSeek's peak prefill throughput *under balanced load* reaches 7735 tokens/s. The 4k tokens/s implied by the averaged production figures reflects a full day of peaks and troughs — periods when the system is not saturated, or when imbalance prevents full overlap.

## 2.4 Decode analysis

DeepSeek has not published an EP144 decoding timeline, so the decode profiling here comes from measurements of DeepGEMM and FlashMLA plus a small amount of estimation.

Take per-GPU $b_\text{mla} \approx 128$, as in [profile-data](https://github.com/deepseek-ai/profile-data), so each microbatch has $b_\text{mla} = 64$.

Under DP144-EP144, each routed expert receives on average $64 \times 144 \times 8 / (256 + 32) = 256$ tokens, i.e. `m_per_group` = 256, with `group_number` = $(256 + 32)/144 = 2$ per GPU.

### 2.4.1 Per-layer profiling, single decode microbatch

Decode uses absorbed MLA, which differs slightly from prefill. **(The absorption diagram has been updated: per the [SGLang implementation](https://github.com/sgl-project/sglang/blob/e1a5e7e47ddc35e55f87a0e66e4306bff62cdef6/python/sglang/srt/models/deepseek_v2.py#L684), the purple section is a BMM rather than an ordinary post-absorption linear. The BMM is currently modelled with `torch.bmm` in BF16, scaled by a factor of 1.7 to approximate FP8 BMM performance; the profiling figures below have been updated accordingly.)**

![](/blog/ds-inference-2-reverse-engineering/fig04.jpg)

**Compute**

- **QKV projection**
  - $X @ \text{Concat}(W_Q', W_p, W_{KR})$: $[bs, 7168] @ [7168, 2112] = 2 \times 64 \times 7168 \times 2112 / 10^9 = 1.94$ GFLOPs
  - $Q'' = Q' @ \text{Concat}(W_{Qf}, W_{QR})$: $[bs, 1536] @ [1536, 24576] = 2 \times 64 \times 1536 \times 24576 / 10^9 = 4.83$ GFLOPs
  - $Q''' = \text{bmm}(Q'', W_{UK}^T)$: $\text{bmm}([128, bs, 128], [128, 128, 512]) = 2 \times 128 \times 64 \times 1 \times 128 \times 512 / 10^9 = 1.07$ GFLOPs
- **MLA/MQA attention**
  - $Q^T @ K$: $2 b s\, s' n_\text{head} (c + r) = 2 \times 64 \times 5000 \times 128 \times 576 / 10^9 = 47.2$ GFLOPs
  - $O = P @ V$: $2 b s\, s' n_\text{head} c = 2 \times 64 \times 5000 \times 128 \times 512 / 10^9 = 41.9$ GFLOPs
- **O projection**
  - $O' = \text{bmm}(PV, W_{UV})$: $[n_\text{head}, bs, c] @ [n_\text{head}, c, d] = 2 \times 128 \times 64 \times 1 \times 512 \times 128 / 10^9 = 1.1$ GFLOPs
  - $O'$ projection: $[bs, 128 \times 128] @ [128 \times 128, 7168] = 2 \times 64 \times 16384 \times 7168 / 10^9 = 15$ GFLOPs
- **Routed expert GEMM** — `group_number` = 2, $m_\text{per\_expert} = bs \times 8 \times d / (256 + 32) = 256$
  - Up & gate: $2 \times [256, 7168] @ [7168, 4096] = 2 \times 2 \times 256 \times 7168 \times 4096 / 10^9 = 30$ GFLOPs
  - Down: $2 \times [256, 2048] @ [2048, 7168] = 2 \times 2 \times 256 \times 7168 \times 2048 / 10^9 = 15$ GFLOPs
- **Shared expert GEMM**
  - Up & gate: $[64, 7168] @ [7168, 4096] = 2 \times 64 \times 7168 \times 4096 / 10^9 = 3.76$ GFLOPs
  - Down: $[64, 2048] @ [2048, 7168] = 2 \times 64 \times 7168 \times 2048 / 10^9 = 1.88$ GFLOPs

**Communication**

Decode node counts will not be small, so assume 8 or more machines. Estimating the network ceiling under the least favourable pattern — sending to 8 other nodes, so at most 8 copies per token:

- **Dispatch (per layer)**: $64 \times 8 \times 7168 / 1024 / 1024 = 3.5$ MB
- **Combine (per layer)**: $2 \times 64 \times 7168 \times 8 / 1024 / 1024 = 7$ MB

Memory-bound operator durations are scaled from the prefill figures in 2.3 in proportion to token count, which is reasonable under a bandwidth bound. For router gate and prepare-shared-GEMM, for instance: $529 / 9 / 7281 \times 2 \times 256 = 4.13$.

Context length uses DeepSeek's actual $s' = P + D/2 \approx 5000$.

| Compute | Shape | GFLOPs | Duration (µs) | TFLOPS |
| --- | --- | --- | --- | --- |
| $X @ \text{Concat}(W_Q', W_p, W_{KR})$ | [7168, 2112] | 1.94 | 10 | 190 |
| $Q' @ \text{Concat}(W_{UQ}, W_{QR})$ | [1536, 24576] | 4.83 | 17 | 280 |
| $Q''' = \text{bmm}(Q'', W_{UK}^T)$ | bmm([128, bs, 128], [128, 512, 128]) | 1.07 | 10 | 112 |
| MLA/MQA attention | | 89.1 | 196 | 462 |
| $O' = \text{bmm}(PV, W_{UV})$ | bmm([128, bs, 512], [128, 512, 128]) | 1.1 | 8 | 132 |
| O projection | [16384, 7168] | 15 | 46 | 326 |
| Shared up & gate | [7168, 4096] | 3.76 | 14 | 270 |
| Shared down | [2048, 7168] | 1.88 | 7 | 258 |
| Routed up & gate | [7168, 4096] | 30 | 33 | 898 |
| Routed down | [2048, 7168] | 15 | 20 | 753 |

| Communication | Traffic (MB) | Duration (µs) | Bandwidth (GB/s) |
| --- | --- | --- | --- |
| Dispatch all-to-all | 3.5 | 88 | 39 |
| Combine all-to-all | 7 | 175 | 39 |

| Other (23 µs in total, estimated) | Duration (µs) |
| --- | --- |
| Attention: add & LayerNorm & RoPE | 4.29 |
| Attention: BF16→FP8 for O projection | 1.81 |
| Gate: router gate & prepare shared GEMM | 4.13 |
| Expert: prepare routed GEMM | 5.69 |
| Expert: routed SwiGLU | 2.45 |
| Expert: combine reduce | 4.64 |

Principal compute time: 350 µs.

| | GFLOPs | Duration (µs) | Model TFLOPS | MFU |
| --- | --- | --- | --- | --- |
| GEMMs + attention (SM 132) | 164 | 361 | 454 | 23% |
| GEMMs + attention + memory ops (SM 132) | ~164 | 384 | 427 | 21.5% |

### 2.4.2 Per-GPU decode throughput

**From the timeline.** Without a complete timeline, decompose the decode duration:

![](/blog/ds-inference-2-reverse-engineering/fig05.jpg)

*DP-144, EP-144 decode two-microbatch overlap timeline*

- **Shared + Attn0** — shared + MLA QKV GEMM + after-combine-reduce + before-core-attention: $10 + 17 + 10 + 14 + 7 + 4.1 + 4.6 + 4.29 = 71$ µs < 88 µs. This does not fully hide the communication, so 88 µs.
- **MLP** — prepare + routed GEMM + SwiGLU: $5.69 + 33 + 20 + 2.45 = 61$ µs.
- **Attn1** — core attention + O projection + routing gate: $196 + 8 + 46 + 1.81 + 4.13 = 255$ µs > 174 µs, which covers the combine, so about 255 µs.

A single layer's forward is therefore about $88 + 71 + 255 = 414$ µs, with the first three layers at roughly 384 µs since they carry no communication. One forward iteration, including both microbatches, is

$$(384 \times 3 + 414 \times 58) \times 2 = 50\ \text{ms}$$

**That puts TPOT at about 50 ms — roughly 20 tokens/s per user** — with per-GPU throughput of $64 \times 2 \times 1000 / 50 = 2560$ tokens/s.

**From production data.** Per-GPU decode throughput is about $14.8 \times 1000 / 8 = 1850$ tokens/s, implying an actual per-GPU concurrency of $b_\text{mla} = 1850/21 \approx 88$.

**Conclusion.** Peak decode throughput *under balanced load* reaches 2560 tokens/s; the 1850 tokens/s from the averaged production data reflects the daily peaks and troughs during which the system is not saturated.

## 2.5 Choosing the overlap scheme

The official pipeline diagrams show prefill and decode using different overlap strategies: prefill allocates 24 SM cores to communication, while decode consumes none.

Setting aside the IBGDA implementation differences introduced for latency, consider how to overlap communication with compute from first principles.

> I take IBGDA to be primarily about achieving lower latency at small transfer sizes, which shows up in DeepEP as higher effective bandwidth. It does not change the overlap analysis.

For prefill, compute-intensive GEMMs dominate, so memory-bound operator costs can largely be ignored. Draw the single-microbatch dependency graph — the blue section — and the remaining problem is filling the bubbles left by dispatch (dispatch notify + all-to-all ≈ 5 ms) and combine (cache notify + all-to-all ≈ 9.6 ms). QKV + core attention + O projection takes about 6 ms, and the MLP about 5.9 ms — both close to the dispatch duration, with dependencies that stagger conveniently. The two-microbatch overlap below follows naturally; the shared-expert computation is moved to overlap with combine, to fill it as fully as possible.

![](/blog/ds-inference-2-reverse-engineering/fig06.jpg)

*Prefill overlap*

Decode GEMMs are all relatively small, so memory-bound operator costs cannot be ignored here. Writing:

- **QKV** (attention add & LayerNorm & RoPE + QKV GEMM) = $10 + 17 + 10 + 4.29 = 41.3$ µs
- **ATTN + O + Gate** (core MLA attention + O projection + routing gate) = $196 + 8 + 46 = 250$ µs
- **Shared** (prepare + shared expert GEMMs) = $4.13 + 14 + 7 = 25.1$ µs — the gate is small enough that it is folded in here
- **MLP** (prepare + routed GEMMs + SwiGLU) = $5.69 + 33 + 20 + 2.45 = 61$ µs

Again, start from the single-microbatch dependency graph, in blue below.

![](/blog/ds-inference-2-reverse-engineering/fig07.jpg)

Core MLA attention dominates decode, so it can no longer be overlapped with dispatch; it should be overlapped with combine instead. That gives the scheme below, which matches the one DeepSeek published.

![](/blog/ds-inference-2-reverse-engineering/fig08.jpg)

## 3. Conclusion

That is a reasonably complete decomposition of what DeepSeek has published. The generalization to other configurations turned out to be too much material for one post; it is the subject of [part 3](/en/blog/ds-inference-3-decode-generalization).

## References

- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [profile-data](https://github.com/deepseek-ai/profile-data)
- [DeepSeek V3/R1 inference system overview](https://zhuanlan.zhihu.com/p/27181462601)
