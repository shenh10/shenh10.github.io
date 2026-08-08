---
date: 2025-03-17
title: "DeepSeek V3/R1 Inference Efficiency (1): A Back-of-the-Envelope Decoding Throughput Ceiling"
short: "DeepSeek Inference Efficiency (1): Throughput Ceiling"
titleZh: "DeepSeek 推理效率分析（1）：吞吐极限估计"
description: Before DeepSeek published its inference-system numbers, how close could you get from the V3 paper alone? A memory-and-compute budget for Attention DP + MoE EP, and what it says about the achievable tokens/s per H800.
---

# DeepSeek V3/R1 Inference Efficiency (1): A Back-of-the-Envelope Decoding Throughput Ceiling

> Originally published in Chinese on [Zhihu](https://zhuanlan.zhihu.com/p/27292649125), March 17, 2025.

::: tip This is part 1 of 3
The series follows the analysis as it developed: part 1 is a coarse order-of-magnitude estimate from the V3 paper's parameters alone; parts 2 and 3 refine it quantitatively against the figures DeepSeek subsequently published. Precision improves substantially across the three, so anyone interested in the topic should read through to part 3 — several questions raised about the accuracy of part 1 are analysed and corrected there.
:::

DeepSeek published [an overview of its V3/R1 inference system](https://zhuanlan.zhihu.com/p/27181462601). Before those numbers were available, and without the three hundred–odd GPUs needed to run the experiment properly — large distributed MoE inference had not been anticipated internally, and the scale-out capability of SGLang and vLLM was still being built — we had made some theoretical estimates, as an attempt to reverse-engineer how DeepSeek's deployment must be configured.

What follows is that estimate, tidied up. A number of arithmetic errors were caught in discussion with colleagues in the field, which is reflected here.

**Acknowledgements.** The initial batch-size discussion came out of conversations with Changcheng Tang and Xiangcheng Liu from Kuaishou's LLM inference team; Yuliang Liu reviewed some of the conclusions. The compute and communication ceilings were reviewed and corrected by Mingran Wang of SambaNova, and the network-ceiling method follows zarbot's approach.

### Parameters

MLA:

- $n_h = 128$, $d_h = 128$
- KV compression dim $d_c = 512$, query compression dim $d_c' = 1536$, $d_h^R = 64$

Expert — three GEMMs (up, down, gate): $d_h = 7168$, $d_\text{expert} = 2048$.

DeepSeek-V3 has 671B parameters total, of which 37B activate per token. It has 61 transformer layers: the first 3 are non-MoE, and the remaining 58 have their FFN expanded into an MoE of 1 shared expert plus 256 routed experts, with $h = 2048$. Each token activates 8 routed experts; during training each token is sent to at most 4 nodes to limit inter-node traffic.

### Dense vs. sparse parameter split

A single expert's weights are $7168 \times 2048 \times 3 = 44$M. With 3 of 61 layers dense and 58 MoE, $257 \times 58 + 3 \times 9$ experts gives 657B, so the dense part is $671 - 657 = 14$B — of which MLA accounts for 11.4B and the rest is embedding, linear and so on. For a detailed derivation see [ZihaoZhao: decomposing DeepSeek-V3's 671B parameter count](https://zhuanlan.zhihu.com/p/21455638257).

![](/blog/ds-inference-1-throughput-ceiling/fig01.jpg)

*DeepSeek-V3 parameter distribution*

Working backwards from the 37B activated per token: $37\text{B} - 44\text{M} \times 61 \times 9 / 1000 = 12.8$B, which roughly matches the 14B.

So as approximate weight figures:

- per expert: 44M = 42 MB
- dense part: 14B ≈ 13 GB
- all experts: 657B ≈ 612 GB

### Average sequence length

Assumptions on the input/output length distribution:

- DeepSeek V3: 1k + 1k on average, matching NVIDIA's official benchmark convention.
- DeepSeek R1: 1k + 5k on average, consistent with the figure DeepSeek published — "the average KV cache length per output token is 4989."

### Choosing a device count that reaches compute bound

The most basic deployment of V3/R1 is 8-way TP within one machine, which needs 625 GB for weights — so the entry configuration is typically H20 96G × 8 or H800 80G × 16. In both, the KV cache capacity left over is very small, which drives down both the attention QKV GEMM batch size and the MoE batch size. The MoE experts end up firmly memory-bound, even latency-bound, which is hostile to continuous batching. **Raising batch size is therefore the basic requirement for throughput, and scaling out across machines — lowering the per-GPU weight footprint — is the basic means of doing so.**

TP is unhelpful for MLA's KV cache: MLA compresses multiple heads into a single hidden vector, so the cache cannot be split by head within a TP group, and every GPU stores a redundant copy. Scaling out therefore starts from Attention DP + TP/EP on the MoE. Accounting for cross-device communication efficiency, Attention DP + MoE EP is the more workable of the two.

The V3 paper uses 40 × 8 H800 for decode nodes: attention as 4-way TP+SP plus 80-way DP, MoE as 320-way EP. Since TP+SP stays within a machine and is not the binding constraint on batch or communication, **the rest of this post assumes Attention DP + MoE EP for simplicity.**

Which brings the basic question:

**What is the minimum number of H800s needed for the MoE experts to become compute-bound?**

> To keep the estimate tractable, assume no redundant experts, and set aside the shared expert and the three dense layers.

### 1. The memory constraint

- per-GPU sequence length: $s$ (typically 1 when decoding)
- average sequence length per request: $s'$
- per-GPU MLA batch size: $b_\text{mla}$
- per-expert batch size: $b_\text{ep}$
- GPU count: $d$
- expert count: $n_e$
- Per-GPU activation footprint: for $s' > 1000$ this is roughly a constant multiple of $b_\text{mla} s'$ — write it $C_\text{act} \cdot b_\text{mla} s'$ — dominated by MLA activations. SGLang's activation memory management is currently poor; measured with $s$ spread over 1k–8k, activations stay under 8 GB. (A smaller activation figure does not change the order of magnitude of anything below.)
- MLA KV cache per token: $(d_c + d_R) \times 61 = 576 \times 61 = 34.3$ KB in FP8. (SGLang and similar frameworks currently only support a BF16 KV cache; the ceiling here is computed for FP8 storage.)
- Assume $d \le n_e$, so each GPU holds $\lceil n_e / d \rceil$ experts on average.

The weight-conservation identity is then

$$80d - 14d/1.024^3 - 612 = \frac{34.3}{1024 \times 1024} \, d \, b_\text{mla} s' + C_\text{act} \, b_\text{mla} s' d$$

$$\left(67 - \left(32.7 \times 10^{-6} + C_\text{act}\right) b_\text{mla} s'\right) d = 612$$

Substituting the empirical 8 GB activation assumption:

$$\left(59 - 32.7 \times 10^{-6} \, b_\text{mla} s'\right) d = 612$$

**The ranges this admits for $d$ and $b_\text{mla}$:**

1. The formula assumes experts are stored without redundancy (612 GB), so $d \le 256$.
2. It requires $59 - 32.7 \times 10^{-6} b_\text{mla} s' \ge 0$, i.e. the tokens held per GPU must satisfy $b_\text{mla} s' \le 1.8 \times 10^6$.
   - Even with zero activation the ceiling is only $2 \times 10^6$.
   - Condition 1 gives a tighter bound, $59 - 32.7 \times 10^{-6} b_\text{mla} s' \ge 612/256$, for a ceiling of $1.73 \times 10^6$.
   - Neither changes the order of magnitude.

   For V3, with $s' = 2000$: $b_\text{mla} \le 900$. For R1, with $s' = 6000$: $b_\text{mla} \le 300$.
3. $59d \ge 612$, so $d \ge 11$.

**Conclusion 1.** For the Q/K/V projection matrices, at long sequence lengths no amount of H800 scale-out reaches the FP8 compute saturation point. ([DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) results put the empirical FP8 saturation point for H800 at $m > 4096$ for matmuls at this $m \times n \times k$ scale.) The larger memory of H200 141G and MI300X 192G brings QKV projection closer to saturation.

### 2. The constraint for MoE to reach compute bound

The MoE receives $b_\text{mla} \cdot d \cdot 9$ tokens in total; assuming balanced routing, each expert averages $b_\text{mla} \cdot 9 \cdot d / n_e$. Writing the MoE GEMM saturation batch size as $b_\text{ep,sat}$, we need

$$b_\text{mla} \cdot 9 \cdot d / n_e \ge b_\text{ep,sat}$$

so

$$d / n_e \ge b_\text{ep,sat} / (b_\text{mla} \cdot 9)$$

For V3 at $s' = 2000$, still taking the saturation point as $b_\text{ep,sat} = 4096$: with $b_\text{mla} = 900$, $d \ge 0.5 \times 256 \approx 128$.

For R1 at $s' = 6000$ and $b_\text{ep,sat} = 4096$: with $b_\text{mla} = 300$, $d \ge 1.51 \times 256 > 256$ — so the ideal saturation point is out of reach. At $d = 256$, $b_\text{ep} \le 2700$; at $d = 128$, $b_\text{ep} \le 1350$.

**Conclusion 2.** On H800, the MoE device count need not be large for V3 — both $d = 128$ and $d = 256$ look reasonable. For R1, $d = 256$ is the better choice for saturating the MoE. On H20, $b_\text{ep,sat}$ is small enough that compute bound is reachable well below 256 devices; on H200 and MI300X, the larger $b_\text{mla}$ likewise allows a smaller $d$.

## Estimating the throughput ceiling

Optimizing for throughput, we want a batch large enough to be compute-bound, which makes it unlikely that memory bandwidth is the overall limit — leaving compute or communication.

**1. The compute-bound ceiling**

Estimating decode FLOPs: outside MLA, the usual $2N$ approximation applies. Within MLA, the Q@K and P@V work induced by the context window also has to be counted once $s$ is large.

Decode MLA normally runs in absorbed form:

- FLOPs are $2 b_\text{mla} s' n_\text{head} (2 d_c + r)$; substituting MLA's parameters, $2 b_\text{mla} s' \cdot 128 \cdot (512 \times 2 + 64) \approx 2.8 \times 10^5 \, b_\text{mla} s'$ FLOPs.
- So decode compute is about $2 \times 37 \times b_\text{mla} \, s \, d + 61 \times 2.8 \times 10^{-4} b_\text{mla} s' d = (74 s + 170.8 \times 10^{-4} s') \, b_\text{mla} d$ GFLOPs.
- Aggregate FP8 compute is $1978d$ TFLOPS on H800 and $296d$ TFLOPS on H20.

For V3 at $s = 1$, $s' = 2000$: total FLOPs are $108 \, b_\text{mla} d$, giving a per-GPU ceiling of 18300 tokens/s on H800 and 2740 tokens/s on H20.

For R1 at $s = 1$, $s' = 6000$: total FLOPs are $176 \, b_\text{mla} d$, giving roughly 11000 tokens/s on H800 and 1681 tokens/s on H20.

**2. The network-bound ceiling** (zarbot's method)

Per token the payload is 7168 B, against 50 GB/s of inter-node RDMA bandwidth; a single dispatch across 61 layers and 9 experts totals about 4 MB. zarbot folds in an all-reduce for a total of 8 MB and concludes a per-GPU ceiling of 6000 tokens/s. But under DP+EP the DP portion carries no traffic — nothing is sent during the DP stage — so what remains is dispatch plus combine, also 8 MB, or 12 MB if combine is BF16. At 50 GB/s of RDMA bandwidth (40 GB/s in practice) that supports roughly 5000 tokens/s per GPU in FP8, or 3333 in BF16.

> Two refinements worth noting.
>
> zarbot's 9-expert figure assumes decode uses EP320 for minimum latency, leaving only 8 experts per node; estimating traffic at 9 experts is probably still the better approximation, given hop count and balance.
>
> If communication is arranged to minimize inter-node bandwidth — as in training, where a token can be capped at 4 nodes, communicating first and duplicating within the device — then only 4 nodes' worth of traffic counts. More precisely, with the first 3 transformer layers non-MoE, a single dispatch is 58 layers × 4 experts ≈ 1.6 MB. With combine in FP8 that is 3.2 MB, or 4.8 MB in BF16; at 40 GB/s effective RDMA bandwidth, the per-GPU ceiling becomes 12800 tokens/s for FP8 combine and 8500 tokens/s for BF16. This raises the communication ceiling considerably.

Taking compute and communication together, without MTP, R1 on EP256 H800 in FP8 has a per-GPU ceiling of 3300 tokens/s (BF16 combine) to 5000 tokens/s (FP8 combine); H20 sits around 1600 tokens/s.

> Two caveats.
>
> 1. The network constraint is straightforward to estimate from measured bandwidth. Where compute is the binding constraint, MFU losses have to be accounted for — 60–70% of the figure is a realistic estimate, which puts H20 nearer 1000 tokens/s.
> 2. H800 being network-bound presupposes its compute ceiling is high enough. Double the network bandwidth, or account for H800's very low MFU, and compute may bind first. The general principle holds: the narrowest pipe sets the ceiling.

## The latency constraint

Take a 10 ms latency budget. Estimating from the per-iteration memory traffic of the experts against aggregate bandwidth, we need

$$\frac{671 + 14(d-1)}{1.024^3 \cdot 3300 d} \le 0.01$$

which gives $d \ge 31$. So a basic requirement follows: **meeting an aggressive latency target requires enough machines to relieve the memory pressure of the experts.**

Against the 20 tokens/s per user SLO common in commercial MaaS: at full batch this is equivalent to the time to load 80 GB once, at minimum 80 GB / 3350 = 24 ms. At batch sizes large enough to be compute-bound this rises. At small batch, a single expert matrix read is only $7168 \times 2048 = 14$M, and MBU empirically tops out around 50%, so one full load takes an estimated 24–50 ms. (Measured TPOT at batch size 1 on one or two machines, across various lengths, falls between 35 ms and 50 ms.)

There is a trade-off here between memory-bound and compute-bound operation: scale out far enough to bring batch-size-1 latency down, then find the sweet spot in device count that maximizes batch size subject to a 50 ms TPOT constraint.

## Summary

For reference, the figures DeepSeek published:

> Total output tokens: 168B. Average output rate 20–22 tps; average KV cache length per output token, 4989.
>
> Average per-H800 throughput: for prefill, roughly 73.7k tokens/s input (including cache hits); for decode, roughly 14.8k tokens/s output.

That is 1.85k tokens/s per GPU — 56% of the 3300 tokens/s estimated above for BF16 combine (per DeepEP), while holding 20–22 tokens/s per user. Reaching 56% of the theoretical ceiling under a latency constraint is, in my view, a very strong system result, assuming MTP is not in play.

Plenty of detail remains in trading latency against throughput — whether placing more experts per GPU would raise the network-bound ceiling, for instance, and what else is available to tune.

## References

1. [zarbot: 谈谈微信+DeepSeek](https://mp.weixin.qq.com/s/TGzU5oA4hEOvqFJYzBaRSw)
