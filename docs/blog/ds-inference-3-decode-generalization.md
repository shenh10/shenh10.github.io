---
date: 2025-03-17
title: "DeepSeek V3/R1 Inference Efficiency (3): Generalizing the Decode Configuration"
description: A simulator built on DeepGEMM, FlashMLA and torch, sweeping DP-EP and TP-DP-EP configurations on H800 and H20 — what device count, batch size, KV cache dtype and overlap scheme actually maximize throughput under a latency SLO.
---

# DeepSeek V3/R1 Inference Efficiency (3): Generalizing the Decode Configuration

> Originally published in Chinese on [Zhihu](https://zhuanlan.zhihu.com/p/29540042383), March 17, 2025.

::: tip This is part 3 of 3
[Part 1](/blog/ds-inference-1-throughput-ceiling) estimated a throughput ceiling from the V3 paper; [part 2](/blog/ds-inference-2-reverse-engineering) reverse-engineered DeepSeek's published EP144 deployment. This part generalizes that result across device counts, batch sizes and hardware.
:::

## 1. Introduction

With the [reverse-engineering](/blog/ds-inference-2-reverse-engineering) done, there is enough data and method to answer two questions:

- How does the DP-EP device count affect throughput?
- How would other hardware run DeepSeek V3/R1?

Prefill is comparatively simple, so this post covers only the generalization of decode.

**TL;DR**

While this was being written, zarbot published a theoretical estimation method for DeepSeek V3 ([DeepSeek-V3/R1 推理效率分析 v0.17](https://mp.weixin.qq.com/s/214lYyKmL3XmPUHnnTrbXg)), iterating rapidly. It is well suited to estimating performance across hardware platforms. Pure theory carries some error in the MFU of mm / grouped mm / bmm / attention and in modelling overlap, but for cross-platform estimation in the absence of a real implementation it is very useful, and worth recommending.

This post takes a different approach: integrating existing performance libraries — DeepGEMM, FlashMLA and torch — into a simulator that measures directly, with the overlap modelled in detail, which should make the results comparatively accurate. The code is open source [here](https://github.com/shenh10/DeepSeek_Simulator). To try it on non-Hopper hardware, adapt your own operator benchmark library to the test script's format and the simulation runs automatically.

**Hardware:** H800 80G, H20 96G.

**Parallelism:** Attention DP + MoE EP; Attention TP+DP + MoE EP.

**Overlap:** two-microbatch overlapping (DeepSeek's own scheme); single-batch compute-communication overlapping.

Full results and analysis are in section 4.

## 2. Generalizing the H800 DP-EP configuration

**Correcting the weight-conservation identity**

- Per-GPU batch size: $b_\text{mla}^\text{PerGPU}$
- Per-microbatch batch size: $b_\text{mla}$
- 32 extra experts per layer: $42\ \text{MB} \times 32 \times 58 = 78$ GB
- Extra shared experts in total: $42\ \text{MB} \times 58 (d-1) = 2.38(d-1)$ GB
- Per FlashMLA, the KV cache is BF16 rather than FP8, so per-token storage is $34.3 \times 2 = 68.6$ KB.

The corrected H800 weight identity is

$$80d - 13d/1.024^3 - 612 - 78 - 2.38(d-1) = \frac{34.3 \times 2}{1024 \times 1024} d \, b_\text{mla}^\text{PerGPU} s' + C_\text{act} \, b_\text{mla}^\text{PerGPU} s' d$$

which simplifies to

$$\left(56.6 - 65.4 \times 10^{-6} \, b_\text{mla}^\text{PerGPU} s'\right) d = 688$$

giving an upper bound on $b_\text{mla}^\text{PerGPU}$ as a function of $d$.

Enumerating over H800 device counts $d$ gives the configurations available at each $b_\text{mla}$. Two things are worth noting:

- An FP8 KV cache raises $b_\text{mla}$ substantially, so its throughput is worth estimating. (Assume FlashMLA's FP8 dequant cost is negligible and approximate with FlashMLA BF16 attention performance.)
- Either the microbatch-1 or the microbatch-2 overlap pipeline can be used — see below.

The largest power of two below $b_\text{mla}^\text{PerGPU,max}$ (FP8, no activation) is taken as the upper bound for a single microbatch's $b_\text{mla}$. Configurations unreachable in a real deployment are simply ignored when estimating performance.

![](/blog/ds-inference-3-decode-generalization/fig01.jpg)

## 2.1 Dense GEMM performance across shapes

From DeepGEMM's dense GEMM curve, FP8 peaks at only 1400 TFLOPS — 70% MFU — and saturates earlier than BF16.

Because of the memory constraint, $b_\text{mla}$ stays within 256, which reaches only the leftmost four points on the curve. Over that range TFLOPS grows roughly linearly with $b_\text{mla}$, with the speedup varying slightly by shape.

![](/blog/ds-inference-3-decode-generalization/fig02.jpg)

Dense GEMM efficiency depends only on $b_\text{mla}$, not on the device count $d$, which gives the decode dense-GEMM performance directly:

![](/blog/ds-inference-3-decode-generalization/fig03.jpg)

## 2.2 MLA attention performance across batch sizes

FlashMLA peaks at just over 500 TFLOPS. I have not studied the FlashMLA implementation closely; this may relate to the partial warp pipeline discussed in [ChiveArchitect's per-GPU throughput estimates for DeepSeek 671B across chips](https://zhuanlan.zhihu.com/p/28963593305).

![](/blog/ds-inference-3-decode-generalization/fig04.jpg)

## 2.3 Device count and all-to-all communication

The communication side deserves some discussion.

- On H800 the ConnectX-7 has a theoretical 50 GB/s, measured at 39 GB/s, in a per-GPU-per-NIC topology.
- The all-to-all considered here is always bound by inter-device communication. On H800, intra-machine NVLINK gives 160 GB/s / 7 = 29 GB/s, below the 40 GB/s inter-machine figure, so with two or fewer machines the intra-node all-to-all could bind instead. That is a relatively unlikely case — 7 experts inside the machine, 1 across — so it is ignored here. Two extensions are worth noting:
  - If the NIC moves to something like ConnectX-8 (an estimated 80 GB/s measured) while intra-machine stays at 160 GB/s, intra-node all-to-all can become slower than inter-node at small node counts. Expert locality requirements then relax, and intra- and inter-machine traffic have to be balanced to optimize the all-to-all. Out of scope here.
  - On other card types — PCIe cards, whether behind a PCIe switch or an AMD CPU direct connection — intra-machine all-to-all easily becomes the global bottleneck. Throughput will not be good in that case; the network is the thing to upgrade. Also out of scope.
- Across device counts from 16 to 288, the volume sent depends on $d$: under intra-device parallelism, with only 4 nodes each token sends at most 3 duplicates, and with 2 nodes at most 1.

![](/blog/ds-inference-3-decode-generalization/fig05.jpg)

## 2.4 Device count and batch size in the routed-expert GEMM

$$\text{num\_groups} = \lceil 288/d \rceil, \qquad \text{m\_per\_group} = b_\text{mla} \cdot 8 \cdot d / (256+32) = b_\text{mla} \cdot d / 36$$

The parameter combinations are too numerous to list; a representative subset:

![](/blog/ds-inference-3-decode-generalization/fig06.jpg)

Some trends are visible. The smaller the device count, the larger the per-GPU $b_\text{mla}$ has to be for the per-expert token count to drive the grouped GEMM to compute bound. And at equal total GFLOPs, a larger `m_per_group` means higher arithmetic intensity and higher TFLOPS.

![](/blog/ds-inference-3-decode-generalization/fig07.jpg)

![](/blog/ds-inference-3-decode-generalization/fig08.jpg)

![](/blog/ds-inference-3-decode-generalization/fig09.jpg)

*TFLOPS at n = 4096, k = 7168 across (num_group, m_per_group)*

## 2.5 Batch size and memory ops

Empirical values from prefill are reused directly here. Ideally, latency would be scaled by bandwidth ratio across card types, but these terms are small and fiddly, and later estimates progressively ignore them — the data below shows that whether these memory-bound operators are counted makes very little difference to the order of magnitude.

![](/blog/ds-inference-3-decode-generalization/fig10.jpg)

## 2.6 Assembling the pipeline

Wrapping the components together gives the following:

![](/blog/ds-inference-3-decode-generalization/fig11.jpg)

How should the pipeline be assembled?

### 2.6.1 Two-microbatch overlapping

The first option is the scheme DeepSeek uses, which assumes attention dominates the compute:

| Module | GFLOPs | Share of FLOPs |
| --- | --- | --- |
| MLA attention | $1.39\, b_\text{mla}$ | 50% |
| O projection | $0.234\, b_\text{mla}$ | 8% |
| Routed up & gate | $0.47\, b_\text{mla}$ | 17% |
| Routed down | $0.235\, b_\text{mla}$ | 8% |

Combine is also relatively expensive. So the long attention is overlapped with combine, and dispatch is overlapped naturally with the smaller QKV and shared-expert operators.

**Case 1: $d = 72$, $b_\text{mla} = 64$.** A single layer's microbatch forward takes $250 + 88 + 88 = 426$ µs, or $41.3 + 250 + 25.1 + 88 + 4.67 = 409$ µs excluding communication. TPOT is 51.9 ms, per-GPU throughput 2468 tokens/s.

![](/blog/ds-inference-3-decode-generalization/fig12.jpg)

### 2.6.2 Single-batch compute-communication overlapping

Two microbatches has side effects: splitting the batch lowers arithmetic intensity, and TPOT has to wait for both microbatches to finish, which is not favourable for latency.

An alternative is to overlap the down GEMM with combine within a single batch.

**Case 2: $d = 32$, $b_\text{mla} = 64$.**

With two microbatches, communication on the critical path is essentially hidden, leaving compute and memory: a single layer's microbatch forward is $41.3 + 250 + 25.1 + 164 + 4.67 = 485$ µs, TPOT $= 485 \times 2 \times 61 / 1000 = 59$ ms, per-GPU throughput 2169 tokens/s.

![](/blog/ds-inference-3-decode-generalization/fig13.jpg)

With a single microbatch: assume tiling can overlap the combine communication with the down GEMM's compute (the O + gate against dispatch overlap is hard to write, so assume it cannot overlap). This works only when combine and the down GEMM are comparable — here the down GEMM is 56 µs against 64 µs of combine, so a good deal overlaps and the communication is hidden. A single layer's microbatch forward is $41.3 + 250 + 32 + 108 + 64 + 4.67 = 500$ µs, TPOT $= (500 \times 58 + 485 \times 3)/1000 = 30$ ms, per-GPU throughput 2133 tokens/s. **Giving up 3% of throughput nearly halves TPOT.** This overlap is worth considering whenever combine and the down GEMM are of comparable magnitude.

![](/blog/ds-inference-3-decode-generalization/fig14.jpg)

**Case 3: $d = 32$, $b_\text{mla} = 128$.**

Since two microbatches implies $b_\text{mla}^\text{PerGPU} = 128$, consider a single microbatch at 128. A single layer's microbatch forward is $49.6 + 417 + 64 + 121 + 128 + 9.34 = 789$ µs, or $49.6 + 417 + 31.3 + 121 + 60 + 9.34 = 688$ µs excluding communication. TPOT $= (789 \times 58 + 688 \times 3)/1000 = 48$ ms, per-GPU throughput 2667 tokens/s — far above the two-microbatch figure of 2169 tokens/s, and with lower TPOT.

The underlying reason: **dispatch and combine are not especially expensive here, so accepting some bubble avoids both the loss of arithmetic intensity and the latency penalty of two microbatches. Where combine is long, throughput would visibly drop instead.**

![](/blog/ds-inference-3-decode-generalization/fig15.jpg)

### 2.6.3 The general form

The pipeline analysis generalizes to the following.

**1) Two-microbatch overlapping**

$$t_\text{moe\_layer} = 2\left(\max(t_\text{Dispatch},\, t_\text{Shared} + t_\text{Reduce} + t_\text{QKV}) + t_\text{MLP} + \max(t_\text{Attn+O+Gate},\, t_\text{Combine})\right)$$

$$t_\text{dense\_layer} = 2\left(t_\text{Shared} + t_\text{Reduce} + t_\text{QKV} + t_\text{MLP} + t_\text{Attn+O+Gate}\right)$$

$$\text{TPOT} = t_\text{moe\_layer} \cdot 58 + t_\text{dense\_layer} \cdot 3, \qquad T_\text{overall} = 2 b_\text{mla} / \text{TPOT}$$

**2) Single-batch compute-communication overlapping**

$$t_\text{dense\_layer} = t_\text{Shared} + t_\text{Reduce} + t_\text{QKV} + t_\text{MLP} + t_\text{Attn+O+Gate}$$

$$t_\text{moe\_layer} = \max(t_\text{Dispatch},\, t_\text{Shared}) + t_\text{Reduce} + t_\text{QKV} + t_\text{Up\&Gate} + \max(t_\text{Down},\, t_\text{Combine}) + t_\text{Attn+O+Gate}$$

$$\text{TPOT} = t_\text{moe\_layer} \cdot 58 + t_\text{dense\_layer} \cdot 3, \qquad T_\text{overall} = b_\text{mla} / \text{TPOT}$$

## 2.7 H800 throughput

Partial H800 throughput data follows. (The $d$ / $b_\text{mla}$ combinations swept here are incomplete; more configurations appear later.)

**1) Two-microbatch overlapping**

![](/blog/ds-inference-3-decode-generalization/fig16.jpg)

**2) Single-batch compute-communication overlapping**

![](/blog/ds-inference-3-decode-generalization/fig17.jpg)

Computing memory-op durations is fiddly. Drawing them into the overlap diagrams illustrates overlap efficiency more clearly, but their share really is small enough to ignore. So how far off is an estimate that counts only GEMM + attention + all-to-all?

**1) Two-microbatch overlapping**

![](/blog/ds-inference-3-decode-generalization/fig18.jpg)

**2) Single-batch compute-communication overlapping**

![](/blog/ds-inference-3-decode-generalization/fig19.jpg)

Comparing before and after, GEMM + attention + all-to-all captures the model's real latency and throughput well enough. **Subsequent estimates therefore drop the memory-op terms**, which simplifies the model towards something general.

## 3.1 The compute required to meet the latency target

Total compute FLOPs for a single layer and single microbatch: $2.79\, b_\text{mla}$ GFLOPs.

| Module | GFLOPs | Share of FLOPs |
| --- | --- | --- |
| MLA attention | $1.39\, b_\text{mla}$ | 50% |
| O projection | $0.234\, b_\text{mla}$ | 8% |
| Routed up & gate | $0.47\, b_\text{mla}$ | 17% |
| Routed down | $0.235\, b_\text{mla}$ | 8% |

MLA attention runs in BF16 and the other GEMMs in FP8. Assuming MFU = 100%, meeting a 20 tokens/s target requires:

**H800**

$$\left(\frac{1.39\, b_\text{mla}}{1024 \times 989} + \frac{(2.79 - 1.39) b_\text{mla}}{1024 \times 1978}\right) \times 1000 \times 61 \le 50$$

giving $b_\text{mla} \le 397$, for a per-GPU ceiling of 7940 tokens/s.

**H20**

$$\left(\frac{1.39\, b_\text{mla}}{1024 \times 148} + \frac{(2.79 - 1.39) b_\text{mla}}{1024 \times 296}\right) \times 1000 \times 61 \le 50$$

giving $b_\text{mla} \le 59$, for a per-GPU ceiling of 1180 tokens/s. (This differs from the theoretical ceiling in [part 1](/blog/ds-inference-1-throughput-ceiling) because MLA is now accounted for as a BF16 implementation.)

Taking the maximum $b_\text{mla}$ for each card, and measuring effective KV cache utilization as $\text{kv\_utility} = \min(b_\text{mla}^\text{latency\_bound}, b_\text{mla}^\text{peak}) / b_\text{mla}^\text{peak}$:

```text
GPU Type:  H800-80
Device number:  (16, 24, 32, 48, 72, 96, 144, 288)
Max batch size per GPU:  (133, 220, 264, 308, 337, 352, 366, 381)

GPU Type:  H20-96
Device number:  (16, 24, 32, 48, 72, 96, 144, 288)
Max batch size per GPU:  (231, 318, 362, 406, 435, 449, 464, 478)
```

| Device count | H800, FP8 KV cache | H20, FP8 KV cache | H20, BF16 KV cache |
| --- | --- | --- | --- |
| 16 | 100% | 26% | 52% |
| 24 | 100% | 19% | 38% |
| 32 | 100% | 16% | 32% |
| 48 | 100% | 15% | 30% |
| 72 | 100% | 14% | 28% |
| 96 | 100% | 13% | 26% |
| 144 | 100% | 13% | 26% |
| 288 | 100% | 12% | 24% |

Under a latency constraint, scaling H20 to buy KV cache is useless — the memory cannot be effectively utilized in the first place.

**For low-compute cards, TP is needed to relax the latency constraint and buy better scaling.** Since attention plus O projection accounts for 58% of the compute, those are the matrices to shard.

## 3.2 MLA (TP-DP) + MoE EP

The MoE expert GEMM shapes are relatively small, so TP is not advisable there. For modelling simplicity and communication efficiency, TP on the attention is confined to within a machine.

MLA is sharded on the head dimension of Q, which affects the matrix shapes as:

$$\text{Concat}(W_Q', W_p, W_{KR}): [h,\, c' + c + r] = [7168,\, 2112]$$
$$\text{Concat}(W_{UQ}, W_{QR}): [c',\, (a/N)(d+r)] = [1536,\, (128/N) \cdot 192]$$
$$W_{UK},\, W_{UV}: 2 \times [c,\, (a/N) d] = 2 \times [512,\, (128/N) \cdot 128]$$
$$O: [(a/N) d,\, h] = [(128/N) \cdot 128,\, 7168]$$

The compute matrix from 3.1, under TP:

| Module | TP=1 | TP=2 | TP=4 | TP=8 |
| --- | --- | --- | --- | --- |
| MLA attention | 1.39 | 0.70 | 0.35 | 0.17 |
| O projection | 0.23 | 0.12 | 0.06 | 0.03 |
| Routed up & gate | 0.47 | 0.47 | 0.47 | 0.47 |
| Routed down | 0.24 | 0.24 | 0.24 | 0.24 |
| Others | 0.46 | 0.46 | 0.46 | 0.46 |
| **Total** | **2.79** | **1.98** | **1.57** | **1.37** |

(All figures in GFLOPs, scaled by $b_\text{mla}$.)

At TP = 8 the compute ceiling updates to:

**H800** — $\left(\frac{0.17 b_\text{mla}}{1024 \times 989} + \frac{(1.37 - 0.17) b_\text{mla}}{1024 \times 1978}\right) \times 1000 \times 61 \le 50$, raising the batch-size ceiling from $b_\text{mla} = 397$ to $b_\text{mla} = 1077$.

**H20** — $\left(\frac{0.17 b_\text{mla}}{1024 \times 148} + \frac{(1.37 - 0.17) b_\text{mla}}{1024 \times 296}\right) \times 1000 \times 61 \le 50$, raising it from $b_\text{mla} = 59$ to $b_\text{mla} = 161$.

### 3.2.1 Updating the memory-conservation identity

Of the 14B dense weights, MLA accounts for approximately

$$(7168 \times 2112 + 1536 \times 24576 + 512 \times 32768 + 16384 \times 7168) \times 61 = 11.4\text{B}$$

Per GPU under TP, the MLA parameter count is

$$(7168 \times 2112 + 1536 \times 24576/tp + 512 \times 32768/tp + 16384 \times 7168/tp) \times 61$$

so across $d$ devices the total MLA weight is $(0.86 + 9.77/tp) \cdot d$, where $d = tp \cdot dp$. The new identity is

$$C d - (2.42 + 0.86 + 9.77/tp) d - 612 - 78 - 2.38(d-1) = \frac{34.3 \times 2}{1024 \times 1024} d\, b_\text{mla}^\text{PerGPU,max} s'$$

which gives the updated memory utilization:

```text
GPU Type: H20-96
TP= 1   Device Number: [16, 24, 32, 48, 72, 96, 144, 288]
        Max batch size per GPU: [231, 318, 362, 406, 435, 449, 464, 478]
TP= 2   Device Number: [16, 24, 32, 48, 72, 96, 144, 288]
        Max batch size per GPU: [240, 328, 371, 415, 444, 459, 473, 488]
TP= 4   Device Number: [16, 24, 32, 48, 72, 96, 144, 288]
        Max batch size per GPU: [245, 332, 376, 420, 449, 463, 478, 493]
TP= 8   Device Number: [16, 24, 32, 48, 72, 96, 144, 288]
        Max batch size per GPU: [247, 335, 378, 422, 451, 466, 480, 495]
```

| Device count | H20, FP8 KV cache | H20, BF16 KV cache |
| --- | --- | --- |
| 16 | 65% | 100% |
| 24 | 48% | 96% |
| 32 | 43% | 86% |
| 48 | 38% | 76% |
| 72 | 36% | 72% |
| 96 | 35% | 70% |
| 144 | 34% | 68% |
| 288 | 33% | 66% |

### 3.2.2 Updating the token-dispatch formulas

With TP enabled, a TP group shares tokens, so the redundancy between TP ranks has to be removed when counting MoE EP tokens.

Routed expert compute:

$$\text{num\_groups} = \lceil 288/d \rceil, \qquad \text{m\_per\_group} = \frac{b_\text{mla} \cdot \text{topk} \cdot dp}{288}, \quad d = dp \cdot tp$$

Routed all-to-all volume: the all-to-all within a TP group can be amortized across its GPUs, so each GPU only sends the activations of $b_\text{mla}/tp$ tokens:

$$\text{Dispatch} = \frac{b_\text{mla}}{tp} \min(\text{num\_node} - 1,\, \text{topk}) \cdot 7168$$
$$\text{Combine} = \frac{b_\text{mla}}{tp} \min(\text{num\_node} - 1,\, \text{topk}) \cdot 7168 \times 2$$

### 3.2.3 Updating communication and the pipeline

Attention TP requires an all-reduce after the O projection, with BF16 traffic of $\frac{2(tp-1)(2 b_\text{mla} \cdot 7168)}{tp}$. At 160 GB/s of unidirectional NVLINK bandwidth, all-reduce durations (µs) are:

| $b_\text{mla}$ | tp=2 | tp=4 | tp=8 |
| --- | --- | --- | --- |
| 8 | 1 | 1 | 1 |
| 16 | 1 | 2 | 2 |
| 32 | 3 | 4 | 5 |
| 64 | 5 | 8 | 9 |
| 128 | 11 | 16 | 19 |
| 256 | 21 | 32 | 37 |

At small volumes this is purely latency-bound, so an empirical floor of around 5 µs applies.

Against the O projection GEMM's own duration on H800:

| $b_\text{mla}$ | tp=2 | tp=4 | tp=8 |
| --- | --- | --- | --- |
| 8 | 31 | 17 | 9 |
| 16 | 30 | 16 | 9 |
| 32 | 26 | 14 | 8 |
| 64 | 25 | 14 | 7 |
| 128 | 26 | 14 | 8 |
| 256 | 31 | 17 | 11 |

At large $b_\text{mla}$ and large $tp$ the all-reduce already exceeds the O GEMM, and overlapping the two becomes worth considering; at small $b_\text{mla}$ it is latency-bound and not worth it. Over an interconnect this fast, leaving the all-reduce unoverlapped is acceptable — it is far smaller than the all-to-all — so the model assumes no overlap.

**Single-batch compute-communication overlapping:**

![](/blog/ds-inference-3-decode-generalization/fig20.jpg)

**Two-microbatch overlapping:**

![](/blog/ds-inference-3-decode-generalization/fig21.jpg)

The overlap formulas update accordingly. Note the introduction of *effective* throughput: with $tp > 1$ the GPUs in a TP group process the same data, so the per-GPU figure divides by $tp$.

**1) Two-microbatch overlapping**

$$t_\text{moe\_layer} = 2\left(\max(t_\text{Dispatch},\, t_\text{Shared} + t_\text{Reduce} + t_\text{QKV}) + t_\text{MLP} + \max(t_\text{Attn+O} + t_\text{AllReduce} + t_\text{Gate},\, t_\text{Combine})\right)$$

$$t_\text{dense\_layer} = 2\left(t_\text{Shared} + t_\text{Reduce} + t_\text{QKV} + t_\text{MLP} + t_\text{Attn+O} + t_\text{AllReduce} + t_\text{Gate}\right)$$

$$\text{TPOT} = t_\text{moe\_layer} \cdot 58 + t_\text{dense\_layer} \cdot 3, \qquad T_\text{overall\_effective} = \frac{2 b_\text{mla}}{tp \cdot \text{TPOT}}$$

**2) Single-batch compute-communication overlapping**

$$t_\text{dense\_layer} = t_\text{Shared} + t_\text{Reduce} + t_\text{QKV} + t_\text{MLP} + t_\text{Attn+O} + t_\text{AllReduce} + t_\text{Gate}$$

$$t_\text{moe\_layer} = \max(t_\text{Dispatch},\, t_\text{Shared}) + t_\text{Reduce} + t_\text{QKV} + t_\text{Up\&Gate} + \max(t_\text{Down},\, t_\text{Combine}) + t_\text{Attn+O} + t_\text{AllReduce} + t_\text{Gate}$$

$$\text{TPOT} = t_\text{moe\_layer} \cdot 58 + t_\text{dense\_layer} \cdot 3, \qquad T_\text{overall\_effective} = \frac{b_\text{mla}}{tp \cdot \text{TPOT}}$$

## 4. H800 and H20 results

Test environment: CUDA 12.6, PyTorch 2.4, Python 3.10.

Simulation results are generated for Attention DP + MoE EP and Attention TP+DP + MoE EP under both pipeline schemes.

- **Yellow** marks the best throughput at each device count that roughly meets a 20 tokens/s user latency.
- **Orange** marks the best online configuration across all device counts meeting that latency.
- **Green** marks maximum FP8 KV cache throughput ignoring the latency constraint.
- **Blue** marks maximum BF16 KV cache throughput ignoring the latency constraint.

Full results are [here](https://github.com/shenh10/DeepSeek_Simulator/tree/main/results).

### 4.1 H800

- DP-EP gives the best offline throughput; enabling attention TP hurts it.
- Best offline throughput with BF16 KV cache: **2844 tokens/s** per GPU, from two-microbatch overlapping at DP288-EP288, $b_\text{mla} = 64$.
- Best offline throughput with FP8 KV cache: **3121 tokens/s** per GPU, from two-microbatch overlapping at DP288-EP288, $b_\text{mla} = 128$, or at DP48-EP48, $b_\text{mla} = 128$.
- Best online throughput with FP8 KV cache (meeting ~20 tokens/s): **2909 tokens/s**, from single-batch compute-communication overlapping at DP24-EP24, $b_\text{mla} = 128$.
- Best online throughput with BF16 KV cache: **2844 tokens/s**, from two-microbatch overlapping at DP288-EP288, $b_\text{mla} = 64$.

![](/blog/ds-inference-3-decode-generalization/fig22.jpg)

*Two-microbatch overlapping, best configurations — H800*

![](/blog/ds-inference-3-decode-generalization/fig23.jpg)

*Single-batch overlapping, best configurations — H800*

As expected, at $d \le 32$ single-batch compute-communication overlapping outperforms two-microbatch overlapping — an encouraging result for anyone who cannot run at high concurrency.

Beyond DeepSeek's own EP144, the marginal return from going to EP288 is fairly limited; the [full results](https://github.com/shenh10/DeepSeek_Simulator/tree/main/results) are there to choose from. Realizing the FP8 KV cache means writing an FP8-KV-cache MLA kernel; realizing single-batch overlapping means writing the down-GEMM / combine all-to-all overlap. Both require some infrastructure work, though neither is difficult.

### 4.2 H20

- DP-EP again gives the best offline throughput, and attention TP hurts it. But on a low-compute card like H20, attention TP helps reach a better TPOT at the same node count, which is what meeting the SLO requires. Note that **too much TP reduces effective per-GPU throughput and lowers overall throughput**; TP = 2 measured as the reasonable online configuration.
- Best offline throughput with BF16 KV cache: **969 tokens/s** per GPU, from two-microbatch overlapping at DP72-EP72, $b_\text{mla} = 64$.
- Best offline throughput with FP8 KV cache: **980 tokens/s** per GPU, from two-microbatch overlapping at DP72-EP72, $b_\text{mla} = 128$.
- Best online throughput, either KV cache dtype (meeting ~20 tokens/s): **820 tokens/s**, from single-batch compute-communication overlapping at DP48-EP48 with $b_\text{mla} = 32$, or TP2-DP24-EP48 with $b_\text{mla} = 64$.

![](/blog/ds-inference-3-decode-generalization/fig24.jpg)

*Two-microbatch overlapping, best configurations — H20*

![](/blog/ds-inference-3-decode-generalization/fig25.jpg)

*Single-batch overlapping, best configurations — H20*

## 5. Conclusion

That completes the goal of the series: generalizing DeepSeek R1's deployment configuration. Small instance groups can run single-batch overlapping themselves, or adopt an FP8 KV cache to raise throughput, which opens up a considerably wider search space of deployment options. DeepSeek's published configuration is not the only choice, and an instance group as large as DP144-EP144 is not required for good online service — trading away some throughput buys a deployment that is more flexible and simpler to run.

H20's throughput efficiency comes out at roughly 30% of H800's, which is the figure to use when working out TCO. I do not have heterogeneous hardware to hand, but this tooling should adapt readily to any chip with a basic operator library, and produce reasonable simulated figures from it.

The post involves a great deal of arithmetic; corrections are welcome.
