# 博客

记录技术探索与思考。

## 文章列表

### [HuggingArch：让模型 arch 分析自动化](/zh/blog/huggingarch)

用 harness 的思路让 LLM 推理算账自动化：三层 spec 系统、贯穿前后端的 sympy 符号代数，以及一套把 agent 钉死在 ground truth 上的校验体系。文末附 DeepSeek V4-Pro 的翻车实验——不设 guard 时 agent 有 94% 概率把错的 spec 自判为通过。

### [GPU-to-GPU Copy over PCIe: From cudaMemcpyAsync to a Custom Kernel](/blog/gpu-d2d-pcie) <Badge type="info" text="English" />

PCIE 拓扑下两块 GPU 之间的 D2D 拷贝到底能跑到多少带宽？从 naive `cudaMemcpyAsync` 出发，依次加上 pinned memory 的 double buffer 流式传输、peer access、向量化的自定义 copy kernel，每一步都用 Nsight 看清楚底下发生了什么，最后和 NCCL 对比。A800 PCIE 与 4090 PCIE 实测。中文原文见[知乎](https://zhuanlan.zhihu.com/p/2847929235)。

### [GPU Clock Throttling: Why You Never Reach Peak FLOPS](/blog/gpu-throttling) <Badge type="info" text="English" />

大矩阵 GEMM 跑不到标称算力，通常不是 vendor 库不够好，而是功耗预算限死了。讲清楚 SM/memory clock 在负载下的行为、怎么用 NVML 的 `clocks_throttle_reasons` 抓现行，以及在 T4、A10、A800 SXM/PCIE、H800 SXM 上的 FP16 方阵 GEMM 扫描——峰值 MFU 从 56% 到 89% 不等。中文原文见[知乎](https://zhuanlan.zhihu.com/p/13866293937)。

### [DeepSeek V3/R1 Inference Efficiency (1): A Back-of-the-Envelope Decoding Throughput Ceiling](/blog/ds-inference-1-throughput-ceiling) <Badge type="info" text="English" />

DeepSeek 公布推理系统数据之前，只靠 V3 论文能估到多准？Attention DP + MoE EP 下的显存与算力预算，以及它推出的单卡 tokens/s 上限。中文原文见[知乎](https://zhuanlan.zhihu.com/p/27292649125)。


### [DeepSeek V3/R1 Inference Efficiency (2): Reverse-Engineering the Production Deployment](/blog/ds-inference-2-reverse-engineering) <Badge type="info" text="English" />

对官方 EP144 部署的逐层 profiling——prefill 与 decode 的 FLOPs、逐算子耗时、MFU，以及由此推出的 overlap 排布。中文原文见[知乎](https://zhuanlan.zhihu.com/p/29841050824)。


### [DeepSeek V3/R1 Inference Efficiency (3): Generalizing the Decode Configuration](/blog/ds-inference-3-decode-generalization) <Badge type="info" text="English" />

基于 DeepGEMM、FlashMLA 与 torch 搭的模拟器，在 H800 与 H20 上扫 DP-EP 与 TP-DP-EP——延迟 SLO 约束下，哪种设备数、batch、KV cache dtype 与 overlap 方式吞吐最高。中文原文见[知乎](https://zhuanlan.zhihu.com/p/29540042383)。
