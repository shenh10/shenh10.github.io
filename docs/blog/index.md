# 博客

记录技术探索与思考。

## 文章列表

### [HuggingArch：让模型 arch 分析自动化](/blog/huggingarch)

用 harness 的思路让 LLM 推理算账自动化：三层 spec 系统、贯穿前后端的 sympy 符号代数，以及一套把 agent 钉死在 ground truth 上的校验体系。文末附 DeepSeek V4-Pro 的翻车实验——不设 guard 时 agent 有 94% 概率把错的 spec 自判为通过。

### [GPU-to-GPU Copy over PCIe: From cudaMemcpyAsync to a Custom Kernel](/en/blog/gpu-d2d-pcie) <Badge type="info" text="English" />

PCIE 拓扑下两块 GPU 之间的 D2D 拷贝到底能跑到多少带宽？从 naive `cudaMemcpyAsync` 出发，依次加上 pinned memory 的 double buffer 流式传输、peer access、向量化的自定义 copy kernel，每一步都用 Nsight 看清楚底下发生了什么，最后和 NCCL 对比。A800 PCIE 与 4090 PCIE 实测。中文原文见[知乎](https://zhuanlan.zhihu.com/p/2847929235)。

### [GPU Clock Throttling: Why You Never Reach Peak FLOPS](/en/blog/gpu-throttling) <Badge type="info" text="English" />

大矩阵 GEMM 跑不到标称算力，通常不是 vendor 库不够好，而是功耗预算限死了。讲清楚 SM/memory clock 在负载下的行为、怎么用 NVML 的 `clocks_throttle_reasons` 抓现行，以及在 T4、A10、A800 SXM/PCIE、H800 SXM 上的 FP16 方阵 GEMM 扫描——峰值 MFU 从 56% 到 89% 不等。中文原文见[知乎](https://zhuanlan.zhihu.com/p/13866293937)。

