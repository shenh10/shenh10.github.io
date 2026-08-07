# 博客

记录技术探索与思考。除 HuggingArch 外均为英文。

<PostList :notes='{"huggingarch": "用 harness 的思路让 LLM 推理算账自动化：三层 spec 系统、贯穿前后端的 sympy 符号代数，以及一套把 agent 钉死在 ground truth 上的校验体系。文末附 DeepSeek V4-Pro 的翻车实验。中文版见 /zh/blog/huggingarch。", "ds-inference-3-decode-generalization": "基于 DeepGEMM、FlashMLA 与 torch 搭的模拟器，在 H800 与 H20 上扫 DP-EP 与 TP-DP-EP——延迟 SLO 约束下，哪种设备数、batch、KV cache dtype 与 overlap 方式吞吐最高。中文原文见知乎 p/29540042383。", "ds-inference-2-reverse-engineering": "对官方 EP144 部署的逐层 profiling——prefill 与 decode 的 FLOPs、逐算子耗时、MFU，以及由此推出的 overlap 排布。中文原文见知乎 p/29841050824。", "ds-inference-1-throughput-ceiling": "DeepSeek 公布推理系统数据之前，只靠 V3 论文能估到多准？Attention DP + MoE EP 下的显存与算力预算，以及它推出的单卡 tokens/s 上限。中文原文见知乎 p/27292649125。", "gpu-throttling": "大矩阵 GEMM 跑不到标称算力，通常不是 vendor 库不够好，而是功耗预算限死了。T4、A10、A800、H800 上的 FP16 方阵扫描，峰值 MFU 从 56% 到 89% 不等。中文原文见知乎 p/13866293937。", "gpu-d2d-pcie": "PCIE 拓扑下两块 GPU 之间的 D2D 拷贝能跑到多少带宽？从 naive cudaMemcpyAsync 一路加到自定义向量化 kernel，最后和 NCCL 对比。中文原文见知乎 p/2847929235。"}' />
