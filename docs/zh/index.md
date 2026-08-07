---
layout: home

hero:
  name: "申晗"
  tagline: Moonshot AI，AI 基础设施 —— 大模型推理、深度学习编译器、GPU 性能优化。
  image:
    src: https://github.com/shenh10.png
    alt: 申晗
  actions:
    - theme: brand
      text: 文章
      link: /zh/blog/
    - theme: alt
      text: 完整简历
      link: /zh/about
    - theme: alt
      text: GitHub
      link: https://github.com/shenh10
---

## 关于

我在 **Moonshot AI** 做 AI 基础设施：大模型推理、深度学习编译器、GPU 性能优化。

此前在快手负责 LLM 推理与 AI 编译团队。编译方向做出了 **KaiCompiler**——从 0 到 1 基于 OpenXLA 搭建、与 TensorFlow 解耦的独立编译器，并行优先的流分配设计使其在数十个生产模型上比 TF2 XLA 高出 20%–30%，已在推荐与商业化业务生产上线，并作为可插拔训练后端平台化。更早在同一团队做的 TensorFlow 全系列半精度训练技术（FP16/BF16 自动混合精度），在典型业务场景上提升近 50% 计算吞吐，一次性节省 600+ 台 GPU 机器。

推理方向，团队在 2023 年底完成业界第一批 AMD GPU 上的生产级大模型推理落地，性能与 A800 MFU 对标；LLAMA 13B/66B/175B 上线交付，直接推动 MI210 成为快手主力推理卡型，承载数十个核心业务。这背后是 Gemm 调优、Attention 算子优化、算子融合、AllReduce 多级拓扑优化、通信计算 overlap 与 int8 kvcache。算子之上，团队的优化工作以 MFU/MBU 为长期目标组织：paged attention 显存预留、Prefix Caching & Offloading、长文本、PD 分离、CPU/GPU 混合推理。

快手之前，我在趋动科技搭建并领导 AI framework 团队，围绕 GPU 虚拟化产品 OrionX 工作——包括业界首个 vGPU 热迁移方案，以及一套批处理 RPC 通信机制，把 PyTorch 下小 batch 远程推理的损耗削掉三分之二以上。更早在地平线机器人做计算机视觉，方向是多目标追踪与视频检测；再往前在 Cisco Systems（San Jose）做 SDN 与交换机工具链。

清华大学计算机科学与技术硕士、电子工程系工学学士。

## 关注方向

- **大模型推理** —— 以 MFU/MBU 为目标的优化、KV cache 与 paged attention、prefix caching 与 offloading、PD 分离、长文本
- **AI 编译器** —— XLA/OpenXLA、图优化、算子融合、代码生成、面向并行的流分配
- **GPU 性能优化** —— 混合精度、GEMM 与 attention 算子、集合通信与拓扑、异构加速器
- **推理成本建模** —— 让部署这笔账变得自动、可验证、可复用

## 在做什么

**[HuggingArch](/blog/huggingarch)** —— 推理成本分析的 harness。给一个 HuggingFace 上开源的模型，产出经过校验的架构 spec，再在上面算 KV cache、并行切分与 prefill/decode 吞吐。spec 由 agent 编写，但每一步都被 checkpoint 的真实张量、`config.json` 与 forward 源码钉死。

**[PaperCache](https://www.papercache.org/)** —— AI 驱动的论文阅读博客，覆盖深度学习、ML 系统与 AI 加速器。

## 精选写作

- [DeepSeek V3/R1 推理效率分析](/blog/ds-inference-1-throughput-ceiling)（三篇）—— 只靠论文估出的吞吐上限、对官方 EP144 部署的逐层逆向工程、以及把结论泛化到不同设备数与硬件的模拟器
- [GPU 降频](/blog/gpu-throttling) —— 大矩阵 GEMM 为什么永远跑不到标称算力，在 T4、A10、A800、H800 上实测
- [PCIE 上的 GPU D2D 拷贝](/blog/gpu-d2d-pcie) —— 从 `cudaMemcpyAsync` 到自定义向量化 kernel，并与 NCCL 对比

## 技术讲座

- *Unlocking the Potential of the AI Compiler in Recommendation Systems* [S72642] —— GTC 2025
- *快手推搜广计算引擎优化实践* —— DataFun Summit 2024

## 论文

计算机视觉方向 6 篇，涵盖多目标追踪、视频目标检测与学习式视频压缩，含 AAAI 2020 Spotlight。完整列表见[简历](/zh/about#论文)。

---

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [知乎](https://www.zhihu.com/people/han-shen-86) · [GitHub](https://github.com/shenh10) · thushenhan@gmail.com
