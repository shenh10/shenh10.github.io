---
layout: page
title: 申晗
description: AI 基础设施——大模型推理、深度学习编译器、GPU 性能优化。
---

<ProfileLayout
  name="申晗"
  title="Member of Technical Staff，Moonshot AI"
  photo="https://github.com/shenh10.png"
  interests-label="关注方向"
  :interests="['大模型推理', 'RL 基础设施', 'AI 编译器', 'GPU 性能优化', '推理成本建模']"
  education-label="教育背景"
  :education="[
    { degree: '计算机科学与技术 硕士', school: '清华大学', year: '2018' },
    { degree: '电子工程 工学学士', school: '清华大学', year: '2015' },
  ]"
  :links="[
    { text: '邮箱', href: 'mailto:thushenhan@gmail.com' },
    { text: 'GitHub', href: 'https://github.com/shenh10' },
    { text: 'Google Scholar', href: 'https://scholar.google.com/citations?user=IOYczjUAAAAJ' },
    { text: 'LinkedIn', href: 'https://www.linkedin.com/in/hanshe/' },
    { text: '知乎', href: 'https://www.zhihu.com/people/han-shen-86' },
  ]"
>

## 关于

你好 👋 我在 **Moonshot AI** 做 AI 基础设施——大模型推理、RL 基础设施、深度学习编译器、GPU 性能优化。工作大多落在模型与硬件之间：让大模型服务成本降下来，也让这笔成本在部署之前就能算得准。

来 Moonshot 之前，我在**快手**负责 LLM 推理与 AI 编译团队，工作横跨大模型推理系统与面向推荐场景的编译器。更早在**趋动科技**搭建并领导 AI framework 团队，做 GPU 虚拟化；再往前在**地平线机器人**做计算机视觉，在 **Cisco Systems**（San Jose）做网络软件。清华大学计算机科学与技术硕士、电子工程系工学学士。

最近在做 [HuggingArch](/blog/huggingarch)，一套让推理算账自动化、可验证的 harness——这也是这里的文章大多在算账的原因：一套部署到底要花多少、瓶颈究竟卡在哪、硬件标称的数字离实际可达还有多远。

## 近期动态

- **2026.08** —— 发布 [HuggingArch](/blog/huggingarch)：给一个 HuggingFace 上开源的模型，自动推导出经过校验的架构 spec，再在上面算 KV cache、并行切分与推理吞吐。
- **2026.07** —— **Kimi K3** 发布 —— [技术报告](https://arxiv.org/abs/2607.24653)。
- **2026.06** —— 加入 **Moonshot AI**，Member of Technical Staff。
- **2025.09** —— 上线 [PaperCache](https://www.papercache.org/)，用 LLM 做论文精读而非摘要的博客——[背后的设计原则](/blog/papercache)。
- **2025.03** —— 在 **GTC 2025** 讲 AI 编译器（[S72642](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/)）；发布 [DeepSeek 推理效率分析系列](/blog/ds-inference-1-throughput-ceiling)，率先在业内证明了 DeepSeek 在 H800/H20 上的部署上限。
- **2024.12** —— 写了 [GPU 降频](/blog/gpu-throttling) 与 [PCIE 上的 GPU D2D 拷贝](/blog/gpu-d2d-pcie)。
- **2024** —— 在 **DataFun Summit 2024** 分享推搜广计算引擎优化实践。

## 精选写作

- [DeepSeek V3/R1 推理效率分析](/blog/ds-inference-1-throughput-ceiling)（三篇）—— 只靠论文估出的吞吐上限、对官方 EP144 部署的逐层逆向工程、以及把结论泛化到不同设备数与硬件的模拟器
- [GPU 降频](/blog/gpu-throttling) —— 大矩阵 GEMM 为什么永远跑不到标称算力，在 T4、A10、A800、H800 上实测
- [PCIE 上的 GPU D2D 拷贝](/blog/gpu-d2d-pcie) —— 从 `cudaMemcpyAsync` 到自定义向量化 kernel，并与 NCCL 对比

全部文章见[博客](/zh/blog/)。

## 项目

- **[HuggingArch](/blog/huggingarch)** —— 让 LLM 推理算账自动、可验证、可复用的 harness。
- **[PaperCache](https://www.papercache.org/)** —— 用 LLM 做论文精读的博客，覆盖机器学习系统、大模型与 AI 加速器。
- **[DeepSeek_Simulator](https://github.com/shenh10/DeepSeek_Simulator)** —— 基于 DeepGEMM、FlashMLA 与 torch 的 decode 配置模拟器。

## 技术讲座

- [*Unlocking the Potential of the AI Compiler in Recommendation Systems*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/) [S72642] —— GTC 2025
- *快手推搜广计算引擎优化实践* —— DataFun Summit 2024

## 论文

完整列表见 [Google Scholar](https://scholar.google.com/citations?user=IOYczjUAAAAJ)。

- Tao Hu, Lichao Huang, **Han Shen**. *Multi-object Tracking via End-to-end Tracklet Searching and Ranking.* CoRR abs/2003.02795, 2020.
- Haojie Liu, **Han Shen**, Lichao Huang, Ming Lu, Tong Chen, Zhan Ma. *Learned Video Compression via Joint Spatial-Temporal Correlation Exploration.* AAAI 2020 (Spotlight).
- Tao Hu, Lichao Huang, Xianming Liu, **Han Shen**. *Real Time Visual Tracking using Spatial-Aware Temporal Aggregation Network.* CoRR abs/1908.00692, 2019.
- Hao Luo, Lichao Huang, **Han Shen**, Yuan Li, Chang Huang, Xinggang Wang. *Object Detection in Video with Spatial-Temporal Context Aggregation.* CoRR abs/1907.04988, 2019.
- Qiang Zhou, Zilong Huang, Lichao Huang, Yongchao Gong, **Han Shen**, Chang Huang, Wenyu Liu, Xinggang Wang. *Proposal, Tracking and Segmentation (PTS): A Cascaded Network for Video Object Segmentation.* CVPR Workshop 2019.
- **Han Shen**, Lichao Huang, Chang Huang, Wei Xu. *Tracklet Association Tracker: An End-to-End Learning-based Association Approach for Multi-Object Tracking.* CoRR abs/1808.01562, 2018.

---

欢迎来信 —— [thushenhan@gmail.com](mailto:thushenhan@gmail.com)

</ProfileLayout>
