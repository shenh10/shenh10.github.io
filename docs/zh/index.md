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
      text: 关于
      link: /zh/about
    - theme: alt
      text: GitHub
      link: https://github.com/shenh10
---

## 关于

你好 👋 我在 **Moonshot AI** 做 AI 基础设施——大模型推理、深度学习编译器、GPU 性能优化。工作大多落在模型与硬件之间：让大模型服务成本降下来，也让这笔成本在部署之前就能算得准。

来 Moonshot 之前，我在**快手**负责 LLM 推理与 AI 编译团队，工作横跨大模型推理系统与面向推荐场景的编译器。更早在**趋动科技**搭建并领导 AI framework 团队，做 GPU 虚拟化；再往前在**地平线机器人**做计算机视觉，在 **Cisco Systems**（San Jose）做网络软件。清华大学计算机科学与技术硕士、电子工程系工学学士。

最近在做 [HuggingArch](/blog/huggingarch)，一套让推理算账自动化、可验证的 harness——这也是这里的文章大多在算账的原因：一套部署到底要花多少、瓶颈究竟卡在哪、硬件标称的数字离实际可达还有多远。

## 近期动态

- **2026.06** —— 加入 **Moonshot AI**，做 AI 基础设施。
- **2026.08** —— 发布 [HuggingArch](/blog/huggingarch)：给一个 HuggingFace 上开源的模型，自动推导出经过校验的架构 spec，再在上面算 KV cache、并行切分与推理吞吐。
- **2025.08** —— DeepSeek 推理系列[第三篇](/blog/ds-inference-3-decode-generalization)：用模拟器在 H800 与 H20 上扫 decode 配置，说明官方的 EP144 并非唯一解。
- **2025.03** —— 在 **GTC 2025** 讲 AI 编译器（[S72642](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/)）；发布 DeepSeek 推理效率分析[第一篇](/blog/ds-inference-1-throughput-ceiling)与[第二篇](/blog/ds-inference-2-reverse-engineering)，率先在业内证明了 DeepSeek 在 H800/H20 上的部署上限。
- **2024.12** —— 写了 [GPU 降频](/blog/gpu-throttling) 与 [PCIE 上的 GPU D2D 拷贝](/blog/gpu-d2d-pcie)。
- **2024** —— 在 **DataFun Summit 2024** 分享推搜广计算引擎优化实践。

## 关注方向

- **大模型推理** —— 以 MFU/MBU 为目标的优化、KV cache 与 paged attention、prefix caching 与 offloading、PD 分离、长文本
- **AI 编译器** —— XLA/OpenXLA、图优化、算子融合、代码生成、面向并行的流分配
- **GPU 性能优化** —— 混合精度、GEMM 与 attention 算子、集合通信与拓扑、异构加速器
- **推理成本建模** —— 让部署这笔账变得自动、可验证、可复用

## 精选写作

- [DeepSeek V3/R1 推理效率分析](/blog/ds-inference-1-throughput-ceiling)（三篇）—— 只靠论文估出的吞吐上限、对官方 EP144 部署的逐层逆向工程、以及把结论泛化到不同设备数与硬件的模拟器
- [GPU 降频](/blog/gpu-throttling) —— 大矩阵 GEMM 为什么永远跑不到标称算力，在 T4、A10、A800、H800 上实测
- [PCIE 上的 GPU D2D 拷贝](/blog/gpu-d2d-pcie) —— 从 `cudaMemcpyAsync` 到自定义向量化 kernel，并与 NCCL 对比

全部文章见[博客](/zh/blog/)，背景与论文见[关于](/zh/about)。

## 技术讲座

- [*Unlocking the Potential of the AI Compiler in Recommendation Systems*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/) [S72642] —— GTC 2025
- *快手推搜广计算引擎优化实践* —— DataFun Summit 2024

---

欢迎来信 —— [thushenhan@gmail.com](mailto:thushenhan@gmail.com)

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [知乎](https://www.zhihu.com/people/han-shen-86) · [GitHub](https://github.com/shenh10)
