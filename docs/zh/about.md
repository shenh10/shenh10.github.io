---
title: 申晗 — 简历
description: AI 基础设施——大模型推理、深度学习编译器、GPU 性能优化。
---

# 申晗

AI 基础设施——大模型推理、深度学习编译器、GPU 性能优化。

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [GitHub](https://github.com/shenh10) · [知乎](https://www.zhihu.com/people/han-shen-86) · thushenhan@gmail.com

## 教育背景

| | |
| --- | --- |
| **清华大学，计算机科学与技术，硕士** | 2015 – 2018 |
| **清华大学，电子工程系，工学学士** | 2010 – 2015 |

## 工作经历

### 快手 —— LLM 推理 & AI 编译团队负责人 <Badge type="tip" text="2021 – 2025" />

**推荐系统的框架及编译优化**

- 开发了 TensorFlow 下的全系列半精度训练技术（FP16、BF16 自动混合精度训练），相关技术领先社区和业界。覆盖推荐广告多个主流 GPU 卡型（A800/A10/T4），在典型业务场景上优化近 50% 计算吞吐，一次性节省 600+ 台 GPU 机器并广泛落地。
- 从 0 到 1 搭建基于 OpenXLA 的独立编译器 **KaiCompiler**，实现 XLA 与 TensorFlow 的解耦合，并创新地设计了并行优先的流分配技术，优化 XLA 内部的图优化、算子融合及代码生成表现，在内部数十个典型模型上比 TF2 XLA 高出 20%–30%。已在内部推荐及商业化业务生产上线，并作为训练框架的可插拔后端平台化，对全域模型进行埋点与性能数据采集，在生产级进行了稀疏场景大范围 AI 编译自动化的尝试。相关工作在 DataFun Summit 2024、GTC 2025 分享。

**大语言模型推理优化**

- **异构方向。** 带领团队完成基于 AMD 的大模型推理框架研发，2023 年底在业界第一批将 AMD GPU 落地生产，性能与 A800 MFU 对标。完成 LLAMA 13B、66B、175B 的推理方案上线与交付，直接推动 MI210 成为快手内部主力推理卡型，承担数十个核心业务的线上服务。通过 Gemm 算子调优、Attention 算子优化、算子融合、AllReduce 多级网络拓扑优化、通信计算 overlap、int8 kvcache 等手段，实现了 MI210 上与 NV GPU 对标的硬件性能，在内部实现全场景正向的 TCO 收益。
- **泛 LLM 推理前沿。** 以 MFU/MBU 的提升为长期目标组织团队优化工作，包括 kvcache paged attention 显存预留策略、Prefix Caching & Offloading、长文本、PD 分离、Nanoflow、CPU/GPU 混合推理等（部分预研，部分落地）。

**行业影响力。** 多次作为 DataFun 平台讲师、主持人进行技术分享；GTC 2025 AI Compiler 主题 talk 主讲。在知乎分享 GPU 优化与大模型推理，其中 [DeepSeek V3/R1 推理性能分析系列](/blog/ds-inference-1-throughput-ceiling) 率先在业内证明了 DeepSeek 在 H800/H20 部署方案的上限；知势榜科技互联网领域上榜答主。

### 趋动科技 —— 高级软件开发工程师 → 研发经理 <Badge type="tip" text="2020 – 2021" />

- 搭建并领导 AI framework 团队，完成 AI framework 在公司主产品 GPU 虚拟化软件 **OrionX** 上的适配与优化，并与销售交付团队紧密配合，保障 OrionX 在客户场景的顺利交付。
- 业界率先开发出基于虚拟 GPU 的热迁移方案（支持 PyTorch 与 TensorFlow）；AI framework 在虚拟化软件上的性能优化；为 OrionX 提供 model zoo 与基准性能测试集。
- **vGPU 通信优化。** 远程调用的 vGPU 在小 batch 推理场景面临较严重的性能损耗，TCP 协议下尤为突出。分析出 RPC 性能与 CUDA 调用频率间的线性关系后，设计并开发了一套基于批处理的 RPC 通信机制，并优化 PyTorch 与 TensorFlow 的高频 API，最终在 PyTorch 下将损耗减少 2/3 以上，TensorFlow 下减少 1/2 以上。
- **分布式性能优化。** 首次建立 vGPU 多机聚合的数学模型，指出随机器增多带来的性能损耗将影响软件 scale out；仿真量化了损耗大小，提出数据异步预取与远程动态缓存的方案。最终方案在低带宽模式下甚至超越无预取的物理 GPU 性能（ResNet50 上提升 10%–30%）。
- 获 2021 年全公司研发部门年度最佳业绩表现，首位在研发部门被授予"趋动之星"。

### 地平线机器人 —— 视觉算法实习生 → 视觉算法工程师 <Badge type="tip" text="2017 – 2020" />

- 覆盖计算机视觉感知各方向：单/多目标追踪、图像/视频目标检测、动作识别。指导实习生完成预研并撰写论文。
- **图片与视频检测。** 负责自动驾驶部门行人检测模型的 tuning 与迭代；搭建并维护兼容多帧融合与跳帧检测的 video detection 基础库。
- **多目标追踪。** 提出一个端到端的、基于带约束整数规划的数据关联方法：先用高置信度特征将检测框连成 tracklet，再基于网络流方法将 tracklet 连成长轨迹。在 MOT16 与 MOT17 上均取得 SOTA。

### Cisco Systems（San Jose）—— 软件开发实习生 → 软件开发工程师（兼职） <Badge type="tip" text="2013 – 2015" />

- **onePK MUX。** 实现服务于 Cisco 交换机设备的 proxy 系统。交换机能建立的连接数非常有限（20–30 个），通过 Linux 服务器做一层 buffer、以 message queue 实现排队与消息分发，使其支持上万并发请求。因支持的是 onePK SDK，还需对基于多语言二进制协议 Thrift 的请求进行解析与再封装转发。
- **Plug-and-Play。** 为 PnP 通信协议实现前端网页接口与前后端交互逻辑，使用户可通过 GUI 配置管理路由与交换机设备。项目 demo 在 Cisco Live 2014 展示。
- **CFEngine。** 对集群管理开源软件进行二次开发，使其支持 onePK SDK 接口。
- **OpenDaylight。** 回国后继续兼职为原部门远程工作，在当时 SDN 领域主流的开源框架 OpenDaylight 上实现 PnP 插件，并参与 Hydrogen 与 Helium 两代 release 的设计与实现。

## 技术讲座

- *Unlocking the Potential of the AI Compiler in Recommendation Systems* [S72642] —— GTC 2025
- *快手推搜广计算引擎优化实践* —— DataFun Summit 2024

## 论文

- Tao Hu, Lichao Huang, **Han Shen**. *Multi-object Tracking via End-to-end Tracklet Searching and Ranking.* CoRR abs/2003.02795, 2020.
- Haojie Liu, **Han Shen**, Lichao Huang, Ming Lu, Tong Chen, Zhan Ma. *Learned Video Compression via Joint Spatial-Temporal Correlation Exploration.* AAAI 2020 (Spotlight).
- Tao Hu, Lichao Huang, Xianming Liu, **Han Shen**. *Real Time Visual Tracking using Spatial-Aware Temporal Aggregation Network.* CoRR abs/1908.00692, 2019.
- Hao Luo, Lichao Huang, **Han Shen**, Yuan Li, Chang Huang, Xinggang Wang. *Object Detection in Video with Spatial-Temporal Context Aggregation.* CoRR abs/1907.04988, 2019.
- Qiang Zhou, Zilong Huang, Lichao Huang, Yongchao Gong, **Han Shen**, Chang Huang, Wenyu Liu, Xinggang Wang. *Proposal, Tracking and Segmentation (PTS): A Cascaded Network for Video Object Segmentation.* CVPR Workshop 2019.
- **Han Shen**, Lichao Huang, Chang Huang, Wei Xu. *Tracklet Association Tracker: An End-to-End Learning-based Association Approach for Multi-Object Tracking.* CoRR abs/1808.01562, 2018.

## 项目

- **[HuggingArch](/blog/huggingarch)** —— 让 LLM 推理算账自动、可验证、可复用的 harness，适用于任何在 HuggingFace 上开源的模型。
- **[DeepSeek_Simulator](https://github.com/shenh10/DeepSeek_Simulator)** —— 基于 DeepGEMM、FlashMLA 与 torch 的 decode 配置模拟器，跨硬件扫 DP-EP 与 TP-DP-EP。
- **[PaperCache](https://www.papercache.org/)** —— AI 驱动的论文阅读博客，覆盖深度学习、ML 系统与 AI 加速器。
