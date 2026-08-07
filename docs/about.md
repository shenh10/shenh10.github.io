---
title: Shen Han — CV
description: AI infrastructure — large-model inference, deep-learning compilers, GPU performance engineering.
---

# Shen Han

AI infrastructure — large-model inference, deep-learning compilers, GPU performance engineering.

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [GitHub](https://github.com/shenh10) · [Zhihu](https://www.zhihu.com/people/han-shen-86) · thushenhan@gmail.com

## Education

| | |
| --- | --- |
| **M.S., Computer Science and Technology** — Tsinghua University | 2015 – 2018 |
| **B.S., Electronic Engineering** — Tsinghua University | 2010 – 2015 |

## Experience

### Moonshot AI <Badge type="tip" text="2026 – present" />

AI infrastructure.

### Kuaishou — Head of LLM Inference & AI Compiler <Badge type="tip" text="2021 – 2026" />

**Framework and compiler optimization for recommendation systems**

- Built the full mixed-precision training stack for TensorFlow — FP16 and BF16 automatic mixed precision — ahead of both the community and the industry at the time. Deployed across Kuaishou's main GPU parts for recommendation and advertising (A800, A10, T4), it improved throughput by close to 50% on typical workloads and retired 600+ GPU machines outright.
- Built **KaiCompiler** from scratch on OpenXLA, decoupling XLA from TensorFlow, with a parallelism-first stream assignment design that improved graph optimization, operator fusion and code generation. 20–30% over TF2 XLA across dozens of internal production models. Shipped to production for internal recommendation and commercial workloads, then platformized as a pluggable training backend with instrumentation and performance collection across the whole model fleet — a production-scale attempt at automated AI compilation for sparse workloads. Presented at DataFun Summit 2024 and GTC 2025.

**LLM inference optimization**

- **Heterogeneous accelerators.** Led development of an AMD-based LLM inference stack, reaching production in late 2023 — among the first in the industry — at MFU parity with A800. Delivered LLAMA 13B, 66B and 175B into service, which made the MI210 Kuaishou's primary inference part, serving dozens of core products. Reaching NVIDIA-comparable hardware performance on MI210 took GEMM tuning, attention kernel optimization, operator fusion, multi-level topology optimization for AllReduce, compute-communication overlap and an INT8 KV cache, and produced a positive TCO across every scenario internally.
- **Inference frontier.** Organized the team's optimization work around MFU and MBU as the standing objective: paged-attention KV memory reservation, prefix caching and offloading, long context, prefill-decode disaggregation, Nanoflow, and CPU/GPU hybrid inference — some shipped, some as research.

**Community.** Repeat speaker and host on the DataFun platform; speaker on AI compilers at GTC 2025. Occasional writing on GPU optimization and LLM inference — including the [DeepSeek V3/R1 inference-performance series](/blog/ds-inference-1-throughput-ceiling), the first public work to establish the deployment ceilings for DeepSeek on H800 and H20.

### VirtAI Tech — Senior Software Engineer → Engineering Manager <Badge type="tip" text="2020 – 2021" />

- Built and led the AI framework team, taking the company's flagship GPU virtualization product **OrionX** through framework adaptation and optimization, working closely with sales and delivery to get it into customer environments.
- Delivered the industry's first live-migration scheme for virtual GPUs, supporting both PyTorch and TensorFlow; framework-level performance optimization on the virtualization layer; a model zoo and benchmark suite for OrionX.
- **vGPU communication optimization.** Remote-invoked vGPUs suffered heavy overhead in small-batch inference, particularly over TCP. Having traced the effect to a linear relationship between RPC cost and CUDA call frequency, designed and built a batching RPC transport and optimized the high-frequency APIs of PyTorch and TensorFlow — cutting overhead by more than two thirds under PyTorch and more than half under TensorFlow.
- **Distributed performance.** Built the first mathematical model of multi-machine vGPU aggregation, identifying how per-machine overhead limits scale-out; quantified it in simulation and proposed asynchronous data prefetch with remote dynamic caching. In low-bandwidth mode the result exceeded physical-GPU performance without prefetch — 10–30% on ResNet50.
- Received the R&D organization's annual award for best performance in 2021, and was the first in R&D to be named a "VirtAI Star."

### Horizon Robotics — Vision Algorithm Intern → Vision Algorithm Engineer <Badge type="tip" text="2017 – 2020" />

- Worked across perception: single- and multi-object tracking, image and video object detection, action recognition. Supervised interns through research and publication.
- **Detection.** Owned tuning and iteration of the pedestrian detection model for the autonomous driving group; built and maintained the video-detection library supporting multi-frame fusion and frame skipping.
- **Multi-object tracking.** Proposed an end-to-end data-association method based on constrained integer programming: high-confidence features link detections into tracklets, then a network-flow method links tracklets into long trajectories. State of the art on both MOT16 and MOT17.

### Cisco Systems, San Jose — Software Engineering Intern → Software Engineer (part-time) <Badge type="tip" text="2013 – 2015" />

- **onePK MUX.** Built a proxy system for Cisco switches. A switch supports only 20–30 connections; interposing a Linux server as a buffer, with message queues for queuing and dispatch, allowed tens of thousands of concurrent requests. Supporting the onePK SDK also required parsing and re-encapsulating requests over Thrift, its multi-language binary protocol.
- **Plug-and-Play.** Implemented the web front end and the front-to-back interaction for the PnP protocol, letting users configure routers and switches through a GUI. Demoed at Cisco Live 2014.
- **CFEngine.** Extended the cluster-management project to support Cisco's onePK SDK.
- **OpenDaylight.** After returning to China, continued part-time with the same group, building the PnP plugin on OpenDaylight — then the mainstream open-source SDN framework — and contributing to its design and implementation across two releases, Hydrogen and Helium.

## Talks

- *Unlocking the Potential of the AI Compiler in Recommendation Systems* [S72642] — GTC 2025
- *Compute-engine optimization for recommendation, search and advertising at Kuaishou* — DataFun Summit 2024

## Publications

- Tao Hu, Lichao Huang, **Han Shen**. *Multi-object Tracking via End-to-end Tracklet Searching and Ranking.* CoRR abs/2003.02795, 2020.
- Haojie Liu, **Han Shen**, Lichao Huang, Ming Lu, Tong Chen, Zhan Ma. *Learned Video Compression via Joint Spatial-Temporal Correlation Exploration.* AAAI 2020 (Spotlight).
- Tao Hu, Lichao Huang, Xianming Liu, **Han Shen**. *Real Time Visual Tracking using Spatial-Aware Temporal Aggregation Network.* CoRR abs/1908.00692, 2019.
- Hao Luo, Lichao Huang, **Han Shen**, Yuan Li, Chang Huang, Xinggang Wang. *Object Detection in Video with Spatial-Temporal Context Aggregation.* CoRR abs/1907.04988, 2019.
- Qiang Zhou, Zilong Huang, Lichao Huang, Yongchao Gong, **Han Shen**, Chang Huang, Wenyu Liu, Xinggang Wang. *Proposal, Tracking and Segmentation (PTS): A Cascaded Network for Video Object Segmentation.* CVPR Workshop 2019.
- **Han Shen**, Lichao Huang, Chang Huang, Wei Xu. *Tracklet Association Tracker: An End-to-End Learning-based Association Approach for Multi-Object Tracking.* CoRR abs/1808.01562, 2018.

## Projects

- **[HuggingArch](/blog/huggingarch)** — a harness making LLM inference cost analysis automatic, verifiable and reusable for any model open on HuggingFace.
- **[DeepSeek_Simulator](https://github.com/shenh10/DeepSeek_Simulator)** — a decode-configuration simulator over DeepGEMM, FlashMLA and torch, sweeping DP-EP and TP-DP-EP across hardware.
- **[PaperCache](https://www.papercache.org/)** — an AI-driven paper-reading blog covering deep learning, ML systems and AI accelerators.
