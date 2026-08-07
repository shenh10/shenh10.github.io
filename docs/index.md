---
layout: home

hero:
  name: "Shen Han"
  text: "LLM Inference · AI Compilers · GPU Performance"
  tagline: I build the infrastructure that makes large models cheap to serve — and the tools that tell you, before you deploy, what serving them will cost.
  image:
    src: https://github.com/shenh10.png
    alt: Shen Han
  actions:
    - theme: brand
      text: Writing
      link: /blog/
    - theme: alt
      text: Full CV
      link: /about
    - theme: alt
      text: GitHub
      link: https://github.com/shenh10
---

## About

I work on AI infrastructure at **Moonshot AI**: large-model inference, deep-learning compilers, and GPU performance engineering.

Before that I led the LLM inference and AI compiler team at Kuaishou. On the compiler side that produced **KaiCompiler**, an OpenXLA-based compiler built from scratch and decoupled from TensorFlow, whose parallelism-first stream assignment brought 20–30% over TF2 XLA across dozens of production models; it runs in production for recommendation and advertising, and is being platformized as a pluggable training backend. Earlier on the same team, a full mixed-precision training stack — FP16 and BF16 automatic mixed precision for TensorFlow — improved throughput by close to 50% on typical workloads and retired 600+ GPU machines outright.

On the inference side, the team delivered one of the industry's first production LLM deployments on AMD GPUs, at MFU parity with A800 — LLAMA 13B, 66B and 175B in service, which made the MI210 Kuaishou's primary inference part across dozens of core products. Getting there meant GEMM and attention kernel work, operator fusion, topology-aware AllReduce, compute-communication overlap, and an INT8 KV cache. Above the kernels, the team's optimization work was organized around MFU and MBU as the objective: paged-attention KV reservation, prefix caching and offloading, long context, prefill-decode disaggregation, and CPU/GPU hybrid inference.

Before Kuaishou I built and led the AI framework team at VirtAI Tech, working on GPU virtualization (OrionX) — including the first live-migration scheme for virtual GPUs, and a batched-RPC transport that cut small-batch remote-inference overhead by two thirds under PyTorch. Earlier still I worked on computer vision at Horizon Robotics, on multi-object tracking and video detection, and on SDN and switch tooling at Cisco Systems in San Jose.

I hold an M.S. in Computer Science and a B.S. in Electronic Engineering, both from Tsinghua University.

## Research Interests

- **Large-model inference** — MFU/MBU-driven optimization, KV cache and paged attention, prefix caching and offloading, prefill-decode disaggregation, long context
- **AI compilers** — XLA/OpenXLA, graph optimization, operator fusion, code generation, stream assignment for parallelism
- **GPU performance engineering** — mixed precision, GEMM and attention kernels, collective communication and topology, heterogeneous accelerators
- **Cost modelling for inference** — making the arithmetic of deployment automatic, verifiable, and reusable

## Current Work

**[HuggingArch](/blog/huggingarch)** — a harness for inference cost analysis. Give it a model that is open on HuggingFace and it produces a validated architecture spec, then computes KV cache, parallel sharding, and prefill/decode throughput on top of it. The spec is written by an agent but anchored at every step to the checkpoint's real tensors, its `config.json`, and its forward source.

**[PaperCache](https://www.papercache.org/)** — an AI-driven paper-reading blog covering deep learning, ML systems, and AI accelerators.

## Selected Writing

- [DeepSeek V3/R1 inference efficiency](/blog/ds-inference-1-throughput-ceiling), in three parts — a throughput ceiling from the paper alone, a layer-by-layer reverse-engineering of the published EP144 deployment, and a simulator generalizing it across device counts and hardware
- [GPU clock throttling](/blog/gpu-throttling) — why a large GEMM never reaches a card's rated TFLOPS, measured across T4, A10, A800 and H800
- [GPU-to-GPU copy over PCIe](/blog/gpu-d2d-pcie) — from `cudaMemcpyAsync` to a custom vectorized kernel, against NCCL

## Talks

- *Unlocking the Potential of the AI Compiler in Recommendation Systems* [S72642] — GTC 2025
- *Compute-engine optimization for recommendation, search and advertising at Kuaishou* — DataFun Summit 2024

## Publications

Six papers in computer vision — multi-object tracking, video object detection, learned video compression — including an AAAI 2020 Spotlight. Listed in full on the [CV](/about#publications).

---

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [Zhihu](https://www.zhihu.com/people/han-shen-86) · [GitHub](https://github.com/shenh10) · thushenhan@gmail.com
