---
layout: home

hero:
  name: "Shen Han"
  tagline: AI infrastructure at Moonshot AI — large-model inference, deep-learning compilers, GPU performance engineering.
  image:
    src: https://github.com/shenh10.png
    alt: Shen Han
  actions:
    - theme: brand
      text: Writing
      link: /blog/
    - theme: alt
      text: About
      link: /about
    - theme: alt
      text: GitHub
      link: https://github.com/shenh10
---

## About

I work on AI infrastructure at Moonshot AI — large-model inference, deep-learning compilers, and GPU performance engineering.

Before that I led the LLM inference and AI compiler team at Kuaishou, where the work spanned inference systems for large models and a compiler for recommendation workloads. Earlier I led the AI framework team at VirtAI Tech, working on GPU virtualization; before that, computer vision at Horizon Robotics, and networking software at Cisco Systems in San Jose.

I hold an M.S. in Computer Science and a B.S. in Electronic Engineering, both from Tsinghua University.

## Research Interests

- **Large-model inference** — MFU/MBU-driven optimization, KV cache and paged attention, prefix caching and offloading, prefill-decode disaggregation, long context
- **AI compilers** — XLA/OpenXLA, graph optimization, operator fusion, code generation, stream assignment for parallelism
- **GPU performance engineering** — mixed precision, GEMM and attention kernels, collective communication and topology, heterogeneous accelerators
- **Cost modelling for inference** — making the arithmetic of deployment automatic, verifiable, and reusable

## Current Work

**[HuggingArch](/blog/huggingarch)** — a harness for inference cost analysis. Give it a model that is open on HuggingFace and it produces a validated architecture spec, then computes KV cache, parallel sharding, and prefill/decode throughput on top of it.

**[PaperCache](https://www.papercache.org/)** — an AI-driven paper-reading blog covering deep learning, ML systems, and AI accelerators.

## Selected Writing

- [DeepSeek V3/R1 inference efficiency](/blog/ds-inference-1-throughput-ceiling), in three parts — a throughput ceiling from the paper alone, a layer-by-layer reverse-engineering of the published EP144 deployment, and a simulator generalizing it across device counts and hardware
- [GPU clock throttling](/blog/gpu-throttling) — why a large GEMM never reaches a card's rated TFLOPS, measured across T4, A10, A800 and H800
- [GPU-to-GPU copy over PCIe](/blog/gpu-d2d-pcie) — from `cudaMemcpyAsync` to a custom vectorized kernel, against NCCL

## Talks

- [*Unlocking the Potential of the AI Compiler in Recommendation Systems*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/) [S72642] — GTC 2025
- *Compute-engine optimization for recommendation, search and advertising at Kuaishou* — DataFun Summit 2024 <Badge type="info" text="中文" />

---

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [Zhihu](https://www.zhihu.com/people/han-shen-86) · [GitHub](https://github.com/shenh10) · thushenhan@gmail.com
