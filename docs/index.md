---
layout: page
title: Shen Han
description: AI infrastructure — large-model inference, deep-learning compilers, GPU performance engineering.
---

<ProfileLayout
  name="Shen Han"
  title="AI Infrastructure at Moonshot AI"
  photo="https://github.com/shenh10.png"
  :interests="['LLM Inference', 'AI Compilers', 'GPU Performance', 'Cost Modelling']"
  :education="[
    { degree: 'M.S. in Computer Science', school: 'Tsinghua University', year: '2018' },
    { degree: 'B.S. in Electronic Engineering', school: 'Tsinghua University', year: '2015' },
  ]"
  :links="[
    { text: 'Email', href: 'mailto:thushenhan@gmail.com' },
    { text: 'GitHub', href: 'https://github.com/shenh10' },
    { text: 'LinkedIn', href: 'https://www.linkedin.com/in/hanshe/' },
    { text: 'Zhihu', href: 'https://www.zhihu.com/people/han-shen-86' },
  ]"
>

## About

Hi 👋 I work on AI infrastructure at **Moonshot AI** — large-model inference, deep-learning compilers, and GPU performance engineering. Most of what I do sits between the model and the hardware: making large models cheaper to serve, and making the cost of serving them knowable before anything is deployed.

Before Moonshot I led the LLM inference and AI compiler team at **Kuaishou**, where the work spanned inference systems for large models and a compiler for recommendation workloads. Earlier I led the AI framework team at **VirtAI Tech**, working on GPU virtualization; before that, computer vision at **Horizon Robotics**, and networking software at **Cisco Systems** in San Jose. I hold an M.S. in Computer Science and a B.S. in Electronic Engineering, both from **Tsinghua University**.

Lately I have been building [HuggingArch](/blog/huggingarch), a harness that makes the arithmetic of inference cost automatic and verifiable — which is also why most of what I write here is arithmetic: what a deployment actually costs, where the bound really is, and how far a card's rated numbers are from what you can reach.

## Latest News

- **2026.06** — Joined **Moonshot AI**, working on AI infrastructure.
- **2026.08** — Released [HuggingArch](/blog/huggingarch): give it any model open on HuggingFace and it derives a validated architecture spec, then computes KV cache, parallel sharding and inference throughput on top of it.
- **2025.08** — [Part 3](/blog/ds-inference-3-decode-generalization) of the DeepSeek inference series: a simulator sweeping decode configurations across H800 and H20, showing DeepSeek's published EP144 is not the only good answer.
- **2025.03** — Spoke on AI compilers at **GTC 2025** ([S72642](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/)), and published [parts 1](/blog/ds-inference-1-throughput-ceiling) and [2](/blog/ds-inference-2-reverse-engineering) of the DeepSeek inference-efficiency series — the first public work to establish DeepSeek's deployment ceilings on H800 and H20.
- **2024.12** — Wrote up [GPU clock throttling](/blog/gpu-throttling) and [GPU-to-GPU copy over PCIe](/blog/gpu-d2d-pcie).
- **2024** — Spoke at **DataFun Summit 2024** on compute-engine optimization for recommendation, search and advertising.

## Selected Writing

- [DeepSeek V3/R1 inference efficiency](/blog/ds-inference-1-throughput-ceiling), in three parts — a throughput ceiling from the paper alone, a layer-by-layer reverse-engineering of the published EP144 deployment, and a simulator generalizing it across device counts and hardware
- [GPU clock throttling](/blog/gpu-throttling) — why a large GEMM never reaches a card's rated TFLOPS, measured across T4, A10, A800 and H800
- [GPU-to-GPU copy over PCIe](/blog/gpu-d2d-pcie) — from `cudaMemcpyAsync` to a custom vectorized kernel, against NCCL

All posts are on the [blog](/blog/); background and publications are on the [about page](/about).

## Talks

- [*Unlocking the Potential of the AI Compiler in Recommendation Systems*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72642/) [S72642] — GTC 2025
- *Compute-engine optimization for recommendation, search and advertising at Kuaishou* — DataFun Summit 2024 <Badge type="info" text="中文" />

---

Feel free to drop me a line — [thushenhan@gmail.com](mailto:thushenhan@gmail.com)


</ProfileLayout>
