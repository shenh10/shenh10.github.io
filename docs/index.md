---
layout: home

hero:
  name: "Shen Han"
  text: "AI Infra / Large-Model Systems"
  tagline: I build high-performance training and inference systems. Lately I've been making "what does this model cost to serve" an automatic, verifiable, reusable answer.
  image:
    src: https://github.com/shenh10.png
    alt: Shen Han
  actions:
    - theme: brand
      text: Read the blog
      link: /blog/
    - theme: alt
      text: About me
      link: /about
    - theme: alt
      text: GitHub
      link: https://github.com/shenh10

features:
  - title: HuggingArch
    details: Give it a model that's open on HuggingFace and it works out the parameter count, KV cache, parallel sharding and inference throughput. A three-tier spec system, sympy symbolic algebra end to end, and a validation harness that pins the agent to ground truth.
    link: /blog/huggingarch
  - title: PaperCache
    details: An AI-driven paper-reading blog — let an LLM read the papers for you. Deep learning, ML systems, and AI accelerators.
    link: https://www.papercache.org/
  - title: Claude Code Teardown
    details: A systematic architectural teardown of claude-code v2.1.88 across 12 chapters — agent loop, tool system, permission engine, MCP integration. Written in Chinese.
    link: /projects/claude-code/
---

## What I'm working on

**[HuggingArch](/blog/huggingarch)** — a harness for inference cost analysis. Hand it a HuggingFace model ID and it produces a validated architecture spec, then computes KV cache, parallel sharding, and prefill/decode throughput on top of it. The architecture itself is written by an agent, but every step is anchored to the checkpoint's real tensors, its `config.json`, and its forward source — an ablation shows that without those guards, an agent will confidently declare a broken spec valid.

**[PaperCache](https://www.papercache.org/)** — let an LLM read the papers for you. Close readings in deep learning, ML systems, and AI accelerators, updated continuously.

## Interests

- Systems optimization and engineering practice for large-model pretraining and post-training
- Efficient GPU utilization and performance tuning (memory, bandwidth, parallelism, operator fusion)
- Distributed training and inference architecture (parallelism strategies, communication optimization, scheduling, fault tolerance)

## About

Tsinghua University — Electronic Engineering (BSc) and the Institute for Interdisciplinary Information Sciences (MSc). Since then: computer vision at Horizon Robotics, then AI infrastructure at a GPU startup and at Kuaishou. Full background on the [about page](/about).

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [Zhihu](https://www.zhihu.com/people/han-shen-86) · [GitHub](https://github.com/shenh10)
