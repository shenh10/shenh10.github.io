---
layout: home

hero:
  name: "Shen Han"
  text: "AI Infra / 大模型系统"
  tagline: 做高性能训练与推理系统。最近在把「LLM 推理算多少钱」这件事变成自动、可验证、可复用的工具。
  image:
    src: https://github.com/shenh10.png
    alt: Shen Han
  actions:
    - theme: brand
      text: 阅读博客
      link: /zh/blog/
    - theme: alt
      text: 关于我
      link: /zh/about
    - theme: alt
      text: GitHub
      link: https://github.com/shenh10

features:
  - title: HuggingArch
    details: 只要模型在 HuggingFace 上开源，就自动算出它的参数量、KV cache、并行切分与推理吞吐。三层 spec 系统 + sympy 符号代数 + 一套把 agent 钉在 ground truth 上的校验体系。
    link: /zh/blog/huggingarch
  - title: PaperCache
    details: AI 驱动的论文阅读博客，让 LLM 帮你读论文，覆盖深度学习、ML 系统与 AI 加速器。
    link: https://www.papercache.org/
  - title: Claude Code 源码剖析
    details: 对 claude-code v2.1.88 的系统性架构拆解，12 章覆盖 Agent Loop、工具系统、权限引擎与 MCP 集成。
    link: /zh/projects/claude-code/
---

## 最近在做

**[HuggingArch](/zh/blog/huggingarch)** — 推理成本分析的 harness。给一个 HuggingFace model ID，输出经过校验的架构 spec，再在上面算 KV cache、并行切分、prefill/decode 吞吐。架构本身由 agent 编写，但每一步都被 checkpoint 的真实张量、`config.json` 和 forward 源码钉死——消融实验显示，不设这套 guard 时 agent 有很高概率把错的 spec 自判为通过。

**[PaperCache](https://www.papercache.org/)** — 让 LLM 替你读论文。深度学习、ML 系统与 AI 加速器方向的论文精读，持续更新。

## 关注方向

- 大模型训练与后训练的系统优化与工程实践
- GPU 资源高效利用与性能调优（显存 / 带宽 / 并行度 / 算子融合）
- 分布式训练与推理架构（并行策略、通信优化、调度与容错）

## 关于

清华大学电子工程系（本科）与交叉信息研究院（硕士）。先后在地平线机器人、GPU 创业公司与快手做 CV 算法与 AI Infra。完整经历见[关于我](/zh/about)。

[LinkedIn](https://www.linkedin.com/in/hanshe/) · [知乎](https://www.zhihu.com/people/han-shen-86) · [GitHub](https://github.com/shenh10)
