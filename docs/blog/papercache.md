---
date: 2025-09-05
title: "PaperCache: Close Reading of Papers with an LLM"
short: "PaperCache"
titleZh: "PaperCache：用 LLM 精读论文"
description: A paper-reading blog built on LLM close reads rather than summaries — why existing tools did not fit, and the seven design principles behind it.
---

# PaperCache: Close Reading of Papers with an LLM

> Originally published in Chinese on [Zhihu](https://zhuanlan.zhihu.com/p/1945260580274443582), September 5, 2025.

**[papercache.org](https://www.papercache.org/)** — over a hundred papers read closely with an LLM, published as a blog. It covers three areas on an ongoing basis:

- Machine learning systems
- Large language models and diffusion models
- AI accelerators

![](/blog/papercache/fig01.jpg)

## Why

The immediate cause was a backlog: a large stack of papers to get through. I read slowly, and I read while thinking — skimming feels risky, because in an unfamiliar area the important references and the chain of reasoning in a key paper are exactly what you cannot afford to lose.

Having an LLM read papers is not a new idea. [papers.cool](https://papers.cool/) and [PaperScope](https://www.paperscope.ai/) are both good. But the first suits breadth-first skimming and does not surface a paper's core method, and the second — genuinely well built, and better looking — is oriented toward algorithms rather than the infrastructure work I care about, at a coarser summary granularity than I want.

So I started with some prompt engineering aimed at my own reading needs. Several posts in my paper-reading column were in fact produced this way. Pulling figures out by hand was tedious enough that I built a small tool for it; and since an LLM cannot be relied on to be error-free, some manual work remains regardless. Rather than let that effort go nowhere, it became PaperCache.

The premise underneath all of it: in an era of capable language models, reading should work differently. Using native English fluency to read faster, and assembling notes worth returning to, should stop being a high-effort skill.

## Design principles

**1. Long posts. Keep roughly 80% of a paper's substance, so that reading the post is equivalent to having read the paper closely.**

The motivation is papers like *Demystifying NCCL: An In-depth Analysis*, which are almost entirely detail — summarize one and you have read nothing, because the detail was the point. Design reasoning is especially easy to lose: a sentence explaining that A has advantage C over B reads like a detail and gets filtered out, when it is often the argument. On the principle that skimming a slightly verbose text in one's first language beats skimming the original in a second, longer output is better than shorter.

**2. Keep the core method and experiments, and keep the paper's own order.**

A paper's structure is already the product of editing, and CS papers are formulaic almost to a fault — Introduction → (Related Work) → (Key Observations) → Method → Experiment → Discussion → Conclusion. That order matches the order in which a reader builds understanding, so it should be preserved. Method and experiments get the detail; the rest is compressed.

**3. Keep the important figures and tables, with their captions.**

A paper summary without its figures is not reading.

**4. Keep the important equations, code blocks and algorithm blocks.**

Same reason.

**5. Expand citations for the core method inline.**

A usability detail that turns out to matter more than expected: jumping to a PDF breaks the reading flow, and many readers cannot jump back. Expanded in place, the reference is simply there.

**6. Content over presentation.**

The paper pages are readable, which is as far as I took it. Polish can come later.

**7. Comments and reactions.**

So readers can flag errors and discuss. Both are built on GitHub Pages, and neither is thoroughly tested — bug reports welcome.

That is most of the thinking; the rest is detail. The harder parts are getting an LLM to follow instructions closely, and dealing with what a PDF parser does to equations and figures. The models doing the work are Gemini and Grok — principle 1 alone ruled out most of the alternatives.

Feedback is welcome.
