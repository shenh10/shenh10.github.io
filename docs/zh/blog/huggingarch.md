---
date: 2026-08-04
title: "HuggingArch：让模型 arch 分析自动化"
description: 用 harness 的思路让 LLM 推理算账自动化——只要模型在 HuggingFace 上开源，算账就应当是自动、可验证、可复用的。
---

# HuggingArch：让模型 arch 分析自动化

## Intro

> 这不是一个很正式的成熟项目发布，但是已经有了一个 work 的脚手架可以和大家分享。如果觉得这个项目有意思/有用，欢迎大家 star 及成为 developer，虽然我还没有想好 vibe 项目如何共建比较合理————这也是一个期望和有想法的朋友共同讨论的话题。

HuggingArch 是一个希望用 harness 的思路，让 LLM 模型推理算帐自动化的项目 ———— 只要这个模型在 HuggingFace 上开源。做这个项目的 Motivation 是很直观的，模型越来越多，作为从业者常常要分析和对比各家模型的优劣，这是模型厂和各种 MaaS 厂、芯片厂都关心的 critical issue。算账是一个劳心劳力的过程，而且门槛似乎并不低，过去一年，模型厂们大搞 hybrid，模型结构 diversity 越来越大，其实算账会变得更难算：DSA 和 KDA 哪个在长文本更省？CSA 和 HSA 相比 DSA 能更省多少？minimax 和 stepfun 通过 200B level 小模型部署能多支持多少 TPM？K3 作为 3T 级别的模型会涨多少成本？———— 光想到这些就感觉起码要算几个星期，为了脑子不OOM，2026年了我们必须让 agent 来帮我们算账了。


项目目前主要覆盖了model arch的计算图可视化，以及 LLM Inference 方面的推理理论算帐。将来可以支持特定shape的算子测速或者不同框架的算子性能上报，从而快速获取特定部署性能。理论上，HuggingArch 还可以把各个模型的 model 模块做成可插拔的模块，然后 Arch Designer 可以任意地修改超参，把不同模块单元排列组成成一个新的模型，并且快速算出这个模型在不同卡型和部署方式下的成本 ——— 这也是这个项目最有想象力的地方。


因为是一个纯 side project 的项目，从开始有想法到现在大概花了小半年的时间，一直在利用上下班碎片时间和周末缝缝补补。而模型能力在这半年也在高速飞跃。在当前，利用前沿模型快速分析一个模型的kvcache和理论flops已经不是个难事，那这个项目的意义在哪里呢？简单来说 ———— 我认为是一个**可信任的**、**可纠错的**、**可复用**的**轻量级**基础设施。

- 可信任：纯 vibe 分析一个模型是adhoc的。agent 的分析结果需要细细溯源计算过程，才敢判断这个分析结果是不是可信。而agent反而是一个更擅长告诉你结果，而不是中间过程的思维习惯。缺乏反馈，靠agent一个turn 梭哈直接出正确结果是不现实的，模型越复杂做对的概率越小。而人工反馈是一个很低效的过程。是不是有错，出了错，错在哪里 ———— 这个靠反复追问 agent 只会让会话越来越长，而反复的答案令人感觉越来越不可信。

- 可纠错：而 HuggingArch 设计了一套 validation 系统，通过模型的原始forward代码、config 配置、模型权重等 groundtruth 信息等，能让模型自己纠错，这样低级的错误几乎都能被消灭。此外，整套系统的计算都基于 sympy 的符号系统，使得所有值的计算都源自于特定的计算公式，并且从backend 引擎直接透传到前端，让每一个公式的出错都可以被人工发现。

- 可复用：所有模型都会以spec的形式入库。spec 从primitive、component、model 级别抽象出三层，于是模型之间的公共组件如Attention、MoE等是可以跨模型复用的 ———— 这降低了模型生成spec的难度、减少了计算的歧义，并且控制仓库的熵增在一个比较可控的范围。当模型以模块的形式在仓库里组件搭建起来 ———— 原则上我可以在这基础上为它赋予几乎整个Graph Compiler的能力 ———— 计算图的 shape 传播、quant、激活值liveness分析、device 映射 ———— 区别只是它底层不做实际的 tensor 计算，而是做FLOPs、Memory访存的计算。于是一份简单的、agent driven 的 spec，长出了复杂的模型分析系统。

- 轻量级：整套分析不需要模型能运行————因此不需要算子实现，不需要框架runtime，不需要GPU。这使得任意尺寸的模型都能快速的分析出结果，并且所有结果长期可追溯。


一套 LLM 模型的推理算帐系统需要两个必备因素：
1. 模型架构的表示

虽然可以手动build，但手动重建一个模型还是比较麻烦的，完全可以靠Agent来写。搭建这套系统最直接的agent需要的信息包括 ——— 模型权重形状、计算图结构及模型参数。这一切 HuggingFace 都正好有 ——— 模型权重、transformers 库以及 inference 源代码。所以我们这套 web 系统搭建基于 HuggingFace，但其实也有backend CLI 工具可以DIY（vLLM/SGlang 源代码和model weights 也是work的）。

2. inference 的算帐方法论

Speed-of-Light 估计是本项目的计算基础。LLM 的inference 由 prefill 和 decoding 两个阶段组成。每个阶段的计算 pattern，由主导其forward 时间的算子来决定。前者一般是计算密集型，因为 Gemm 算子和 Attention 算子形状都有较好的 roofline AI（算术密度），因此由GPU TFLOPS来约束其执行时间的下限，后者的 Gemm 算子和 Attention 算子一般是访存密集型，由访存带宽来约束其执行时间的下限。该指标的定义比较好的参考见 [SOL-ExecBench: Speed-of-Light Benchmarking for
Real-World GPU Kernels Against Hardware Limits](https://arxiv.org/pdf/2603.19173)。

$$T_{\mathrm{SOL}} =
\max\left(
\frac{\text{Total FLOPs}}{\text{Compute Throughput}},
\frac{\text{Total Fused Bytes}}{\text{Memory Bandwidth}}
\right)$$


根据每个算子的$T_{sol}$ ，我们可以通过计算图的传播自底向上推算出prefill/decode 阶段的 SOL 时间，从而得到对推理的成本估计。

---

## 错误的路径


在进入正题之前，想聊聊走过的弯路。

一开始想法很美好，过程很艰难。希望做一个轻量级的、不需要 GPU 的跨模型通用分析器——

- 不可能下载模型权重，几个 T 的下载根本无法并发。
- 不能实际运行，GPU 成本太高我付不起，复杂的 CUDA 版本和引擎版本跑起来也太重。

直观想法是使用 transformers 的 accelerator 及基于 meta device 机制，在每个 layer 执行之前 hook 一下计算 FLOPs，而模型结构可以通过图分析之类的方法拿到。但依赖于运行时的方法根本不 work——

### 1) Meta Device 的构建机制处理不了 forward 里的实例化操作

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

整个 `from_config` 跑完，你拿到一个完整结构的 `nn.Module`。但这个方法也存在天然缺陷——任何尝试 forward / 真正运算的操作会报 error，比如有些例子在 forward 里调用 `.cpu()`，这时就会报错。

### 2) 依赖 PyTorch 计算图不是一个好主意

模型结构只能从 forward 函数里推断，但希望从 PyTorch 里拿出计算图这件事相当于走了一遍 PyTorch 编译的来时路：

- 基于 `torch.export` 的方法处理不了 data dependent control flow，生成的算子也过于细碎难以理解。
- 基于 trace / dynamo 的方法需要 mock 输入，但每个函数的签名不保证一样，很难 mock。比如运行 Attention 时传入 kvcache，不同模型的 forward 函数签名不一样，很难构造出数据：

```python
class MiMoV2Attention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class GlmMoeDsaAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ...
```

- AST 的分析过于复杂，引入大量的 Python 原语操作。

### 3) Transformers 后向兼容性极差，以至于模型 runtime 根本跑不起来

以 MiMo-V2-Flash 举个例子：transformers 5.x 把 RoPE 配置从扁平的 `rope_scaling` 字段重构成 `rope_parameters: {default: {...}}` 嵌套字典，MiMo-V2-Flash 的 modeling 代码在 4.40.1 下写的，于是会报错：

```
modeling_rope_utils.py:1021: FutureWarning: `rope_config_validation` is deprecated
and has been removed. Its functionality has been moved to
RotaryEmbeddingConfigMixin.validate_rope method. PreTrainedConfig inherits this
class, so please call self.validate_rope() instead. Also, make sure to use the
new rope_parameters syntax. You can call self.standardize_rope_params() in
the meantime.
```

每个模型在 `config.json` 里指定了 transformers 版本，哪怕是动态多版本路由到正确的 transformers 版本，也会因为比如 Kimi 模型的 `modeling.py` 里直接需要依赖 flash_attention 库而在 CPU 环境里报错。

---

## 一、构建 model spec

HuggingArch的核心架构设计在于一套自底向上的DAG DSL描述系统，以及基于这套DAG 的SpecTree IR————DAG是claude code 写的，SpecTree IR是built-in的框架代码，这让agent始终是在有约束的前提下发挥自己把非结构化数据转变为结构化数据的能力。始终要强调的是计算图是现在深度学习框架的核心抽象，transformers只是基于这一套抽象的特殊结构，构建好计算图理论上能表示任意的复杂模型结构。在这套IR上挂上Sympy 符号计算系统，能同时表达结构关系、形状、与运算。有了这么一个计算图的描述，在这个基础上去做更高层级任务specific的运算————比如inference kvcache容量推导、并行的推导，都只是基于spec系统的应用层。

### 核心：Spec 系统

HuggingArch 根据transformers或者模型自由仓库的model inference源码来构建模型结构。这是一个开放的代码阅读问题，需要建立一套规范的DSL描述，使得agent（如claude code）知道如何去写结构。

#### 1) 分层 Op 系统：Primitive / Component / Model

HuggingArch 把模型组织成**三层 op 系统**——primitive op、component（compound op）、model——每一层都有自己的语法和职责边界。

**Primitive op** 是最底层的算子单元，定义在 `backend/arch_spec/ops.py`。每个 primitive 通过 `@register_op()` 注册，携带三个核心属性：参数量公式（`params_fn`）、FLOPs 公式（`flops_fn`）、形状模板。这些公式不是字符串，而是直接基于 sympy Symbol 的 lambda——比如 `linear` 就是 `2·B·S·In·Out` 这种符号表达式。primitive 目前有 60 多个，覆盖 `linear`、`embedding`、`rmsnorm` / `layernorm`、`attention_score`、`softmax`、`attention_apply`、各类 `activation`（silu / gelu / relu² / sigmoid …）、`rope`、`add` / `mul`、`split_heads` / `merge_heads` 等。这一层是封闭的，不允许业务方扩展，因为越底层的算子越要保证 FLOPs 和 param 公式的可信度。

**Component（compound op）** 是真正承载模型结构 diversity 的一层，每个 component 是一个 YAML 文件，物理上拆成 **三层目录** —— 目录位置即 tier 声明，没有额外字段也没有 tag：

- `backend/arch_spec/components/library/`：拓扑层面真正不同的积木。任何编码新 op 序列、新连接性、或现有结构无法仅靠 binding/rename 表达的变体都落在这里。GQA / MLA / SwiGLU / MoE / `pre_norm` / `gemma4_4norm_scaled_block` / `dsv4_csa_attention` …… 全在这一层。
- `backend/arch_spec/components/adapters/`：纯 binding + rename 的 thin shim —— 单节点 wrapper，把某个 library component 绑定到具体 checkpoint 的命名公约（`wq` vs `q_proj`、stacked-Parameter 专家 vs ModuleList 专家 etc.）。adapter 不引入新图节点；如果你需要加第二个节点，那这件事就该 promote 回 library 了。
- `backend/arch_spec/components/drafts/`：spec-gen agent 生成的临时区。drafts 启动时 parse 但 **不进入全局 namespace**，只对显式声明 `drafts: [<name>]` 的 model spec 可见 —— 一份坏的 V4-Pro draft 不会污染下一个 spec 的 manifest。review 通过后 `python -m backend.arch_spec.promote <name>` 直接 `git mv` 到 `library/`。

每个 component 描述一张 **DAG**：

- `inputs` / `outputs`：声明这个 component 暴露给上层的接口张量。
- `graph`：DAG 节点 dict，每个节点的 `op` 字段可以指向一个 primitive，也可以递归指向另一个 component，构成无限递归嵌套的层级。
- `in:` per node：声明从哪些上游节点拉数据 —— **engine 会沿着 `in:` 真正做形状传播**（细节见下一节）。
- `dims`：声明本节点的局部 dim 绑定（比如 `out: "N * D"`），与上游 shape 一起驱动 sympy 形状推断。
- `repeat` / `active`：用来描述 MoE 这种"声明一份，激活其中 k 份"的稀疏拓扑。
- `params`：参数化 slot，支持把 attention 类型、ffn 类型、norm 类型等做成可注入的角色（role），让上层模型能复用同一个 block 模板。
- `parallelism.shard_symbols`：声明此 component 在 TP 切分时哪些维度可切；inference analyzer 直接读这个字段做权重切分。
- `kv_label` / `kv_window` / `attn_variant`（attention component 专用）：把 KV cache 拓扑/窗口/变体打上标签，KV cache 建模和前端徽章都基于此 —— 加一种新线性 attention 只要在 component yaml 里声明一行 `kv_label:`，所有下游工具自动识别。

举个例子，`gqa_attention.yaml` 长这样：

```yaml
name: gqa_attention
role: attention
description: "Plain Grouped-Query Attention with no QK-norm. Llama, Qwen2, Mistral."
inputs: [x]
parallelism:
  shard_symbols: [N, Nk]                       # TP 切分轴：Q 走 N，K/V 走 Nk
kv_cached: [k_split_heads, v_split_heads]      # KV cache 取这两个节点的 shape_out
kv_label: GQA
attn_variant: GQA
graph:
  q_proj:          { op: linear, in: x, dims: { in: H, out: "N * D",  bias: Qb } }
  k_proj:          { op: linear, in: x, dims: { in: H, out: "Nk * D", bias: Qb } }
  v_proj:          { op: linear, in: x, dims: { in: H, out: "Nk * D", bias: Qb } }
  q_split_heads:   { op: head_split, in: q_proj, dims: { N: N,  D: D } }
  k_split_heads:   { op: head_split, in: k_proj, dims: { N: Nk, D: D } }
  v_split_heads:   { op: head_split, in: v_proj, dims: { N: Nk, D: D } }
  q_rope:          { op: rope, in: q_split_heads, dim: D }
  k_rope:          { op: rope, in: k_split_heads, dim: D }
  qk_score:        { op: attention_score, in: [q_rope, k_rope], dims: { N: N, D: D } }
  softmax:         { op: softmax, in: qk_score, dims: { N: N } }
  attn_apply:      { op: attention_apply, in: [softmax, v_split_heads], dims: { N: N, D: D } }
  o_merge_heads:   { op: head_merge, in: attn_apply, dims: { N: N, D: D } }
  o_proj:          { op: linear, in: o_merge_heads, dims: { in: "N * D", out: H } }
outputs: [o_proj]
```

几个值得注意的点：

- **`role: attention`**：声明这个 component 可以填进上层 block 模板里 `attention` 这个 slot。允许的 role 包含 `attention` / `ffn_dense` / `ffn_moe` / `router` / `block` / `aux_block`，省略则代表是子组件 / utility（compressor、indexer、`head_split` 等不能当 slot 直接插的东西）。
- **`graph` 是 dict 而非 list**：每个 key 是节点名，value 描述 op 类型和它的依赖（`in:`）——节点之间通过名字相互引用，构成 DAG。`dims` 字段直接写 sympy 表达式字符串（`out: "N * D"` 会被 parse 成 `N·D` 的符号 expr）。
- **`parallelism.shard_symbols`**：声明 TP 切分时哪些维度可切（`N` for Q heads，`Nk` for KV heads），inference analyzer 直接读这个字段做权重切分。
- **`kv_label` / `kv_cached` / `attn_variant`**：attention 类 component 给 KV cache 建模用的标签 —— 加一种新线性 attention 只要声明 `kv_label`，所有下游工具自动识别它属于哪个 KV 拓扑家族。
- **`bias: Qb`**：通过一个可选 dim 把"有没有 QKV bias"做成参数化开关 —— 上层 model spec 给 `Qb=0` 或 `Qb=1` 就能复用同一个 component。

整张图就是 12 个节点串起 `q_proj → head_split → rope → attention_score → softmax → attention_apply → head_merge → o_proj` 的标准 GQA 流程。这套 component 系统是开放的 —— 这也是为什么新模型能以组合方式构建：MQA 把 `Nk` 设成 1，MHA 把 `Nk` 设成等于 `N`，partial-RoPE 就再做一个变体替换 `rope` 节点。

> **encapsulation parity**。Agent 在写拓扑结构的时候，很容易丢失语义信息——比如一个没见过的语义模块，在源码中是一个独立的layer，但是agent很容易把它inline展开————这就丢失了对human 友好的语义体系。在prompt 我们需要约束 spec 的封装颗粒度必须 ≥ 源代码 `forward()` 的封装颗粒度 —— 以Deepseek V4 Pro为例，上游作者写了 `def hc_pre(...)` 就说明他画了一条"这是一个原子操作、可被多处调用、可独立测试"的语义线，如果把它inline了，就会变成一堆可读性很差的原子模块，这对于我们阅读分析模型结构式非常不利的。约束spec 可以比源代码 **更抽象**（将复用于多处的同一段实现封装为一个 component），但不能**更碎**，有利于agent找到自己的拆解粒度，形成更好的架构绘制风格。

**Model 层** 在 `ArchSpec` 中定义，描述：本模型有几种 block 模板？每一层 instantiate 哪个 block？block 内的 attention/ffn slot 各填什么 component？比如 `qwen2.yaml` 用一个 `block` 模板加 `params: {attention: gqa_attention, ffn: swiglu_ffn}`，layer assignment 直接展开 `L` 层；hybrid 模型（某几层 MLA 某几层 SWA）就在 `model` 节点里给不同 layer index 写不同 template。可分析范围目前锁在 decoder-only 的（多模态）LLM，详细的 schema 边界与设计取舍见 [`docs/design.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/design.md) §1.4。

#### 2) 自底向上的三层的符号系统

希望 HuggingArch 给出的不只是数字，而是**可解释的公式** —— 这才是一个算账系统必须要具备的能力，否则可解释性就不存在————我是很难相信一个黑盒的planer，因为它做的任何 assumption 可能都藏在代码细节里，且大模型infra的技术随时在变————我不要求算账的系统什么都能建模，但我必须知道他建模有哪些assumption，这才能让我知道这个模型输出的数字可信度。 在前端上 hover 任何一个值都能看到背后的 symbolic expression 是我觉得比较理想的展示方式。这就需要一套贯穿三层 op 系统的符号代数。

每个 primitive 的 `flops_fn` / `params_fn` 都接受统一的维度符号（`B` / `S` / `H` / `N` / `Nk` / `D` / `I` / `V` / `L` / `E` / `R` / `P` / `Dv` …），返回 sympy `Expr`。Component 嵌套时外层把 dim binding 推进 scope，内层接住继续展开 —— 整张图最终是一棵 sympy 表达式树。

**双轨求值是这套设计最大的回报**：同一份 Expr，传 `Symbol` 进去出来是代数式（`2·B·S·H²`），传 `int` 进去出来是数。引擎本身只有一份代码，符号 vs 数值取决于输入的类型 —— 不写两套引擎、不在中间分支判断"这是符号吗"。前端 hover 拿到的是两条同步的渲染树，公式和具体数值并列显现，不会有"数字对不上公式"的尴尬。

**Model 层的聚合刻意保留异构结构**：hybrid 模型最终得到 `Ld·MLA_FLOPs + (L − Ld)·MHA_FLOPs` 这种和式，不打平成单一公式 —— inference analyzer 在算 KV cache、weights、traffic 时能正确把异构 attention 加起来。

但符号系统只解决"数字怎么算"，没解决"图本身对不对" —— 后者要靠下面两节讲的两条主 ground truth（后面会看到，真正的 harness 在这两条骨架上还长出了更多 guard）。

### 事实来源：Ground Truth 从哪来

为了让 spec 生成既精准又便宜，HuggingArch 从 HF 模型的多个事实源拉取信息，**完全不下载权重**：

- **`config.json`**：通过 `backend/analyzer.py` 的 `_fetch_config()`，依次尝试本地 custom-model 缓存、内置 snapshot（`backend/arch_spec/snapshots/`）、HF Hub 在线下载。这是模型尺寸（H/L/N/I/V 等）的标量来源。
- **模型骨架打印**：`_fetch_model_str()` 利用 `init_empty_weights()` 在 meta device 上构建一个空模型，然后 `print(model)` 拿到模块层级树。这一步不下载权重、不跑 forward——只为了拿到嵌套的 `nn.Module` 名称结构和 shape 注解。结果会被缓存在 `~/.cache/huggingarch/model_info/`。
- **forward 源码**：`backend/hf/forward_source.py` 通过 `inspect.getsource()` 拉 `DecoderLayer` / `Attention` / `MLP` 等关键类的 forward 实现。这是描述 op 连接关系唯一可信的事实源，给 spec 生成 agent 看，避免幻觉。
- **safetensors 索引**：`backend/hf/metadata.py`（一个相当大的模块）只用 2–3 个 HTTP range 请求拉到 `model.safetensors.index.json` 和每个 shard 的 metadata block，得到**每个 tensor 的名字、shape、dtype、storage bytes**——但不会下载任何权重字节。这是参数量 + bytes 校验的 ground truth。

对于**不在 transformers 主干、只在 model card 中以 trust_remote_code 形式给出的私有仓库**（典型如 Kimi、MiMo、GLM-MoE 等），HuggingArch 走 HF Hub 拉模型 repo 里的 `modeling_*.py` 与 `configuration_*.py`，将 forward 源码输入给 agent；遇到 `flash_attn` 这类 native 依赖时也不会真的去 import，只是当作文本来读——由此规避了"runtime 无法运行"的困境。

事实源备齐之后，下面两节讲怎么把它们各自变成校验：forward 源码 + 模型骨架对应 forward 数据结构这条路径，safetensors 索引对应参数量 + 量化这条路径。

### Shape inference：forward 数据结构的 ground truth

让 agent 写 spec 时，最容易踩的不是参数量错 —— 那种错容易被后面讲的 tensor 比对兜住。最容易踩的是**几何错**：MLA 的 `k_pe` 忘了 broadcast 到所有 head 就直接 concat、view_split 在一个本来不是 `N·D` 的 flat dim 上做、Q 和 K 的 head_dim 对不齐 …… 这些错在参数量上**完全可能正好抵消**（一个 linear 多算几个参数、另一个 linear 少算几个，bytes diff 还是 0），但 forward 数据流是错的。spec 看似通过校验，跑 inference analyzer 却给出离谱的 KV cache、错误的 attention type、互相不对齐的 shape badge。

**这个问题的根本难点是：HuggingArch 不跑模型，怎么验证 forward 拓扑是对的？**

最初的版本走过一段弯路：试图让每个 component 自己声明每个节点的 `shape:`，引擎只查一致性。但这条路本质是把负担推给 spec 作者 —— 一个有 60 个节点的 V4-Pro block，作者要把每个节点的 shape 算清楚写进 yaml，错的概率比直接写 forward 还高。

**最终的路径是把 shape inference 做成 first-class 机制**：

1. **`in:` 不是注释，是真正的 dataflow edge**。引擎顺着每个节点的 `in:` 在已求值表里查上游 NodeResult，将其 `shape_out` 输入给当前 op 的 shape function。
2. **每个 primitive 声明一个 `shape_kind`**，目前 7 种（`preserve` / `axis_replace` / `axis_concat` / `contract` / `permute` / `source` / `explicit`），引擎套用对应规则做形状推断。新加 op 是**声明性的** —— 不让每个 op 各写一份 mini shape solver，那样错误难定位、新 op 接入成本高。
3. **多尺度统一**：component 内的节点之间、component 嵌套调用、跨 model 层（每一层的 `shape_out` 喂下一层 `inputs[0]`）—— 三个尺度共用同一套 dataflow 传播。V4 那种 `[B, S, Hc, H]` 的 hyper-connection 输入走过整张图，引擎看到的就是真实张量形态，不需要下游反向猜。

把 shape 抬到 first-class 之后才有了关键能力：**validate 阶段可以在不跑数值的前提下做几何一致性检查**。每个多输入 primitive（`axis_concat` / `attention_score` / `contract` / 多输入 `add` / `mul`）的上游 shape 都被验证 —— rank 是否一致、非 axis 维是否相等、Q/K 的 head_dim 是否对齐 …… 几何错在 dataflow 推进时就抓住，不再靠"跑完发现 bytes 对得上但 attention 看起来怪"反推。

**这条路径的真正价值**是把"forward 数据结构"从一个无法直接验证的隐式属性，变成一个可以独立校验的 first-class artifact。Ground truth 是上游 forward 源码描述的连接关系；spec 写出来后，shape inference + 几何一致性检查就是这份 ground truth 的本地复现。

### Tensor weights：参数量与量化的 ground truth

LLM 写出"看起来很合理但其实哪里错了"的 spec 是日常。但只要权重还在 ckpt 里，**真相就在那里** —— safetensors 不会撒谎，它的 tensor 名字和字节数是上游模型团队亲手放进去的事实。问题是怎么把这份事实拿来当校验。

#### 一份事实，两层信息

HuggingFace 的 tensor 命名遵循一套相当稳定的公约：

```
model.layers.X.self_attn.q_proj.weight                         ← 标准 LLM
model.layers.X.mlp.experts.X.gate_proj.weight                  ← MoE
model.vision_tower.encoder.blocks.X.attn.q_proj.weight         ← vision tower
model.language_model.layers.X.self_attn.q_proj.weight          ← 多模态嵌套
```

这套命名同时编码了**两类信息**，校验机制分两路处理这两类信息。

**拓扑信息**（哪一层、哪个 attention slot、哪个 expert）—— 由命名前缀的层级表达。validator 把这些 pattern 拆出 "block-internal suffix"（`self_attn.q_proj` / `mlp.experts.X.gate_proj` / `attn.q_proj`），再用"最长后缀优先"和 spec 节点的层级路径做后缀匹配。异构 stack 里同一个 module path 可能被多个 spec 节点同时认领（`input_layernorm` 在 `dense_block` 出现 Ld 次、在 `moe_block` 出现 L−Ld 次），matcher 把所有命中求和再对账。

**量化打包信息**（这是 GPTQ qweight 还是 MXFP4 _blocks 还是 FP8 .weight + scale_inv）—— 由命名后缀和 dtype 共同表达。这部分 HuggingArch 的策略是把**每种 quant scheme 的 on-disk 表示写成 yaml**：主权重的 packing 因子 / 存储 dtype / 后缀，元数据 sibling 的 suffix 与 cardinality 公式，逻辑 element format 的判定规则，scale granularity 类型。

这句话听起来平淡，但它是一个**有意识的边界划分**：tensor 命名公约和打包细节这种"上游 transformers / autoawq / compressed-tensors / 各家训练框架各有各的写法"的散乱知识，本来散落在多个 Python 库里，每加一个新 scheme 就要进一处侵入式逻辑；归一进 yaml 之后，**新增 quant scheme 是声明 yaml 的活，不是改 Python 的活**。下游所有消费方共享同一份事实源。

#### 量化的解耦：架构 vs 存储

量化模型的复杂之处在于：模型架构和量化策略是**两件可以独立组合的事**。同一个 DSv3 可以有 BF16 ckpt、FP8 ckpt、INT4 ckpt；同一个 ckpt 也可以问"如果重量化到 W4A16 会怎样"。HuggingArch 的解法：

- **架构层是 dtype-无关的**：spec 里的 `linear` op 关心参数量和 FLOPs，是几何事实，公式的 free symbol 里没有 dtype。
- **存储层是 dtype-相关的，且单独表达**：spec 自带一段 `quant_context`，按 role 维度声明量化方案，再加一组 override 处理 per_role 装不下的边界情形。

例如 gpt-oss 这种"只 MoE 量化、其余留 BF16"的混部可简洁声明：

```yaml
quant_context:
  per_role:
    ffn_moe: { method: mxfp4, group_size: 32 }
    default: { method: none }
  overrides:
    - tensors_re: 'layers\.X\.mlp\.router\.weight$'
      dtype: BF16
  default_dtype: BF16
```

DSv3 / Kimi-K2.5 这类把 attention / FFN / router / vision_tower / lm_head 各自挑出量化策略的混部模型，也是这一种形式 —— `per_role` 表达整体方案、`overrides` 处理边界。`materialize.py` 是把 `quant_context` 落到具体 tensor 的**唯一路径**：输入 spec 解析出的逻辑权重清单 + `quant_context` + scheme yaml，输出每条 tensor 的物理 dtype 和字节数。validator、weight breakdown、KV cache、前端徽章全部消费这一份输出 —— 没有"快速路径"绕过它。这是一种克制：一旦量化分散在多处实现，混精模型会立刻让两边对不上。

副产品是 spec 同时能回答两个问题：**它实际上是什么**（用 spec 自带的 `quant_context` 跟真实 checkpoint 对账），**如果换 dtype 部署会怎样**（Inference 页脱开 `quant_context` 让用户挑任意 deployment dtype，sympy 公式重新求值就行）。两条路共享同一份 spec 与同一套符号代数。

#### 兜底：bytes-vs-bytes diff

最终的硬判据是 **storage_bytes 等不等**。spec 这边 materialize 给每个 leaf 的真实 on-disk 字节数，HF 这边读 safetensors header 的 per-tensor 字节数。选 bytes 而不是 numel 的理由是：混精 checkpoint 里（FP4 routed expert + FP8 attention + BF16 norm 同处一个 ckpt），**唯一在单位上不需要折算的就是物理字节**。任何不一致以"哪个 module、HF 多少 bytes、spec 多少 bytes、diff 多少"的结构化形式写回 agent —— 自我纠错的闭环靠这套粒度足够细的反馈撑起来。这层兜底也间接覆盖系数级形状错误：把 `out_features` 系数写错的话参数量立刻不对，agent 拿到的反馈是"`self_attn.o_proj` 我说有 X、实际有 1.5X"，比一句"shape 不匹配"的 solver 报错更可操作。

### 两条主 ground truth 构成一套 guard harness

最根本的两条事实路径先说清楚 —— 它们是整个 harness 的骨架：

|  | Ground truth 来源 | 在 HuggingArch 里复现的方式 | 抓的错误类型 |
|--|------|------|------|
| **Forward 拓扑** | 上游 `modeling_*.py` 描述的连接关系 | shape inference + 多输入 op 几何一致性检查 | 拓扑错（rank 不齐、Q/K head_dim 不匹配、broadcast 漏做）|
| **参数量 + 量化** | `safetensors.index.json` 的 tensor 名字 + 字节数 | tensor name matching + materialize → bytes diff | 数量错（系数写错、漏 module、量化方案错配）|

这两层互补：几何错由 shape consistency check 在 dataflow 推进时就抓住，**不依赖**参数量是否凑巧抵消；数量错由 bytes diff 在 ship 前抓住，**不依赖**几何是否恰好自洽。Agent 不需要"猜对" —— 只需依据反馈修正，修正后重跑，loop 会自我收敛。这套机制是这个项目作为 harness engineering 的核心：**校验比生成贵、比生成难，但是校验跑通后，生成可以放心交给概率模型**。

但得说清楚一点：随着模型越来越复杂，这两条主 GT 之上又长出了更多校验。后来做 guard-ablation 实验时（见文末后记），我们干脆把整套 harness 拆成了 **5 道可以独立开关的 guard**：

- **`weight`** —— 上面那条 bytes / packing / tensor 匹配（事实源：safetensors）。
- **`shape`** —— 上面那条几何一致性（事实源：forward 拓扑）。
- **`source`** —— 把 spec 的调用结构直接和 forward 源码对 diff：call-order、残差 add 的个数对不对。**shape 对了不代表连接顺序对**（事实源：forward 源码）。
- **`axiom`** —— 把 forward 源码里锁定的逐层不变量（典型是 KV 增长：一层每 token 缓存多少、SWA 有没有把窗口封住）当成可检查的断言，专抓 bytes 看不出来的语义错 —— V4 那个"滑窗漏封 → 缓存爆到 128TB 量级"就是它拦下的（事实源：forward 源码）。
- **`structural`** —— 逐模块可执行性 + 一条**独立事实源**：`config.json` 的逐层 config 列表（`moe_layer_freq` / `mlp_layer_types` / `sliding_window` pattern…）。spec 的 per-layer 模板必须**穷尽**这些列表实际取到的每个值，不能靠一个 `rest:` 兜底悄悄吞掉一种没建模的层型（事实源：config.json）。

换句话说，**"两条 ground truth"是骨架、但不是全部**：forward 源码这一条事实源其实支撑了三种不同的校验（shape / 源码 diff / axiom），`config.json` 也不只是标量 dim 的来源、它的逐层列表本身就是一条独立的 gating 事实。文末后记里"不设 guard 94% → 全开 6%"那条曲线，量的正是这 5 道 guard 一档一档装回时 agent 产出可信度的变化。

剩下没被这些 guard 覆盖的边界情况，由一层轻量的 schema 校验补上：`template:` 引用必须能解析、model 层 `inputs:` 必须是前向 DAG（不能引用后面的层、不能成环）、role 必须在白名单里。**但 schema 不挡新拓扑** —— agent 遇到 library 没有的结构，直接在自己 spec 的 `blocks:` 段 inline 一个新 block 就行，schema 看到本地有定义就 pass。novel topology 的成本应该是"写一份新 yaml"，不是"先 PR 改 library 再写新 model"。Schema 是 reject 写错的引用，不是 reject 创新。

### 新模型 spec 的生成流程

回到最初的目标——"输入一个 HF model ID，输出一份 verified spec"。`backend/spec_worker/` 把它编排成一条 agentic 流水线。绝大多数新模型在拓扑上是已有 component 的组合（一个 hybrid GQA + MLA、SwiGLU + 256-expert MoE、加 sliding window、换个 RoPE），agent 只需在 model spec 里把 attention / ffn / norm 等角色 slot 填上现成 component，`adapters/` 层的薄 shim 吸收同一拓扑下不同 checkpoint 的命名差异。确有新颖拓扑（如 DSv4 的 hyper-connection、CSA 的稀疏 indexer、SSM）时，agent 先在 `components/drafts/` 写一份新 component 并在本 spec 内声明，review 通过后由 `promote` 以一次 `git mv` 并入 library——扩展系统与使用系统受同一套校验保护，未 review 的 draft 只对声明它的 spec 可见，不污染全局 manifest。

流水线自动为 agent 组装 prompt：op registry、component manifest、schema 与已验证示例，以及本模型的 ground truth（forward 源码、`config.json`、safetensors tensor 清单）。随后 spawn 一个 agent 后端（`claude` / `codex` / `kimi`，抽象在 `backend/agents/base.py`）在 sandbox 内写 YAML、调用 `validate`、依据 unmatched tensor / shape mismatch / bytes diff 反馈修正，直到 `bytes_diff == 0` 或达到最大轮次——一个完全自动的 refinement loop。校验一旦跑通，生成即可放心交给概率模型。

---

### Guard 必要性：从-0 消融实验

上面这条流水线的产出可信度，完全取决于那套 guard 到底拦不拦得住错。一个自然的质疑是：guard 是不是多余的——足够强的 agent 不用 guard 是不是也能写对？我们把它做成一个受控消融实验来回答（`experiments/guard_ablation/`，完整报告见 [`docs/guard_ablation/report.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/guard_ablation/report.md)），从两个互补的角度切入。

**其一，确定性的唯一覆盖分析。** 对每道 guard，把它专防的那类错误注入到已提交的正确 spec 里，再用**全套** guard 去测——看它是不是唯一能抓到的那个。结果是 14/14 个 live 单元里，每道 gating guard 在它设防的故障上都是**唯一 catcher**，没有一道能被其它 guard 兜住。尤其在不声明任何 axiom 的模型（MiMo）上，SWA 的 `kv_window` 契约是滑窗 KV 被当成 dense 存储爆掉的**唯一后手**。

**其二，公平的从-0 agentic 消融。** 让 spec-writer agent（Claude Opus 4.8）在一个**去污染**的沙盒里**从零**重写每个模型的 spec：把目标自己的 component、全库同类 component、所有别的 model spec、别的 snapshot、fusion plan、以及新颖算子的 primitive 全部剥掉——agent 只剩通用积木 + primitive op + 目标的 ground truth（forward 源码、`config.json`、tensor 名），必须自己**推导**出架构、把新算子当 in-spec `custom_ops` 重新声明（一遍 escape scan 确认没有 run 逃出沙盒）。guard 按 `裸奔 → +weight → +shape → +struct/source → +axiom(全开)` 逐档装回，每档 K=3 个 seed，oracle 是全开 guard 的 validator。

结果是一条干净的严重度阶梯——每装回一道 guard，就消掉它对应的那类错误：

![Guard-necessity ablation：三个模型（MiMo-V2.5-Pro / DeepSeek-V4-Pro / GLM-5.2）在 opus-4.8 下，从裸奔到全开逐档装回 guard 时的违规严重度（K seed 均值）——柱越高错得越多，%✓ 是该档完全通过的 seed 比例](/blog/huggingarch/error_distribution.png)

裸奔那一档 V4-Pro 是一根 32.7 STORAGE + 1.8 GB 的红柱，随 weight / shape / schedule / axiom 逐档装回一路塌到 0；每道 guard 只吃掉它对应的那一色，互不越界：

| 模型 | 裸奔（无 guard） | 全开（全 guard） |
|---|---|---|
| **DeepSeek-V4-Pro**（1T, Hyper-Connections） | 0% valid —— 32.7 STORAGE + 7.3 NAMING + 4.3 SHAPE + 1.3 KV，字节差 1.8 GB | **100% valid** |
| **GLM-5.2**（MLA+DSA+MoE） | 0% valid —— STORAGE | **100% valid** |
| **MiMo-V2.5-Pro**（SWA + fused-QKV） | 0% valid —— STORAGE + SCHEDULE | **100% valid** |

weight→消 STORAGE、shape→消 SHAPE、schedule→消 SCHEDULE、axiom/SWA `kv_window`→消掉最后残留的 KV-语义错误（在 V4-Pro 与 GLM-5.2 上一直撑到全开档才清零）。

**失败模式恰恰是"自信地错"。** 在裸奔 / 仅-weight 这些档位，agent 退出时 `success=True`、而且**它自己那套（被削过的）validation 是绿的**——它以为自己写完了——但全开 oracle 一测，spec 是坏的。agent 看到的和真相之间那道缝，**就是**这道 guard 的贡献：没有它，agent 收不到任何"你错了"的信号，于是自信地宣布成功、交出一份坏 spec。而且 guard 的必要性随"要 agent 推导的东西有多难"正相关——1T 的 Hyper-Connections 模型裸奔时错得最离谱（32.7 条 STORAGE 违规、1.8 GB），要全套 guard 才能把一份自信但错的草稿变成正确的 spec。

这里的 oracle 不是自证的：`validate()` 锚的是**真实模型 artifact**（checkpoint 的 tensor 名 / 字节 / dtype、`config.json` 值、forward 源码），不是任何 spec；对已提交 golden 做的结构等价交叉核对（名字无关的 params / bytes / 逐 block profile / KV 语义比对）在清晰 case 上与 oracle 一致，分歧处 oracle 抓得**更多**——SCHEDULE 的逐层调度覆盖、NAMING 的 tensor 映射，正是纯 params/结构比对看不见的那部分。（成本上，最难的 V4-Pro 每次从零推导要 62–87 轮 agent turn、13–25 M token、44–63 分钟、$14–22，是另两个模型的 3–4×。）

结论就一句：**每道 guard 都必要**——确定性上各是其设防故障的唯一 catcher，经验上一旦拿掉，从零重写的 agent 就会带着虚假的自信把那类错误交付出去。换句话说，guard 拦下的不是"算错"，而是"错了还以为对"。

---

### spec 的可视化：Gallery、Inspector

spec 可信之后，最直接的用途是对其进行可视化与交互。**Gallery**（`GET /gallery/overview`）将整个 snapshot 库聚合为全局对比视图——发布时间线、total-vs-active 参数散点、各模型按原生 dtype 在长上下文下的 prefill/decode SOL 曲线；所有数值复用 Inference 的同一条计算路径，可在单一视图中横向比较不同时期的架构演进。**Inspector**（`backend/rendering/builders.py` + `frontend/inspector.html`）将 spec 递归展开为可逐层展开的 DAG，每个节点携带 shape / params / FLOPs，鼠标悬停时同步显示 symbolic 公式与 numeric 数值两条同源渲染，使每个数值的来源均可追溯；从单模型切换到对比模式，可并排两个模型逐 block 比较 attention 类型、KV cache 与参数分布。

## 二、构建 Inference 算账系统

基于这份可信 spec 与贯穿三层的 sympy 双轨求值，"某个模型在某张卡上能否部署、如何部署"即成为 spec 之上的应用层——无需修改 modeling 代码，所有数值均由同一棵表达式树派生。以下五节对应该成本估算系统的五个机制层：推理框架如何将多个 op 融合为单个 kernel（Fusion）、部署方案如何将张量切分到各设备（并行切分）、切分后各设备实际执行的形状如何导出（Shape 系统）、基于 SOL 的 prefill/decode 吞吐与容量如何计算（SOL 吞吐预测），以及如何用实测将 SOL 上界修正至可达吞吐（实测算子库驱动）。

### 2.1 Fusion

Spec 描述的是数学意义上的计算图，但真实推理框架运行时会将一串 op 融合为一个 kernel——spec 里分开的 `q_proj` / `k_proj` / `v_proj`，在 vLLM 里可能就是一个 fused QKV GEMM。Fusion Patterns 页（`frontend/fusion.html`，后端 `GET /fusion/registry` + `/fusion/fired`）把这层"框架怎么融"显式建成一份 registry：每个 fusion plan 声明一组可折叠的 op 拓扑，每个 framework binding（vLLM / sglang，按版本）声明这个框架实际启用了哪些 plan。选一个模型 + 框架 + 版本，`/fusion/fired` 即在该模型 resolved spec 上执行一次 `apply_fusion`，报告哪些 plan 命中、锚在哪些 leaf 节点、折叠掉了哪些 sibling。因为 fusion 匹配是纯拓扑的（不依赖 TP / EP / GPU），这个分析在 batch=1 / seq=1 就能算，很便宜。它的价值是让 spec 不止停在"理论算子图"，还能对上"某个框架某个版本真正会怎么跑"——fusion plan 和 binding 都是从真实的 vLLM / sglang 源码推导出来的，而非依据博客推测。

### 2.2 并行切分

一个模型分布到多卡时，每张卡的显存与通信取决于张量如何被切分。最朴素的做法是"全量 ÷ cluster_size"的均匀除法，但当模型在不同 region 采用不同 TP 度（attention 切 8、MoE expert 切 32）、并叠加 CP 与 EP 时，全局除法无法表达哪些张量处于同一通信组、哪条边需要插入通信。HuggingArch 因此将并行建模为一套 placement 代数（`backend/arch_spec/sharding.py`），其抽象借鉴 PyTorch 的 [DTensor](https://pytorch.org/docs/stable/distributed.tensor.html)（及其思想源头 GSPMD，[Xu et al. 2021](https://arxiv.org/abs/2105.04663)）。DTensor 的核心是：一个逻辑张量的分布由它在设备网格（DeviceMesh）各维度上的 **placement** 描述，取 `Shard` / `Replicate` / `Partial` 三态；当算子在分布式张量上执行时，输出的 placement 由输入 placement 按算子语义推导，二者不一致处**自动插入 redistribute**（即所需的 all-gather / all-reduce / reduce-scatter / all-to-all），从而让分布式张量的编程与单机保持一致、通信不需手写。HuggingArch 沿用这套 placement 语义，但不做实际张量计算，只在计算图上传播 shape / bytes / FLOPs：每个张量携带一个 `TensorShardingState`，记录它在每个并行 world（`tp_attention` / `tp_moe` / `ep` / `cp` …）上的 placement（`Replicated` / `Sharded(axis, world)` / `Partial(reduce_op, world)`），多个 world 正交叠加（例如 TP-attention 的 head-shard 与 CP-Ulysses 同时作用，即两个 world 各自在 `N` 轴上 `Sharded`）。

这一抽象的价值在于通信不再逐 op 特判，而由生产者与消费者的 state 差异自动派生：`Sharded → Replicated` 派生 all-gather，`Partial → Replicated` 派生 all-reduce，多个 world 逐轴分解、各自产生一个 collective。EP routing、CP 的 Ulysses / ring、decode 的 KV 分片与 LSE combine（对齐 vLLM DCP）都由同一套代数导出，没有任何一处通信是为特定并行组合硬编码的。计算时间上，comm 字节按 region-aware 的 mesh 判定走 intra 或 inter 带宽，FLOPs 则除以仅含 model-parallel 轴、排除 DP 的 `placement_divisor`；Inference 页将 TP / PP / EP / CP / DP 暴露为一组控件，每个 leaf 的通信时间在 `max(t_compute, t_memory, t_comm)` 中单独呈现。这套"op 原生 sharding mode → 分片状态传播 → 通信注入"的完整规则、状态代数与 per-rank 成本见 [`docs/parallelism.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/parallelism.md)。

### 2.3 基于 SOL 的 Prefill/Decode 吞吐预测

Intro 给出了单个算子的 SOL 时间 $T_{\mathrm{SOL}} = \max(\text{FLOPs}/\text{峰值算力},\ \text{bytes}/\text{带宽})$；沿计算图自底向上聚合，即得到 prefill 与 decode 两阶段的端到端 SOL 时间，进而是 TTFT、TPOT、吞吐与 max-batch 容量。这一层不依赖任何 kernel 实测即自成闭环，其价值在于提供一个统一、可复现、与具体实现无关的分析基准：

- **性能上界**：SOL 是硬件 roofline 决定的理论下限时间，任何真实实现都无法突破，可据此快速判断一个模型在给定 GPU 与部署方案下能否部署、量级如何。
- **部署方案对比**：同一模型在不同 TP / EP / 并行度 / GPU 上的相对优劣，在同一 SOL 口径下可直接比较，无需分别实测。
- **跨模型对比分析**：不同架构（MLA 与 GQA、DSA 与 KDA、各类 hybrid）在长上下文与高并发下的相对成本，SOL 提供了唯一一个不受 kernel 实现差异干扰的对照基准——这正是 Intro 所述"算账"最核心的诉求。

下面是 SOL 闭环各项分量的计算方式：

这个闭环最终要算两个数：每张卡装不装得下（显存），和一步要多久（时间）。分头算它俩之前，有两样输入是共用的——而且都不只影响显存：

- **并行方式**：前面 2.2 那套 placement 代数决定每个张量在每张卡上怎么切。`estimate_capacity()` 收一个 `ParallelConfig`，把笼统的 TP 拆成 attention / dense-FFN / MoE-expert 三路各自的并行度，再叠加 EP、PP、CP、两类 DP 与 SP。同一套切分，weight、KV、激活、通信全跟着走——所以并行既进显存，也进时间。
- **GPU 硬件**：`backend/inference/gpu_specs.py` 是一张 CSV 的 GPU 参数表。HBM 容量给出显存上界，HBM 带宽与 peak BF16/FP8 算力给出时间的 roofline。

这两样输入之上，显存与时间各自怎么算，如下。

**显存**走的是同一套 sympy 引擎，把量化与 attention 类型纳入符号系统：

- **量化**：weight bytes 由 `weight_dtype × params + quant_overhead` 决定（见前面 Quantization 段）。
- **KV cache 的 attention-type 感知**：`backend/inference/symbolic.py` 的 `kv_cache_per_token_per_layer()` 对每种 attention 给出独立公式：MHA / GQA / SWA 是 `2·Nk·D·dtype_bytes`，MLA 是 `(R + P)·dtype_bytes`（压缩 latent + RoPE-only key，不再 fanout 到每个 head）。线性 attention / SSM（GLA、GatedDeltaNet、Mamba2）这类循环态不是"每 token 累加"，而被建成一类**常量存储的 KV**——`backend/inference/kv_topology.py` 给它一个 `recurrent_state` 节点，每 token 的 element 数是常量、不随 context 增长，`kv_cache.py` 里以 `constant_floor` 落成固定字节。Hybrid 模型的 KV cache 自然按 layer 类型加权求和——per-token 增长的分支和常量循环态各算各的，不会被不加区分地处理。

**理论时间与 MFU**：`backend/inference/phases.py` 把单步推理拆成 phase——weight load、compute、KV store / load、通信。每个 phase 的理论时间 = bytes / bandwidth 或 FLOPs / peak_TFLOPs，phase 之间取最大值（roofline 视角）。MFU 在 `/inference/estimate` 端点直接计算成 `model_FLOPs_per_token / (peak_TFLOPs × measured_latency)`，配合 phases 拆解就能定位"是带宽 bound 还是计算 bound"。

以上完全基于 SOL，无需一次 kernel 实测。真实 kernel 只能达到该上界的一部分；后两节讨论如何将其修正到实测可达吞吐——但这不改变 SOL 作为跨模型、跨方案对比基准的地位，实测只是对这一上界的修正。

### 2.4 Shape 系统：推导每张卡的真实形状

要把 SOL 上界修正到真实实现，第一步是知道每张卡上每个 op 实际执行的形状——这需要一个可靠的 shape 系统。它与章 1 的 shape 推断复用同一套机制：sharding 被建模为计算图上的一次**幂等变换**，per-rank shape 作为变换后计算图的**下游派生**得到，而非对全量 shape 的独立换算。

具体地，一个合法部署方案（`ParallelConfig`）只改变引擎输入的原子 model dim——attention head `N ÷ tp_attention`、dense intermediate `I ÷ tp_dense_ffn`、expert intermediate `Im ÷ tp_moe_expert`，EP 则将 expert 数量减为 `E / ep` 而不改变任何 dim——随后以这组 dim 重新运行章 1 的 `compute_model`。由于每个 op 的 shape 本就由其 `shape_kind` 规则从输入 dim 派生（例如 `out = N·(D + P)` 随 `N` 自动重算），切分后整棵计算图的 per-rank 形状即一致地派生得到，无需为任意 TP / EP / CP 组合单独编写推导逻辑。这是"单一 shape 底座 + 幂等图变换"的核心价值：形状的正确性由派生规则保证，而非依赖对每种并行组合的专门处理。

如此得到的形状即为各设备上真实执行的规模——例如 DeepSeek-V2-Lite 在 tp_ep8 下 grouped_gemm 的 `K` 为 1408，可直接作为下节实测的输入。它与章 1 的 Shape inference 用途互补：章 1 在构建 spec 时以其做几何一致性校验，此处以其将一个部署方案正向推导为各设备的真实 kernel 形状。placement 代数的细节见 [`docs/parallelism.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/parallelism.md)。

一个可靠的 shape 系统还带来一个关键收益：由于它能从部署方案正向推导出**分布式部署下每张卡的真实 kernel 形状**，只需在**单张卡**上以该形状 profile 一个算子，即可得到通常要真正搭建大规模分布式集群才能测到的 kernel 性能。换言之，用最轻量的单卡实测覆盖分布式部署的算子成本——这也是下一节实测的前提。

### 2.5 基于实测的算子库驱动

有了 2.4 正向推导出的 per-rank shape，就能把 2.3 的 SOL 上界修正到实测可达吞吐，填补理论上界与真实实现之间的差距：实测 `t_measured` 与理论 `t_sol` 并列、不覆盖理论值。

以真实 per-rank shape 为输入，通过 **adapter 对接开源算子库**进行实测：每个库（FlashAttention 2/3、FlashInfer、FlashMLA、DeepGEMM、FLA、Mamba-SSM、vLLM、sglang 等）注册一个 provider，将 canonical 输入适配到各自的 layout 并执行真实 kernel；`kernel_bench` 的 driver 以一个 torch-eager 实现为参考、按 dtype-scaled tolerance 做数值 cross-check，结果不一致的 kernel 不予采纳。取跨库最快的正确 kernel 作为该 op 的实测包络，`op_calibration` overlay 将其回填至 `t_measured` 轨道。语料分为两层：**共享语料**是部署的公开基线（物化至磁盘，core 仅读取磁盘、不访问数据库），**私有上传**是登录用户的私有行，在每次估算时即时叠加以校准该用户自身的估算、无需公开。上述"部署方案 → per-rank 真实 shape → adapter 实测"的机制、库与 kernel 的支持矩阵及两层语料的细节见 [`docs/measured_calibration.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/measured_calibration.md)，该形状推导所依赖的 placement 代数见 [`docs/parallelism.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/parallelism.md)。

---

## Roadmap

HuggingArch 仍是一个持续演进的脚手架，接下来几个方向：

- **更精确的 inference 建模**：把 speculative decoding、EP balance（专家负载均衡）等对真实吞吐有实质影响的机制纳入模型，并以通信实测值替代纯理论 comm 时间，让估算更贴近真实部署。
- **更厚的实测语料**：扩展 measured corpus 覆盖的 GPU 型号、算子库与模型——每多一条实测，SOL 上界就多一分收窄到可达吞吐，这一层最适合社区共建。
- **从 spec 分析转向 model design**：从"读懂已发布模型"扩展到自定义模型（custom model）的算账，支持架构设计期的成本探索与对比。
- **训练成本建模**：将成本建模从推理扩展到训练。
- **spec 系统的抽象演进**：随着 DeepSeek-V4-Pro 这类模型把结构复杂度推高，将目前仍作特殊情形处理的 side-branch、dynamic-axis、dim-family 类型等升级为一等抽象（详见 [`docs/future_work.md`](https://github.com/shenh10/HuggingArch/blob/main/docs/future_work.md)）。

## 如何 Contribute

这是一个 side-project 性质的开放项目，欢迎 star，也欢迎成为 developer 一起共建。参与方式：

- **贡献模型 spec**：为尚未支持的模型运行 agentic 生成流程，spec 通过全套校验（`bytes_diff == 0`）后即可 nominate；review 通过后由 `promote` 并入 component library。在个人资料里填上 GitHub handle，贡献的 spec 会在 PR 中以 co-authored-by 署名。
- **贡献实测语料**：在自己的 GPU 上运行 `kernel_bench` 并上传 measured CSV。这些数据即时校准你自己的估算（无需公开），也可以 nominate 进共享语料、经 review 后成为所有人可见的公开基线。
- **贡献 component 与 fusion binding**：新的拓扑 component，或从 vLLM / sglang 源码推导出的新框架 fusion plan，都走同一套 draft → review → promote 流程。
- **报告问题、参与讨论**：任何一个数值对不上、或某个新架构尚未支持，都欢迎提 issue / PR。

上手路径与代码结构见仓库的 `docs/`（`README.md` 是分层地图），机制细节见各篇 deep-dive。

## 后记：让 agent 算一篇 DeepSeek V4-Pro，翻车了——这才是这个项目的意义

在 DeepSeek 发了 V4-Pro 之后我做了一个分析实验，这次我没自己动手，而是让一个 agent 写了一篇 V3 / V3.2 / V4-Pro 在 8×B200 上的部署算账——prefill、decode、max-batch，上下文从 4K sweep 到 1M。Agent 干得有模有样：架构差异梳理得很细（CSA / HCA 交错、每层 SWA 分支、shared-KV MQA、FP4 MoE、mHC……），定性结论也站得住（V4 把累积 KV 砍到 V3 的 ~10%，单机长上下文高并发是它的设计靶子），通篇 roofline 公式、speedup 倍数、一张张 sweep 表，看起来专业得不行。

然后我把这篇文档丢进 HuggingArch 对了一遍账。结果很能说明问题：**定性全对，定量翻车。**

最典型的一处是 prefill。文档里 V3 @1M 写的是 73.48 µs/token，而 HuggingArch 按第一性原理重算（128 query head × qk192/v128 × 61 层 × causal½，8×B200 BF16 聚合）是 ~150——**agent 算的时候漏乘了一个 ×2**。这种错单看那一格根本发现不了，73 和 147 都"看起来挺合理"，agent 自己回头读也读不出哪里不对。但它是 baseline，一错就污染一整列相对值：因为 V3 被低估了一半，V4 相对 V3 的 prefill 加速被写成了 4.7×，**真值其实是 9.4×**——agent 把这个模型最该吹的卖点，说小了一倍。

更隐蔽的是短上下文。文档默认 prefill 是 compute-bound，纯按 FLOPs/peak 算，于是 4K 给了个 ~1 µs/token 的数。但单流（B=1）短上下文 prefill 其实是 **weight-bound** 的——一堆 MoE 专家权重得从 HBM 搬上来，这个地板被整列忽略了。这不是抄错数字，是**建模假设本身在某个 regime 悄悄失效**——这种错，最难靠"再读一遍"看出来。

这就是开头那句"口嗨一个 xx 倍很容易，但禁不住考验"的活体标本，只不过这回口嗨的是 agent。而且要命的恰恰在这里：让 agent 自由发挥地算账，它产出的东西**看起来比人写的还专业**——公式、表格、倍数一应俱全，置信度拉满，但里面藏着的 ×2 和失效假设，它自己完全没有能力发现。你也别指望"再让 agent review 一遍"能兜住——幻觉 review 幻觉，只会越看越觉得对。

而且这不是孤例的观感。写完那篇之后，我们把它做成了一个受控实验：让 8 个不同的 coding agent 写同一批模型的 spec，再把 ground-truth 校验从全关逐档装回，看每一步产出怎么变。结果很干净——**裸奔（不带任何 guard）时，agent 有 94% 的概率把一份其实错了的 spec 自判为"通过"**：它本地 validate 是绿的，因为能抓错的那把尺子被拿掉了；把 guard 全部装回后，这个"自信地错"的比例掉到 6%。失败几乎全落在钉死于物理事实的检查上（权重字节、单 token KV、逐层调度），且高度集中在最难的那个 V4-Pro——再次说明：agent 会不会算错不是关键，错了有没有 ground truth 接得住才是。

这正是 HuggingArch 整个设计的出发点：**不是不让 agent 算账，而是不让 agent 在没有 ground truth 的真空里算账。** 同样一个 V4-Pro，在 HuggingArch 里 agent 写的不是一篇自由发挥的 Markdown，而是一份受约束的 spec——每个 op 的 FLOPs / 参数量公式是 built-in 可信的，每个数字都钉死在 `config.json` / safetensors / forward 源码上。那个漏掉的 ×2，要么根本写不出来，要么立刻和第一性原理、和真实 tensor 字节数对不上而被反馈打回。Agent 不需要"恰好算对"，它只需要踩到 ground truth 的反馈就修——这跟它撒开手写一篇算账，是两种完全不同的可靠性等级。

> 顺带一提，对账是双向的：这一轮也把 HuggingArch 自己的一个 bug 揪了出来。V4 那种"一层里同时缓存一条滑窗 KV + 一条时间压缩 KV、两条基数不同"的拓扑，工具早先把压缩分支也按滑窗 W=128 封了顶，导致 max-batch 在 4K→1M 全程纹丝不动——明显违反物理。能发现它，靠的也是同一套纪律：每个数字都得能跟 spec 里真实声明 `kv_cached` 的节点、跟官方报告的"~10% KV"对得上，对不上就一定哪里错了。修法也守着项目的规矩——不给 V4 打 `if`，而是补一层"每个被缓存节点各自带存储/读取基数"的通用抽象（顺带修对了 V3.2 indexer 全长 scan 的口径）。**结论是：agent 也好、人脑也好、连工具本身也好，只要在建模就会犯错；唯一的区别是错了有没有 ground truth 接得住。**

所以 HuggingArch 想解决的，从来不只是"省得每个新模型手动建一遍模型"的体力问题，更是"agent 算出来的账到底可不可信"的信任问题。Agentic coding 的时代，让模型生成一篇分析太容易了；难的是让它生成**可信**的分析。一个把公式摊开、把假设写明、把每个数字都钉死在事实源上的系统——才敢让你把一句"V4 比 V3 快 9.4×"放心地发出去，也才敢在 agent 下次一本正经地口嗨时，红着脸把它拦下来。
