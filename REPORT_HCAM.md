# HCAM 100M 技术报告 / Technical Report

**HCAM = Hybrid Convolution-Attention Model / 混合卷积-注意力模型**

---

## 中文

### 摘要

HCAM 100M 是一个面向中文文本的双向掩码语言模型（MLM）。模型没有采用“每层全局自注意力”的纯 Transformer 编码器，而是使用 **7 个双向门控扩张卷积局部层 + 3 个双向 Gated-GQA 全局注意力层**构成 10 层混合编码器。设计目标是在中文高频局部词法和短语模式上使用成本较低的局部混合，同时保留少量全局注意力用于远距离信息交互。

模型使用 **18,000 个字符 token**作为真正输入输出词表，隐藏维度 1056，上下文长度 600，总计 **102,778,220 个唯一可训练参数**。SentencePiece 不参与模型 token ID 编码，只在预训练阶段为整组和连续跨度掩码提供词/子词边界指导。

预训练语料约包含 **1.638B 个有效不重复字符 token**。训练被划分为 **10 个 virtual epochs**：前 5 个 virtual epochs 合计完成第一遍完整语料，后 5 个使用半窗口 offset 并合计完成第二遍完整语料。因此总训练暴露量约为 **3.276B character-token exposures**。virtual epoch 只是训练调度与 masking/LR 阶段单位，并不等同于一次完整语料遍历。最终固定验证 MLM loss 为 **1.9900**、masked accuracy 为 **60.76%**。在额外真实文本基准中，HCAM 的 REAL-MLM Top-5 达到 **88.38%**，与 Chinese-RoBERTa-WWM-ext 的 88.50% 接近；REAL-LONG 中远距离 Cue 帮助率达到 **95.00%**。这些结果表明，即使多数层使用门控扩张卷积而非全局自注意力，HCAM 仍然建立了明确的普通 MLM 与数百字级远距离上下文利用能力。

此外，本报告记录一个小型“的/地/得”纠错微调实验，用于说明 HCAM 作为下游中文编码器基座的可适配性；该实验不是本次开源的核心任务。

### 项目缘起

HCAM 100M 的起点并不是“为了超过某个现有模型”，而是作者对小型语言模型架构和从零训练过程的持续兴趣。在 HCAM 100M 之前，同一条混合卷积-注意力架构思路已经先后被用于 16M、30M、55M 和 100M 规模的自回归语言模型实验。

在完成这些实验后，作者希望验证同一类结构在双向掩码语言建模中的表现，因此重新设计了预训练目标、字符级 tokenizer、动态多粒度 masking 和训练流程，并从零训练了 HCAM 100M。项目最初没有预设下游目标，也没有计划进行移动端部署；后续在“的/地/得”纠错任务上的微调结果超出最初预期，才进一步发展出量化和 Android 端部署。这个从架构实验到真实应用的演进过程，也是本次公开预训练基座和完整技术资料的主要动机之一。

### 1. 架构

| 项目 | 配置 |
|---|---:|
| 参数量 | **102,778,220** |
| 字符词表 | 18,000 |
| 上下文长度 | 600 |
| 隐藏维度 | 1056 |
| 总层数 | 10 |
| 双向门控扩张卷积层 | 7 |
| 双向 Gated-GQA | 3 |
| GQA 位置 | 第 3 / 7 / 10 层 |
| Query / KV heads | 16 / 4 |
| Head dim | 66 |
| 局部内部维度 | 1600 |
| GQA FFN 隐藏维度 | 2944 |
| Norm | RMSNorm |
| FFN | SwiGLU |
| Position | RoPE |
| Dropout | 0.10 |
| 输入/输出权重共享 | 是 |

```text
Token Embedding + learned Mask Embedding
          │
          ▼
  1  Gated Dilated Conv   k=3 d=1
  2  Gated Dilated Conv   k=3 d=2
  3  Gated GQA
  4  Gated Dilated Conv   k=5 d=2
  5  Gated Dilated Conv   k=3 d=1
  6  Gated Dilated Conv   k=3 d=2
  7  Gated GQA
  8  Gated Dilated Conv   k=5 d=2
  9  Gated Dilated Conv   k=3 d=1
 10  Gated GQA
          │
       RMSNorm
          │
 Tied 18K Character LM Head
```

#### 1.1 双向门控扩张卷积层

局部层先执行 RMSNorm，再由无偏置线性映射产生 `content`、`activation_gate` 和 `modulation_gate` 三个分支。`content` 进入 depthwise dilated Conv1d，并使用对称 padding，因此可以同时利用目标位置左右上下文。核心混合为：

```text
SiLU(dilated_conv(content))
× SiLU(activation_gate)
× sigmoid(modulation_gate)
```

随后投影回 1056 维并执行残差连接。7 个局部层按 `(kernel=3,dilation=1)`、`(3,2)`、`(5,2)` 循环使用。

#### 1.2 Gated-GQA 全局层

三个全局层使用双向 Grouped-Query Attention：16 个 Query heads、4 个 KV heads、每头 66 维，Q/K 在注意力前分别进行 RMSNorm，并使用 RoPE。注意力为非因果模式。SDPA 输出在最终输出投影前乘以逐 token、逐 head 的 sigmoid gate。每个 GQA 层后跟 RMSNorm + SwiGLU FFN，FFN 隐藏维度为 2944。

#### 1.3 Mask embedding 与 tied head

`[MASK]` 不占用字符词表 ID，而使用单独的可学习 mask embedding。最终 18K LM head 与输入 embedding 共享权重。模型前向还支持只对 `prediction_indices` 指定的位置生成词表 logits，以减少 MLM 中无关位置的输出计算。

### 2. 字符 tokenizer 与多粒度遮蔽

真实模型 token 为字符级。词表包含 `<unk> <pad> <bos> <eos> <用户> <助手> <换行> <段落>` 8 个固定特殊 token，其余 17,992 个位置为普通 Unicode 字符。极罕见、未进入 18K 词表的字符回退到 `<unk>`。

SentencePiece 只作为边界指导：识别字符组、提供整组 mask 边界、提供连续多个组的跨度 mask 边界。SentencePiece piece ID 不作为 HCAM 模型输入。

预训练总遮蔽率由第 1 轮约 30% 逐步下降到第 10 轮 15%。被选中的目标位置使用 80/10/10 替换：80% learned mask embedding、10% 随机字符、10% 保留原字符。

| 轮次 | 字符 | 整组 | 相邻组跨度 |
|---|---:|---:|---:|
| 1–3 | 50.00% | 37.50% | 12.50% |
| 4–6 | 45.00% | 41.25% | 13.75% |
| 7–9 | 40.00% | 45.00% | 15.00% |
| 10 | **25.00%** | **50.00%** | **25.00%** |

### 3. 预训练数据与优化

预训练有效不重复语料规模约为 **1.638B 字符**。完整训练实际包含 **2 次全语料遍历**，但为了动态 masking、学习率调度、验证和断点续训，被进一步切分为 10 个 virtual epochs：第 1–5 个 virtual epochs 合起来覆盖第一遍完整语料；第 6–10 个使用半窗口 offset，合起来覆盖第二遍完整语料。因此总训练暴露量约为 **3.276B 字符 token**。固定验证集约 1M 字符 token。

目标数据混合比例：Ultra-FineWeb 中文 ≤50%、SkyPile-150B 中文 32%、Wikipedia 14%、自有/用户语料 4%。当 Ultra-FineWeb 或 Wikipedia 不足目标量时，缺口由 SkyPile 回补，而不是重复较早文本。仓库不重新分发原始训练语料；详见 `DATA_LICENSES.md`。

| 优化配置 | 数值 |
|---|---:|
| Optimizer | AdamW |
| Peak LR | 7e-4 |
| Warmup | 总训练 token 的 4% |
| LR schedule | token-based cosine |
| 常规最低比例 | peak LR × 0.08 |
| 第 10 轮 LR floor | 1e-4 |
| Weight decay | 0.08 |
| Z-loss | 1e-4 |
| Gradient clip | 1.0 |
| Effective global batch | 384 |

训练使用自动混合精度、PyTorch SDPA，并支持双 GPU DDP；主要训练环境为 Kaggle T4×2。由于 Kaggle 单次 GPU 会话时长受限，完整 10 轮预训练并不是在一次连续会话中完成，而是跨多次、跨数天断点续训完成。训练过程在安全边界保存 `last.pt`，并在下一次会话恢复模型参数、AdamW 优化器状态、GradScaler、轮次/步数、学习率相关状态以及训练签名；续训包还进行 CRC / SHA256 等完整性校验，以尽量保证会话切换前后的训练连续性。换言之，报告中的 10 轮代表同一训练轨迹的连续恢复，而不是每天重新开始训练。

### 4. 预训练结果

第 10 轮结束时固定验证集：

| 指标 | 结果 |
|---|---:|
| MLM Loss | **1.9900** |
| Masked Accuracy | **60.76%** |
| Masked PPL | **7.32** |

训练结束后还运行两套零微调 MLM 基准：

| 基准 | Loss | PPL | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| 标准混合遮蔽 | 2.0044 | 7.42 | 60.58% | 75.50% |
| 词/跨度压力测试 | 3.0540 | 21.20 | 43.87% | 59.49% |

这里的 PPL 是 masked cross-entropy 的指数，只适合同一词表和同一评测定义下观察，不应直接与不同 tokenizer 的 BERT/RoBERTa/MacBERT PPL 横向比较。

#### 4.1 真实文本 MLM、稀有跨度与长距离依赖测试

为了进一步观察模型在真实文本中的掩码恢复以及远距离上下文利用能力，我额外构造了 REAL-MLM、REAL-RARE-SPAN 和 REAL-LONG 三组测试，并使用 Chinese-RoBERTa-WWM-ext 与 RoFormerV2-Char-Base 作为参考基线。

| 模型 | 参数量 | REAL-MLM Top-1 | REAL-MLM Top-5 | Rare 字符 Top-1 | Rare 整词恢复 |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 102.78M | **68.50%** | **88.38%** | 9.79% | 1.33% |
| Chinese-RoBERTa-WWM-ext | 102.29M | 74.00% | 88.50% | 19.71% | 4.33% |
| RoFormerV2-Char-Base | 94.18M | 38.38% | 61.00% | 17.02% | 4.67% |

HCAM 的 REAL-MLM Top-5 为 **88.38%**，与 Chinese-RoBERTa-WWM-ext 的 **88.50%** 仅相差 0.12 个百分点。虽然 HCAM 的 Top-1、Rare Span 和整词恢复仍有提升空间，但这一结果说明，在普通真实文本 MLM 中，HCAM 已经能够较稳定地把正确答案放入高概率候选集合。

REAL-LONG 则对远距离实体线索进行删除消融。四个距离桶（64–469 字）简单平均结果如下：

| 模型 | 完整 Top-1 | 去 Cue Top-1 | 完整整词 | Cue 帮助率 | 平均 ΔlogP |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 46.54% | 13.32% | 32.50% | **95.00%** | **+2.467** |
| Chinese-RoBERTa-WWM-ext | 58.54% | 22.87% | 45.63% | 96.25% | +2.296 |
| RoFormerV2-Char-Base | 60.09% | 20.55% | 45.00% | 95.63% | +2.574 |

在最远的 **384–469 字**距离桶中，HCAM 保留远端 Cue 时 Top-1 为 **42.45%**，删除 Cue 后降至 **10.38%**，平均 `ΔlogP=+2.441`。这说明三层全局 Gated-GQA 已经能够让数百字外的信息稳定进入并影响目标位置表示。HCAM 当前的主要差距并不是“长距离信息无法传递”。

从结构上看，HCAM 的信息流大致是：

```text
local → local → GLOBAL
→ local → local → local → GLOBAL
→ local → local → GLOBAL
```

一次全局注意力将远端信息写入 hidden state 后，这些信息会通过残差连接继续保留，并在后续局部门控扩张卷积中被加工；下一次全局层再进行新的全序列信息交换。因此，“只有三层全局注意力”并不等于“只有三层拥有全局信息”。REAL-LONG 的 Cue 消融为这一点提供了直接实验证据。

这里必须强调训练资源并不匹配。HCAM 是从零训练模型，所使用的**有效不重复语料规模约为 1.638B 字符**；10 个 virtual epochs 实际只对应 **2 次完整语料遍历**，总训练暴露量约为 **3.276B character-token exposures**。HFL 官方资料记录 Chinese-RoBERTa-WWM-ext 所用 EXT 语料总词数约 **5.4B**；RoFormerV2 官方则记录约 **280GB 无监督数据**，并追加约 **20GB、77 个标注数据集构造的 92 个任务**进行有监督多任务训练。字符、词与 GB 不能直接一一换算，因此这里不声明一个不严谨的精确倍数，但公开训练资源的规模显然不在同一水平。

**在明显更小的训练数据与个人算力预算下，HCAM 仍在普通真实文本 MLM 和长距离上下文利用上达到了相对于这些成熟基线并不弱、部分指标处于同一量级的能力。** REAL-MLM Top-5 与 Chinese-RoBERTa-WWM-ext 几乎持平，而 REAL-LONG 的 Cue 帮助率也与两个成熟基线接近。

Rare Span 与最终精确恢复率仍有差距，但这类能力高度依赖长尾词、专名与低频组合的覆盖次数，因此其中相当一部分差距很可能来自训练数据规模、数据覆盖度和训练预算，而不能直接归因于 HCAM 的混合结构。

结合此前自回归缩放实验、当前 MLM 结果以及 REAL-LONG 的 Cue 消融，**目前没有发现 HCAM 相比纯 Transformer 存在可明确归因于混合架构本身的系统性性能差异或明显精度损失。** 现有结果没有显示“用门控扩张卷积替换大部分全局注意力层”本身造成了明显能力退化。严格验证架构差异仍需要未来进行同数据、同参数量、同训练步数的控制实验。

完整测试方法、RoFormerV2 健康检查、各距离桶结果与更详细解释见 [`REAL_BENCHMARK.md`](REAL_BENCHMARK.md)。

### 5. 小型下游微调：中文“的/地/得”纠错

该实验来自 Dededi 中文纠错项目，只作为基座下游适配示例。每个待判断的“的/地/得”位置在进入模型时被替换为 learned mask embedding，让 HCAM 根据左右上下文恢复正确类别。三类在 18K 词表中的 ID 为：`的=138`、`地=164`、`得=243`。部署时可以直接从 tied embedding 抽取这三行形成 3×1056 分类 head，在 FP32 下与完整 LM head 对这三个类别的 logits 数学等价。

第一版实际部署模型采用第 3 轮 EMA 权重（E3-EMA）。当时的内部微调数据没有整理成可公开复现的冻结训练快照，因此本报告不声明一个无法核验的精确训练样本数。可以确认训练样本覆盖真实“的/地/得”用法、固定搭配、专名、方式状语、程度补语、质地/性质结构以及误改困难样本增强。

主要对比结果：

| 测试集 | 模型 | Macro-F1 | Macro-F0.5 | 其他 |
|---|---|---:|---:|---:|
| Fresh Hard, 120句 / 383位 | HCAM E3-EMA | 97.42% | **97.63%** | Target Acc. 97.39% |
| Fresh Hard | RoFormerV2 | **97.44%** | 97.46% | Target Acc. 97.39% |
| Mixed Total, 720句 / 2,275位 | **HCAM E3-EMA** | **98.17%** | **98.24%** | Target Acc. **98.42%** |
| Mixed Total | RoFormerV2 | 95.85% | 96.26% | Target Acc. 96.53% |
| External DEV, 1,600句 / 3,025位 | **HCAM E3-EMA** | **97.496%** | **97.270%** | Clean false action **0.628%** |
| External DEV | HCAM Base | 95.456% | 96.416% | Clean false action 0.893% |
| External DEV | RoFormerV2 E3-EMA | 95.806% | 95.437% | Clean false action 0.826% |
| Natural DEV3, 1,200段 / 2,734位 | **HCAM E3-EMA** | **95.148%** | **95.882%** | Clean false action **0.988%** |
| Natural DEV3 | RoFormerV2 E3-EMA | 83.205% | 84.774% | Clean false action 3.548% |

External DEV 中 HCAM E3-EMA 的分类别 F1：的 99.143%、地 96.690%、得 96.654%。Natural DEV3 由 DRCD-dev 和 Wikipedia 各 600 条组成，目标分布为 的 2268 / 地 333 / 得 133，冻结 SHA256 为 `4a0d29f320dcf4ebb46a8ae73de61f372ea5201045918733713e75dcdd700487`。

不同测试集之间分数变化较大，说明“的/地/得”任务对句式和数据来源非常敏感。曾有一套人工模板比例较高的专项压力集明显偏向 RoFormerV2，因此本报告不以单一集合给出通用架构优劣结论，也不把该下游实验外推为通用中文能力排名。

### 5.1 实际应用部署示例

除受控评测外，HCAM E3-EMA 还被集成到作者的 Android 中文纠错应用中，用于真实文本环境下的交互式测试。README 中展示了一次实际设备运行截图：在“专业（HCAM）”模式下，应用对约 1,187 字文本给出 3 条“的/地/得”纠错建议，并显示对应置信度；该次运行界面记录的耗时为 0.44 秒。截图中的实际端侧模型是 **ExecuTorch/XNNPACK 量化部署版本**，而不是 FP32 预训练基座；部署主干使用 dynamic INT8 per-channel 量化，最终三分类 head 保持 FP32。

这一截图用于说明 HCAM 下游模型已经完成端侧应用集成，而不是作为独立 benchmark。实际速度会受到设备、文本长度、运行时状态和部署实现等因素影响，因此界面中的单次耗时不应视为所有设备上的固定推理速度。本仓库的主要开源对象仍然是 **HCAM 100M 预训练基座**，Dededi 微调模型与 Android 应用仅作为下游适配案例。

### 6. 已知限制

1. HCAM 是双向 MLM 编码器，不是自回归聊天模型。
2. 上下文长度为 600，不属于长上下文模型。
3. 18K 高频字符词表无法覆盖全部 Unicode，极罕见字符会回退到 `<unk>`。
4. 模型只有三层全局注意力，但 REAL-LONG 已确认 **400+ 字远端 Cue 能稳定影响预测**；是否会限制更复杂的多轮全局交互，仍需要同数据控制实验进一步验证。
5. 稀有词与连续跨度恢复仍弱于成熟大语料基线；由于基线训练资源显著更大且训练流程不同，目前不能把这一差距直接归因于 HCAM 架构。
6. 下游“的/地/得”实验不等于 CLUE、NER、阅读理解等通用中文 benchmark。
7. 训练数据来自多个上游来源，各自许可不同；本项目不重新分发原始数据，模型权重许可也不能消除第三方上游条款。

---

## English

### Abstract

HCAM 100M is a bidirectional Chinese masked language model built around a **hybrid convolution-attention encoder** rather than a full-attention Transformer stack. Its 10 layers contain **7 bidirectional gated dilated-convolution local mixers and 3 bidirectional Gated-GQA global-attention blocks**. The design shifts most high-frequency lexical and phrase modeling to local operators while retaining sparse global interactions for longer-range context.

The model uses a true **18,000-token character vocabulary**, hidden size 1056, context length 600, and contains **102,778,220 unique trainable parameters**. SentencePiece IDs are never fed to the model; SentencePiece is used only as a word/subword-boundary guide when constructing whole-group and adjacent-group span masks during pretraining.

Pretraining uses approximately **1.638B effective non-duplicate character tokens**. Training is divided into **10 virtual epochs**, but they represent only **two full corpus passes**: virtual epochs 1–5 together cover the first pass, while 6–10 use a half-window offset and together cover the second. Total training exposure is therefore about **3.276B character tokens**. A virtual epoch is a scheduling segment, not a full corpus traversal. The final fixed validation MLM loss is **1.9900** with **60.76% masked accuracy**. On the additional real-text benchmark, HCAM reaches **88.38% REAL-MLM Top-5** and a **95.00% long-range cue-help rate**, providing direct evidence that the hybrid encoder retains strong ordinary MLM and long-distance context utilization despite using global attention in only three layers.

A small Chinese 的/地/得 correction fine-tuning experiment is also reported as a downstream case study, but it is not the primary objective of this release.

### Motivation

HCAM 100M did not begin as an attempt to outperform a particular existing model. It grew out of the author's ongoing interest in small language-model architectures and training models from scratch. Before HCAM 100M, the same hybrid convolution-attention line had already been explored through autoregressive models at 16M, 30M, 55M, and 100M scales.

After those experiments, the author wanted to test how the same type of architecture would behave under bidirectional masked language modeling. This led to a redesigned pretraining objective, character-level tokenizer, dynamic multi-granularity masking strategy, and training pipeline, and ultimately to HCAM 100M. The project initially had no predefined downstream target and no plan for mobile deployment; quantization and Android deployment only followed after the 的/地/得 fine-tuning results exceeded the author's initial expectations. This progression from architecture experimentation to practical deployment is one of the motivations for releasing the pretrained base and its technical documentation.

### 1. Architecture

| Item | Value |
|---|---:|
| Parameters | **102,778,220** |
| Character vocabulary | 18,000 |
| Context length | 600 |
| Hidden size | 1056 |
| Total layers | 10 |
| Bidirectional gated dilated-conv layers | 7 |
| Bidirectional Gated-GQA layers | 3 |
| Global-attention layers | 3 / 7 / 10 |
| Query / KV heads | 16 / 4 |
| Head dim | 66 |
| Local inner dim | 1600 |
| GQA FFN dim | 2944 |
| Norm / FFN / position | RMSNorm / SwiGLU / RoPE |
| Dropout | 0.10 |
| Input/output weight tying | Yes |

The local mixer applies RMSNorm, projects into content/activation-gate/modulation-gate branches, and computes:

```text
SiLU(dilated_conv(content))
× SiLU(activation_gate)
× sigmoid(modulation_gate)
```

The depthwise convolution uses symmetric padding, making it bidirectional. Local layers cycle through `(k=3,d=1)`, `(k=3,d=2)`, and `(k=5,d=2)`.

Each Gated-GQA block uses 16 query heads, 4 KV heads, 66 dimensions per head, Q/K RMSNorm, RoPE, non-causal SDPA, and a token-wise/head-wise sigmoid gate on the attention output before the output projection. It is followed by RMSNorm + SwiGLU with a 2944-dimensional hidden layer.

The learned mask embedding is separate from the 18K character IDs. The final LM head is tied to the input embedding. The forward API supports `prediction_indices`, allowing the model to compute vocabulary logits only at selected MLM positions.

### 2. Character tokenizer and masking

The runtime tokenizer is character-level. Eight fixed special tokens are reserved: `<unk> <pad> <bos> <eos> <用户> <助手> <换行> <段落>`. The remaining 17,992 entries are ordinary Unicode characters; very rare unseen characters fall back to `<unk>`.

SentencePiece is only a boundary guide. It defines whole-piece and adjacent-piece masking spans but its piece IDs are not model inputs.

The overall mask ratio anneals from about 30% in epoch 1 to 15% in epoch 10. Selected targets use an 80/10/10 corruption policy: learned mask embedding / random character / unchanged character.

| Epochs | Character | Whole group | Adjacent-group span |
|---|---:|---:|---:|
| 1–3 | 50.00% | 37.50% | 12.50% |
| 4–6 | 45.00% | 41.25% | 13.75% |
| 7–9 | 40.00% | 45.00% | 15.00% |
| 10 | **25.00%** | **50.00%** | **25.00%** |

### 3. Pretraining data and optimization

The effective non-duplicate training corpus contains about **1.638B characters**. The full run makes **two complete passes** over the corpus, split into 10 virtual epochs for masking/LR scheduling, validation, and checkpointing. Virtual epochs 1–5 jointly cover the first pass; virtual epochs 6–10 use a half-window offset and jointly cover the second. Total training exposure is therefore about **3.276B character tokens**. The fixed validation stream contains about 1M character tokens.

The target source mixture is ≤50% Ultra-FineWeb Chinese, 32% SkyPile-150B Chinese, 14% Wikipedia, and 4% custom/user corpus. Shortfalls in Ultra-FineWeb or Wikipedia were filled from SkyPile instead of repeating earlier text. Raw corpora are not redistributed; see `DATA_LICENSES.md`.

Optimization uses AdamW, peak LR `7e-4`, 4% token-based warmup, cosine decay, weight decay 0.08, z-loss `1e-4`, gradient clipping 1.0, and effective global batch 384. The final epoch uses an LR floor of `1e-4`. Training primarily ran on Kaggle T4×2 with AMP, PyTorch SDPA, and DDP support. Because Kaggle GPU sessions are time-limited, the full 10-epoch pretraining run was completed across multiple sessions over several days rather than in one uninterrupted job. Checkpoints preserved the model state, AdamW state, GradScaler state, epoch/step position, LR-related progress and training signature, and the resume packages were integrity-checked with CRC/SHA256 before continuing. The reported 10 epochs therefore represent one continuously resumed training trajectory, not repeated restarts from scratch.

### 4. Pretraining results

| Evaluation | Loss | Masked PPL | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| Final fixed validation | **1.9900** | **7.32** | **60.76%** | — |
| Standard mixed-masking benchmark | 2.0044 | 7.42 | 60.58% | 75.50% |
| Whole-piece/span stress benchmark | 3.0540 | 21.20 | 43.87% | 59.49% |

Masked PPL is `exp(masked cross-entropy)` and is meaningful only under compatible tokenizer and masking definitions.

### 4.1 Real-text MLM, rare-span and long-range evaluation

The additional benchmark uses REAL-MLM, REAL-RARE-SPAN and REAL-LONG, with Chinese-RoBERTa-WWM-ext and RoFormerV2-Char-Base as reference baselines.

| Model | Params | REAL-MLM Top-1 | REAL-MLM Top-5 | Rare char Top-1 | Rare whole-span |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 102.78M | **68.50%** | **88.38%** | 9.79% | 1.33% |
| Chinese-RoBERTa-WWM-ext | 102.29M | 74.00% | 88.50% | 19.71% | 4.33% |
| RoFormerV2-Char-Base | 94.18M | 38.38% | 61.00% | 17.02% | 4.67% |

HCAM reaches **88.38% Top-5** on REAL-MLM, only **0.12 percentage points** below Chinese-RoBERTa-WWM-ext at 88.50%.

REAL-LONG removes distant entity cues and re-evaluates the same prediction. Averaged across the four distance buckets (64–469 Chinese characters):

| Model | Full Top-1 | No-cue Top-1 | Whole-span | Cue help rate | Mean ΔlogP |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 46.54% | 13.32% | 32.50% | **95.00%** | **+2.467** |
| Chinese-RoBERTa-WWM-ext | 58.54% | 22.87% | 45.63% | 96.25% | +2.296 |
| RoFormerV2-Char-Base | 60.09% | 20.55% | 45.00% | 95.63% | +2.574 |

In the **384–469-character** bucket, removing the distant cue reduces HCAM Top-1 from **42.45% to 10.38%**, with mean `ΔlogP=+2.441`. This is direct evidence that distant context is genuinely used rather than merely available in principle.

The hybrid encoder alternates local processing and global communication. Once a Gated-GQA block writes distant information into a token representation, residual connections allow it to persist and local convolutional layers can refine that representation before the next global exchange. Therefore, having only three global-attention layers does not mean that global information exists only inside those three layers.

These comparisons are **not** matched-data architectural ablations. HCAM was trained from scratch on about **1.638B effective non-duplicate characters** and made **two full corpus passes**, split into 10 virtual epochs, for about **3.276B character-token exposures** in total. HFL documents roughly **5.4B words** for the EXT corpus, while the RoFormerV2 authors report roughly **280GB of unsupervised data** followed by about **20GB of supervised multi-task data** built from 77 labeled datasets and 92 tasks. These units are not directly interchangeable; corpus-coverage comparisons should primarily use the independent corpus scale rather than repeated-epoch exposure, and the published training resources remain clearly unmatched.

**Despite substantially smaller training data and an individual-scale compute budget, HCAM still achieves real-text MLM and long-range context utilization that are not weak relative to these mature baselines, with several metrics in the same range.** The remaining gaps, particularly on rare-span and exact recovery, are strongly confounded by long-tail corpus coverage and training scale.

Across the existing autoregressive scaling experiments, MLM evaluation, and REAL-LONG cue ablation, **no systematic performance degradation has yet been identified that can be clearly attributed to the hybrid convolution-attention architecture itself**. The current evidence does not show that replacing most attention layers with gated dilated convolutions inherently causes a meaningful loss of capability. A strict architectural conclusion would require a future matched-data, matched-parameter, matched-step ablation.

Full definitions, RoFormerV2 health checks, and per-distance results are provided in [`REAL_BENCHMARK.md`](REAL_BENCHMARK.md).

### 5. Small downstream fine-tuning case study: 的 / 地 / 得

The Dededi experiment masks every target 的/地/得 position with HCAM's learned mask embedding and predicts the correct class from bidirectional context. Their character IDs are `的=138`, `地=164`, and `得=243`. For deployment, the three corresponding tied-embedding rows can be used as a 3×1056 classifier head, mathematically matching the three relevant full-vocabulary logits in FP32.

The first deployed HCAM fine-tune used epoch-3 EMA weights (E3-EMA). The original internal training set was not frozen as a public reproducibility snapshot, so this report intentionally does not claim an unverifiable exact fine-tuning sample count. The data covered natural 的/地/得 usage, fixed expressions, proper names, manner adverbials, complement structures, texture/property constructions, and difficult false-positive cases.

Key comparisons:

- Mixed Total (720 sentences / 2,275 targets): HCAM **98.17% Macro-F1** vs RoFormerV2 95.85%.
- External DEV (1,600 sentences / 3,025 targets): HCAM **97.496% Macro-F1** vs RoFormerV2 95.806%; HCAM clean false-action rate 0.628%.
- Natural DEV3 (1,200 snippets / 2,734 targets): HCAM **95.148% Macro-F1 / 95.882% Macro-F0.5** vs RoFormerV2 83.205% / 84.774%; clean false-action 0.988% vs 3.548%.

External DEV per-class HCAM F1: 的 99.143%, 地 96.690%, 得 96.654%. Natural DEV3 contains 600 snippets from DRCD-dev and 600 from Wikipedia, with target counts 的 2268 / 地 333 / 得 133 and frozen SHA256 `4a0d29f320dcf4ebb46a8ae73de61f372ea5201045918733713e75dcdd700487`.

Scores vary substantially across distributions, and one synthetic/template-heavy stress set strongly favored RoFormerV2. Therefore these results are reported as task-specific evidence rather than a universal ranking of Chinese encoders.

### 5.1 Practical deployment example

Beyond controlled evaluation, HCAM E3-EMA has also been integrated into the author's Android Chinese correction application for interactive testing on real-world text. The README includes a real-device screenshot in which Professional (HCAM) mode processes approximately 1,187 Chinese characters and returns three 的/地/得 correction suggestions with confidence scores; the interface records 0.44 seconds for that particular run. The on-device model shown in the screenshot is an **ExecuTorch/XNNPACK quantized deployment build**, not the FP32 pretrained base; the deployed backbone uses dynamic INT8 per-channel quantization while the final three-class head remains FP32.

This screenshot is intended to demonstrate that the downstream HCAM model has been integrated into an on-device application, not to serve as a standalone benchmark. Runtime depends on device hardware, text length, runtime state, and deployment implementation, so the displayed latency should not be interpreted as a fixed speed across devices. The primary open-source artifact of this repository remains the **HCAM 100M pretrained base model**; the Dededi fine-tune and Android application are presented only as a downstream adaptation case study.

### 6. Limitations

1. HCAM is a bidirectional MLM encoder, not an autoregressive chat model.
2. Context length is 600.
3. The 18K vocabulary does not cover every Unicode character.
4. Only three layers use global attention, but REAL-LONG confirms that **400+ character distant cues reliably affect predictions**; whether this limits more complex multi-step global interaction still requires matched-data evaluation.
5. Rare-word and contiguous-span recovery remains below mature large-corpus baselines; because the training resources and training procedures are substantially unmatched, this gap cannot currently be attributed directly to the HCAM architecture.
6. The 的/地/得 case study is not a general Chinese benchmark.
7. Training sources have heterogeneous upstream licenses. This repository does not redistribute raw corpora, and the model-weight license does not erase third-party terms.

---

## Release and licensing / 发布与许可

- Repository code / 仓库代码: **Apache-2.0** + `NOTICE`
- Model weights / 模型权重: **CC BY 4.0** to the extent of the author's rights; attribution required
- Upstream data / 上游数据: see `DATA_LICENSES.md`
- Raw training corpora / 原始训练语料: **not redistributed / 不重新分发**

Suggested attribution / 建议署名:

> **HCAM 100M by qilimary** — https://github.com/qilimary/HCAM-100M-Chinese-MLM
