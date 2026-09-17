# HCAM 100M 技术报告 / Technical Report

**HCAM = Hybrid Convolution-Attention Model / 混合卷积-注意力模型**

---

## 中文

### 摘要

HCAM 100M 是一个面向中文文本的双向掩码语言模型（MLM）。模型没有采用“每层全局自注意力”的纯 Transformer 编码器，而是使用 **7 个双向门控扩张卷积局部层 + 3 个双向 Gated-GQA 全局注意力层**构成 10 层混合编码器。设计目标是在中文高频局部词法和短语模式上使用成本较低的局部混合，同时保留少量全局注意力用于远距离信息交互。

模型使用 **18,000 个字符 token**作为真正输入输出词表，隐藏维度 1056，上下文长度 600，总计 **102,778,220 个唯一可训练参数**。SentencePiece 不参与模型 token ID 编码，只在预训练阶段为整组和连续跨度掩码提供词/子词边界指导。

预训练使用约 **1.638B 个有效不重复字符 token**，通过前后两组错位窗口形成约 **3.276B 字符 token 的训练暴露量**。最终固定验证 MLM loss 为 **1.9900**、masked accuracy 为 **60.76%**。此外，本报告记录一个小型“的/地/得”纠错微调实验，用于说明 HCAM 作为下游中文编码器基座的可适配性；该实验不是本次开源的核心任务。

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

预训练有效数据规模约 1.638B 字符。前 5 轮使用第一组窗口，后 5 轮使用错位后的第二组窗口，使相同原文在不同上下文边界下再次暴露，总训练暴露量约 3.276B 字符 token。固定验证集约 1M 字符 token。

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

训练使用自动混合精度、PyTorch SDPA，并支持双 GPU DDP；主要训练环境为 Kaggle T4×2。

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

### 6. 已知限制

1. HCAM 是双向 MLM 编码器，不是自回归聊天模型。
2. 上下文长度为 600，不属于长上下文模型。
3. 18K 高频字符词表无法覆盖全部 Unicode，极罕见字符会回退到 `<unk>`。
4. 只有三层全局注意力，这是效率与密集全局交互能力之间的结构折中。
5. 词/跨度压力测试明显比标准混合遮蔽困难，连续跨度恢复仍有提升空间。
6. 下游“的/地/得”实验不等于 CLUE、NER、阅读理解等通用中文 benchmark。
7. 训练数据来自多个上游来源，各自许可不同；本项目不重新分发原始数据，模型权重许可也不能消除第三方上游条款。

---

## English

### Abstract

HCAM 100M is a bidirectional Chinese masked language model built around a **hybrid convolution-attention encoder** rather than a full-attention Transformer stack. Its 10 layers contain **7 bidirectional gated dilated-convolution local mixers and 3 bidirectional Gated-GQA global-attention blocks**. The design shifts most high-frequency lexical and phrase modeling to local operators while retaining sparse global interactions for longer-range context.

The model uses a true **18,000-token character vocabulary**, hidden size 1056, context length 600, and contains **102,778,220 unique trainable parameters**. SentencePiece IDs are never fed to the model; SentencePiece is used only as a word/subword-boundary guide when constructing whole-group and adjacent-group span masks during pretraining.

Pretraining uses approximately **1.638B effective non-duplicate character tokens** and two offset window passes, for about **3.276B character-token exposures**. The final fixed validation MLM loss is **1.9900** with **60.76% masked accuracy**. A small Chinese 的/地/得 correction fine-tuning experiment is also reported as a downstream case study, but it is not the primary objective of this release.

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

The effective training corpus contains about 1.638B characters. The first five virtual epochs use one window alignment and the final five use an offset alignment, producing about 3.276B character-token exposures. The fixed validation stream contains about 1M character tokens.

The target source mixture is ≤50% Ultra-FineWeb Chinese, 32% SkyPile-150B Chinese, 14% Wikipedia, and 4% custom/user corpus. Shortfalls in Ultra-FineWeb or Wikipedia were filled from SkyPile instead of repeating earlier text. Raw corpora are not redistributed; see `DATA_LICENSES.md`.

Optimization uses AdamW, peak LR `7e-4`, 4% token-based warmup, cosine decay, weight decay 0.08, z-loss `1e-4`, gradient clipping 1.0, and effective global batch 384. The final epoch uses an LR floor of `1e-4`. Training primarily ran on Kaggle T4×2 with AMP, PyTorch SDPA, and DDP support.

### 4. Pretraining results

| Evaluation | Loss | Masked PPL | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| Final fixed validation | **1.9900** | **7.32** | **60.76%** | — |
| Standard mixed-masking benchmark | 2.0044 | 7.42 | 60.58% | 75.50% |
| Whole-piece/span stress benchmark | 3.0540 | 21.20 | 43.87% | 59.49% |

Masked PPL is `exp(masked cross-entropy)` and is meaningful only under compatible tokenizer and masking definitions.

### 5. Small downstream fine-tuning case study: 的 / 地 / 得

The Dededi experiment masks every target 的/地/得 position with HCAM's learned mask embedding and predicts the correct class from bidirectional context. Their character IDs are `的=138`, `地=164`, and `得=243`. For deployment, the three corresponding tied-embedding rows can be used as a 3×1056 classifier head, mathematically matching the three relevant full-vocabulary logits in FP32.

The first deployed HCAM fine-tune used epoch-3 EMA weights (E3-EMA). The original internal training set was not frozen as a public reproducibility snapshot, so this report intentionally does not claim an unverifiable exact fine-tuning sample count. The data covered natural 的/地/得 usage, fixed expressions, proper names, manner adverbials, complement structures, texture/property constructions, and difficult false-positive cases.

Key comparisons:

- Mixed Total (720 sentences / 2,275 targets): HCAM **98.17% Macro-F1** vs RoFormerV2 95.85%.
- External DEV (1,600 sentences / 3,025 targets): HCAM **97.496% Macro-F1** vs RoFormerV2 95.806%; HCAM clean false-action rate 0.628%.
- Natural DEV3 (1,200 snippets / 2,734 targets): HCAM **95.148% Macro-F1 / 95.882% Macro-F0.5** vs RoFormerV2 83.205% / 84.774%; clean false-action 0.988% vs 3.548%.

External DEV per-class HCAM F1: 的 99.143%, 地 96.690%, 得 96.654%. Natural DEV3 contains 600 snippets from DRCD-dev and 600 from Wikipedia, with target counts 的 2268 / 地 333 / 得 133 and frozen SHA256 `4a0d29f320dcf4ebb46a8ae73de61f372ea5201045918733713e75dcdd700487`.

Scores vary substantially across distributions, and one synthetic/template-heavy stress set strongly favored RoFormerV2. Therefore these results are reported as task-specific evidence rather than a universal ranking of Chinese encoders.

### 6. Limitations

1. HCAM is a bidirectional MLM encoder, not an autoregressive chat model.
2. Context length is 600.
3. The 18K vocabulary does not cover every Unicode character.
4. Only three layers use global attention, trading dense global interaction for efficiency.
5. Whole-piece/span masking remains much harder than standard mixed masking.
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
