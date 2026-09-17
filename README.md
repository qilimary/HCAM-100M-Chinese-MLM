# HCAM 100M 中文掩码语言模型 / Chinese Masked Language Model

**HCAM = Hybrid Convolution-Attention Model（混合卷积-注意力模型）**

HCAM 100M 是一个约 **102.78M 参数**的中文双向掩码语言模型。它使用 **7 个双向门控扩张卷积层 + 3 个双向 Gated-GQA 全局注意力层**，以字符级 18K 词表建模中文文本，并通过 SentencePiece 边界指导多粒度 MLM 掩码。

HCAM 100M is a **102.78M-parameter** bidirectional Chinese masked language model. It combines **7 bidirectional gated dilated-convolution layers with 3 bidirectional Gated-GQA global-attention layers**, uses a true 18K character vocabulary, and employs SentencePiece boundaries only as a guide for multi-granularity MLM masking.

> 本仓库的主体是预训练基座。Dededi“的/地/得”微调仅作为一个小型下游实验。  
> The pretrained base is the main release. The Dededi 的/地/得 correction fine-tune is included only as a small downstream case study.

## 项目缘起 / Motivation

HCAM 100M 最初并不是为了超越某个现有模型而训练的。这个项目来自作者长期对小型语言模型架构、从零训练流程，以及卷积与注意力混合结构的兴趣。在训练 HCAM 100M 之前，同一条架构思路已经被作者逐步用于 **16M、30M、55M 和 100M** 规模的自回归语言模型实验。

在这些自回归实验之后，作者开始关注同一类混合卷积-注意力结构在**双向掩码语言建模**中的表现，于是重新设计了预训练目标、动态多粒度掩码策略、字符级 tokenizer 与训练流程，并最终从零训练出 HCAM 100M。

项目一开始只是一个个人研究实验，并没有预设必须达到怎样的下游成绩，也没有计划把模型部署到手机端。后续在中文“的 / 地 / 得”纠错任务上的微调结果超出了最初预期，才进一步推动了模型压缩、量化以及 Android 端部署。也正因为这条从架构实验、预训练、下游微调一直走到真实应用的路径，作者决定将 HCAM 100M 的预训练基座、架构实现、训练方法与实验结果公开。

HCAM 100M was not originally trained with the goal of outperforming a particular existing model. The project grew out of the author's long-standing interest in small language-model architectures, training models from scratch, and hybrid convolution-attention designs. Before HCAM 100M, the same architectural line had already been explored through autoregressive models at **16M, 30M, 55M, and 100M** scales.

After those autoregressive experiments, the author wanted to see how the same hybrid convolution-attention idea would behave under **bidirectional masked language modeling**. This led to a redesigned pretraining objective, dynamic multi-granularity masking strategy, character-level tokenizer, and training pipeline, eventually resulting in HCAM 100M.

The project began simply as a personal research experiment, without a predefined downstream target or a plan for mobile deployment. Later, its fine-tuning results on Chinese 的/地/得 correction exceeded the author's initial expectations, which led to further work on compression, quantization, and Android deployment. That progression—from architecture experiments to pretraining, downstream adaptation, and practical deployment—is one of the main reasons the HCAM 100M base model, architecture, training recipe, and evaluation results are being released publicly.

## 核心规格 / Key specifications

| 项目 / Item | 配置 / Value |
|---|---:|
| Parameters | **102,778,220** |
| Vocabulary | 18,000 character tokens |
| Context length | 600 |
| Hidden size | 1056 |
| Layers | 10 |
| Local gated dilated-conv layers | 7 |
| Global Gated-GQA layers | 3 (layers 3 / 7 / 10) |
| Query / KV heads | 16 / 4 |
| Head dim | 66 |
| Local inner dim | 1600 |
| GQA FFN dim | 2944 |
| Norm / FFN / Position | RMSNorm / SwiGLU / RoPE |
| Dropout | 0.10 |
| Weight tying | Input embedding = LM head |

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

## 预训练 / Pretraining

- 有效不重复训练字符：约 **1.638B** / ~**1.638B** effective unique training characters
- 两组错位窗口总训练暴露量：约 **3.276B** character-token exposures
- 固定验证集：**1M** character tokens
- 虚拟训练轮数：10
- 有效全局 batch：384
- Peak LR：`7e-4`
- Weight decay：`0.08`
- 总遮蔽率：30% → 15%
- 被选位置替换：80% learned mask embedding / 10% random char / 10% unchanged

训练数据目标比例 / target mixture:

- Ultra-FineWeb Chinese: ≤50%
- SkyPile-150B Chinese: 32%
- Wikipedia: 14%
- Custom/user corpus: 4%

详细来源与许可风险见 [`DATA_LICENSES.md`](DATA_LICENSES.md)。Raw corpora are **not redistributed** by this repository.

## 预训练结果 / Pretraining results

| Evaluation | MLM Loss | Masked PPL | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| Final fixed validation | **1.9900** | **7.32** | **60.76%** | — |
| Standard mixed masking benchmark | 2.0044 | 7.42 | 60.58% | 75.50% |
| Whole-piece/span stress benchmark | 3.0540 | 21.20 | 43.87% | 59.49% |

这里的 PPL 是 `exp(masked cross-entropy)`，不能直接与不同 tokenizer / 不同 MLM 定义的模型横向比较。  
The reported PPL is `exp(masked cross-entropy)` and should not be compared directly across incompatible tokenizers or masking setups.

## 小型下游实验：的 / 地 / 得 / Small downstream case study

第一版 HCAM E3-EMA 微调把每一个待判断的“的/地/得”位置替换为预训练 learned mask embedding，再基于双向上下文恢复类别。部署时只需要来自 tied embedding 的三行权重（的=138、地=164、得=243），无需输出完整 18K logits。

The first HCAM E3-EMA fine-tune masks every target 的/地/得 position with the pretrained learned mask embedding and predicts the correct class from bidirectional context. Deployment can use only the three tied-embedding rows for 的=138, 地=164, 得=243 instead of the full 18K vocabulary head.

| Evaluation | HCAM E3-EMA | RoFormerV2 E3-EMA |
|---|---:|---:|
| Mixed Total, 720 sentences / 2,275 targets — Macro-F1 | **98.17%** | 95.85% |
| External DEV, 1,600 sentences / 3,025 targets — Macro-F1 | **97.496%** | 95.806% |
| Natural DEV3, 1,200 snippets / 2,734 targets — Macro-F1 | **95.148%** | 83.205% |
| Natural DEV3 — Macro-F0.5 | **95.882%** | 84.774% |
| Natural DEV3 — clean false-action rate | **0.988%** | 3.548% |

这些结果仅代表该纠错任务，不代表 HCAM 在所有中文任务上优于 RoFormerV2。Natural DEV3 来自 DRCD-dev 与 Wikipedia 各 600 条，目标分布为 的 2268 / 地 333 / 得 133。

These results are task-specific and are **not** a claim that HCAM is universally superior to RoFormerV2. Natural DEV3 contains 600 snippets from DRCD-dev and 600 from Wikipedia, with target counts 的 2268 / 地 333 / 得 133.

## 实际应用示例 / Real-world Application Demo

HCAM 100M 的预训练基座已经被进一步微调用于中文“的 / 地 / 得”纠错，并集成到作者的 Android 应用中。

The HCAM 100M pretrained base has also been fine-tuned for Chinese 的/地/得 correction and integrated into the author's Android application.

下图为一次真实运行示例。专业模式由 HCAM 下游微调模型负责“的 / 地 / 得”判断。此次示例中，模型在约 1,187 字文本中给出了 3 条纠错建议，并显示对应置信度。截图中的 Android 端实际运行的是**量化后的 ExecuTorch/XNNPACK 部署版本**，并非 392 MiB 的 FP32 预训练基座；部署模型采用 dynamic INT8 per-channel 主干量化，并保留 FP32 三分类输出头。

The screenshot below shows a real application run. In Professional (HCAM) mode, the downstream HCAM model performs 的/地/得 classification and provides confidence scores for its correction suggestions. The Android app shown here runs a **quantized ExecuTorch/XNNPACK deployment build**, not the 392 MiB FP32 pretrained base; the deployment uses dynamic INT8 per-channel quantization for the main quantized operators while keeping the final three-class head in FP32.

![HCAM downstream application demo](app_demo.jpg)

> **说明 / Note:**  
> 本仓库的主要开源对象仍是 **HCAM 100M 预训练基座**。截图展示的是一个下游微调和移动端部署案例，并非独立基准测试。界面所示耗时取自一次实际设备运行，不代表所有设备的固定推理速度。  
>  
> The primary release of this repository remains the **HCAM 100M pretrained base model**. The screenshot is a downstream fine-tuning and mobile deployment example, not a standalone benchmark. The displayed latency is from one real-device run and should not be interpreted as a fixed speed across devices.

更多细节见 [`REPORT_HCAM.md`](REPORT_HCAM.md)。

## 下载权重 / Download weights

正式 FP32 基座约 **392 MiB**，不放入普通 Git 历史。请从仓库 **Releases** 下载：

- `hcam_100m_base_fp32.pt`
- `tokenizer.model`

The FP32 base checkpoint is about **392 MiB** and is distributed through **GitHub Releases**, not ordinary Git history.

## 快速使用 / Quick start

```bash
pip install -r requirements.txt
```

```python
import torch
from hcam import HCAMTokenizer, load_hcam, masked_logits

tok = HCAMTokenizer("tokenizer.model")
model, meta, _ = load_hcam(
    "hcam_100m_base_fp32.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

text = "温室里的植物正在安静地生长。"
ids = torch.tensor([tok.encode(text)], device=next(model.parameters()).device)
# mask character position 3 in the original text; +1 accounts for BOS
positions = torch.tensor([[4]], device=ids.device)
logits = masked_logits(model, ids, positions)
print(tok.pieces[int(logits[0, 0].argmax())])
```

也可以运行：

```bash
python benchmark.py --model hcam_100m_base_fp32.pt --tokenizer tokenizer.model
```

## 仓库文件 / Repository files

- `README.md` — 项目首页 / project overview
- `REPORT_HCAM.md` — 中英双语完整报告 / bilingual technical report
- `hcam.py` — 架构、tokenizer、加载函数 / architecture + tokenizer + loader
- `benchmark.py` — 最小 MLM 测试 / minimal MLM benchmark
- `requirements.txt` — Python dependencies
- `LICENSE` — 代码 Apache-2.0 / code license
- `NOTICE` — 代码署名通知 / attribution notice
- `MODEL_LICENSE.md` — 模型权重 CC BY 4.0 / model-weight license
- `DATA_LICENSES.md` — 数据来源与上游许可说明 / data provenance and upstream terms

## 许可与署名 / License and attribution

- **Code:** Apache License 2.0 + `NOTICE`
- **Model weights:** CC BY 4.0, attribution required, to the extent of rights the author can grant
- **Training data:** governed by their own upstream terms; see `DATA_LICENSES.md`

建议署名 / Suggested attribution:

> **HCAM 100M by qilimary** — https://github.com/qilimary/HCAM-100M-Chinese-MLM

## 已知限制 / Known limitations

- 这是双向 MLM 编码器，不是自回归聊天模型。 / This is a bidirectional MLM encoder, not an autoregressive chat model.
- 上下文长度固定为 600。 / Context length is 600.
- 极罕见字符可能回退到 `<unk>`。 / Very rare characters may fall back to `<unk>`.
- 只有 3 层全局注意力，这是效率与全局交互能力之间的折中。 / Only 3 layers use global attention; this is an efficiency/global-interaction trade-off.
- 词/跨度压力测试明显更难。 / Whole-piece/span masking remains substantially harder.
- 下游“的/地/得”实验不是通用中文基准。 / The 的/地/得 case study is not a general Chinese benchmark.

## Author

`qilimary` — 2026
