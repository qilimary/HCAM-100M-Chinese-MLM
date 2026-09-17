# HCAM 100M 真实文本与长距离依赖基准 / Real-text & Long-range Benchmark

本页记录 HCAM-100M、Chinese-RoBERTa-WWM-ext 与 RoFormerV2-Char-Base 在同一套真实文本 MLM、稀有跨度恢复和长距离实体利用测试中的结果。

This page reports HCAM-100M, Chinese-RoBERTa-WWM-ext, and RoFormerV2-Char-Base on the same real-text MLM, rare-span recovery, and long-range entity-use benchmark.

> **重要说明 / Important caveat**
>
> 这不是“同数据、同训练预算”的架构消融实验。三种模型的训练数据量、训练目标、训练阶段和 tokenizer 均不完全相同，因此绝对分数差异不能直接解释为架构差异。HCAM 从零训练时使用约 **1.638B 个有效不重复字符**，两组错位窗口合计约 **3.276B character-token exposures**。HFL 官方文档说明 Chinese-RoBERTa-WWM-ext 使用的 EXT 语料总词数约 **5.4B**；RoFormerV2 官方文档则说明其使用约 **280GB 无监督数据**，随后还有约 **20GB、77 个标注数据集构成的 92 个任务**进行有监督多任务训练。单位并不完全可直接换算，但训练资源规模显著不匹配。
>
> This is **not** a matched-data, matched-compute architectural ablation. Training corpus size, objectives, stages, and tokenizers differ across models, so absolute score gaps should not be interpreted as architecture-only effects. HCAM was trained from scratch on about **1.638B effective non-duplicate characters**, with about **3.276B character-token exposures** through two offset window passes. HFL documents roughly **5.4B words** for the EXT corpus used by Chinese-RoBERTa-WWM-ext, while the RoFormerV2 authors report about **280GB of unsupervised data** followed by roughly **20GB of supervised multi-task data** built from 77 labeled datasets and 92 tasks. These units are not directly interchangeable, but the training-resource scales are clearly unmatched.
>
> Sources: [HFL Chinese-BERT-wwm / RoBERTa-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm), [RoFormerV2](https://github.com/ZhuiyiTechnology/roformer-v2).

## 1. Baseline health check

RoFormerV2-Char-Base 在测试前进行了位置表健康检查。加载时发现 `roformer.encoder.embed_positions.weight` 未随权重提供，原始位置表包含非有限值；重新构造官方实现所需的正弦位置表后：

- shape: `(512, 64)`
- finite: `True`
- `requires_grad=False`
- 全模型参数 finite 检查通过
- 参数量：**94.18M**

随后运行作者 README 示例：

```text
今天[MASK]很好，我[MASK]去公园玩。
MASK 1 Top-5: ['我', '天', '晴', '园', '玩']
MASK 2 Top-5: ['想', '要', '会', '就', '带']
```

两个 MASK 的 Top-5 都与作者 README 示例 **5/5 重合**，因此才继续正式测试。

Before benchmarking, RoFormerV2-Char-Base was checked for positional-table integrity. After rebuilding the required sinusoidal positional table, all model parameters were finite and the official README MLM example matched **5/5 entries for both masked positions** in Top-5.

## 2. REAL-MLM 与稀有跨度恢复 / REAL-MLM & rare-span recovery

| 模型 / Model | 参数量 | REAL-MLM Top-1 | REAL-MLM Top-5 | Rare 字符 Top-1 | Rare 整词完全恢复 |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 102.78M | **68.50%** | **88.38%** | 9.79% | 1.33% |
| Chinese-RoBERTa-WWM-ext | 102.29M | 74.00% | 88.50% | 19.71% | 4.33% |
| RoFormerV2-Char-Base | 94.18M | 38.38% | 61.00% | 17.02% | 4.67% |

HCAM 的 REAL-MLM Top-5 为 **88.38%**，与 Chinese-RoBERTa-WWM-ext 的 **88.50%** 相差仅 **0.12 个百分点**；Top-1 相差 5.50 个百分点。这说明 HCAM 在普通真实文本 MLM 中经常已经把正确答案排入高概率候选，只是在第一名排序稳定性上仍有提升空间。

稀有跨度恢复是 HCAM 当前更明显的弱项。不过，Rare 测试对长尾词、专名、低频组合和语料覆盖非常敏感；在训练数据规模显著不匹配的情况下，不能把这里的差距直接解释为混合卷积-注意力架构的固有损失。

HCAM reaches **88.38% Top-5** on REAL-MLM, only **0.12 percentage points** below Chinese-RoBERTa-WWM-ext at 88.50%. The larger Top-1 gap suggests that correct candidates are often present but not always ranked first. Rare-span recovery is currently a clearer weakness, but it is also highly sensitive to long-tail coverage and corpus scale, so the gap should not be attributed directly to the hybrid architecture without a matched-data ablation.

## 3. REAL-LONG：真实远距离实体利用 / Real long-range entity use

REAL-LONG 在目标位置之外保留一个远距离实体线索（Cue），然后再次评测删除该 Cue 后的结果。`Cue 帮助率`表示删除远端线索后正确答案受到负面影响的样本比例；`ΔlogP`表示保留远端 Cue 时正确答案 log-probability 的平均提升。

REAL-LONG keeps a distant entity cue and then re-evaluates the same item after removing that cue. `Cue help rate` measures how often the distant cue improves the target prediction; `ΔlogP` is the average log-probability gain for the correct answer when the cue is present.

### HCAM 按距离分桶 / HCAM by distance

| Cue 距离 | N | 完整 Top-1 | 去 Cue Top-1 | 完整整词 | 去 Cue 整词 | Cue 帮助率 | ΔlogP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64–127 字 | 40 | 47.18% | 5.63% | 32.50% | 0.00% | **100.00%** | **+3.119** |
| 128–255 字 | 40 | 44.37% | 16.56% | 27.50% | 2.50% | 92.50% | +1.926 |
| 256–383 字 | 40 | 52.14% | 20.71% | 37.50% | 10.00% | 97.50% | +2.381 |
| 384–469 字 | 40 | 42.45% | 10.38% | 32.50% | 2.50% | **90.00%** | **+2.441** |

即使在 **384–469 字**距离上，HCAM 删除远端 Cue 后 Top-1 仍从 **42.45% 降到 10.38%**，同时平均 `ΔlogP=+2.441`。这说明模型并不是只依靠 MASK 附近的局部文本猜测答案；远端信息确实进入并影响了目标位置的表示。

Even at **384–469 characters**, removing the distant cue drops HCAM Top-1 from **42.45% to 10.38%**, with an average `ΔlogP=+2.441`. This is direct evidence that the model is actually using distant context rather than relying only on local text around the masked span.

### 三模型长距离汇总 / Three-model long-range summary

四个距离桶简单平均：

| 模型 / Model | 完整 Top-1 | 去 Cue Top-1 | 完整整词 | Cue 帮助率 | 平均 ΔlogP |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 46.54% | 13.32% | 32.50% | **95.00%** | **+2.467** |
| Chinese-RoBERTa-WWM-ext | 58.54% | 22.87% | 45.63% | 96.25% | +2.296 |
| RoFormerV2-Char-Base | 60.09% | 20.55% | 45.00% | 95.63% | +2.574 |

HCAM 的最终绝对恢复率低于两种成熟基线，但其 **95.00% Cue 帮助率**与两者的 96.25% / 95.63% 非常接近，平均 **ΔlogP=+2.467** 也处于相同量级。这说明 HCAM 的主要差距并不是“远距离信息无法传递”。至少在这套测试中，三层全局 Gated-GQA 已经足以让数百字外的信息稳定影响目标位置。

HCAM has lower absolute recovery accuracy than the two mature baselines, but its **95.00% cue-help rate** is very close to 96.25% / 95.63%, and its mean **ΔlogP=+2.467** is in the same range. The main observed gap therefore does **not** look like a failure of long-range information transport. In this benchmark, only three global Gated-GQA layers are sufficient for context hundreds of Chinese characters away to consistently affect the target representation.

## 4. 如何解释这些结果 / Interpretation

HCAM 的结构是：

```text
local conv → local conv → GLOBAL
→ local conv → local conv → local conv → GLOBAL
→ local conv → local conv → GLOBAL
```

全局注意力虽然只出现在第 3 / 7 / 10 层，但一次全局注意力写入 hidden state 后，远端信息会通过残差连接继续保留，并由后续门控扩张卷积进一步局部加工；下一次全局层再重新进行全序列信息交换。因此，“只有三层全局注意力”并不等价于“只有三层能拥有全局信息”。

The encoder alternates local processing and global communication. Once a global-attention block writes distant information into a token representation, that information can persist through residual connections and be refined by later local convolutional blocks before the next global exchange. Therefore, having only three global-attention layers does not mean that global information exists only inside those three layers.

### 当前能支持的结论 / What the current evidence supports

1. **HCAM 的普通真实文本 MLM 能力并不弱。** 在参数量相近的情况下，REAL-MLM Top-5 已与 Chinese-RoBERTa-WWM-ext 基本持平。
2. **HCAM 已建立明确的长距离上下文利用能力。** 400 字以上的远端 Cue 仍能显著改变预测，Cue 帮助率与两个成熟 Transformer 基线接近。
3. **当前更明显的短板是 Rare Span 与最终精确恢复率，而不是远距离信息“传不过来”。**
4. **在明显更小的训练数据与个人训练预算下，HCAM 仍表现出了相对于成熟基线并不弱的普通 MLM 与长距离上下文能力。** REAL-MLM Top-5 与 Chinese-RoBERTa-WWM-ext 几乎持平，REAL-LONG 的 Cue 帮助率也与两个成熟基线处于同一量级；这说明小规模个人训练并没有阻止 HCAM 建立具有竞争力的上下文表示能力。
5. **训练数据规模与覆盖度很可能贡献了相当一部分剩余绝对分数差距。** HCAM 与两个基线并非同数据、同训练预算；尤其 Rare/实体恢复高度依赖长尾语料覆盖。
6. **在目前已有的自回归缩放实验、MLM 结果与 REAL-LONG 测试中，没有发现 HCAM 相比纯 Transformer 存在可明确归因于混合架构本身的系统性性能退化。** 当前绝对分数差距不应直接写成“卷积替换注意力导致的精度损失”。要严格区分架构效果，仍需要未来进行同数据、同参数量、同训练步数的控制实验。

In short, the present evidence does **not** reveal a systematic performance degradation that can be confidently attributed to the hybrid convolution-attention architecture itself. HCAM retains strong ordinary MLM performance and clear long-range cue use despite replacing most Transformer attention layers with gated dilated convolutions. The remaining absolute gaps are confounded by substantially different training resources and should not be presented as architecture-only losses. A strict architecture claim would require a future matched-data, matched-parameter, matched-step ablation.

## 5. 测试规模 / Benchmark size

- REAL-MLM: **800** items
- REAL-RARE-SPAN: **300** items
- REAL-LONG: **160** items total, 40 per distance bucket
- Long-range buckets: `64–127`, `128–255`, `256–383`, `384–469` Chinese characters

由于每个 LONG 距离桶只有 40 条，而且不同桶不是同一句话只改变距离，因此不同距离桶之间的绝对 Top-1 不应该被直接解释成严格的“随距离衰减曲线”。这套测试更适合回答：**模型是否真正利用了数百字外的线索？** 对 HCAM 来说，答案是肯定的。

Because each LONG bucket contains only 40 items and the buckets do not reuse exactly the same sentence with only distance changed, the absolute scores across buckets should not be treated as a clean distance-decay curve. The benchmark is better suited to the question: **does the model genuinely use cues hundreds of characters away?** For HCAM, the answer is clearly yes.
