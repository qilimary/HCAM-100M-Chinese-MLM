README 建议改动

1. 在“预训练结果 / Pretraining results”后插入：

## 真实文本与长距离依赖基准 / Real-text & long-range benchmark

在额外的真实文本基准中，HCAM 与 Chinese-RoBERTa-WWM-ext、RoFormerV2-Char-Base 进行了同套 REAL-MLM / REAL-RARE-SPAN / REAL-LONG 测试。

| Model | Params | REAL-MLM Top-1 | REAL-MLM Top-5 | Rare char Top-1 | Rare whole-span |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 102.78M | **68.50%** | **88.38%** | 9.79% | 1.33% |
| Chinese-RoBERTa-WWM-ext | 102.29M | 74.00% | 88.50% | 19.71% | 4.33% |
| RoFormerV2-Char-Base | 94.18M | 38.38% | 61.00% | 17.02% | 4.67% |

REAL-LONG 四个距离桶（64–469 字）简单平均：

| Model | Full Top-1 | No-cue Top-1 | Whole-span | Cue help rate | Mean ΔlogP |
|---|---:|---:|---:|---:|---:|
| **HCAM-100M** | 46.54% | 13.32% | 32.50% | **95.00%** | **+2.467** |
| Chinese-RoBERTa-WWM-ext | 58.54% | 22.87% | 45.63% | 96.25% | +2.296 |
| RoFormerV2-Char-Base | 60.09% | 20.55% | 45.00% | 95.63% | +2.574 |

HCAM 的绝对恢复率仍有提升空间，但其远距离 Cue 帮助率和 ΔlogP 与两种成熟 Transformer 基线处于相近量级；在 384–469 字距离上，删除远端 Cue 会使 HCAM Top-1 从 42.45% 降到 10.38%。这说明三层全局 Gated-GQA 已经能够让数百字外的信息稳定影响目标位置，当前差距并不是简单的“长距离信息传不过来”。

需要特别说明：这不是同数据、同训练预算的架构消融。HCAM 从零训练仅使用约 1.638B 个有效不重复字符（约 3.276B character-token exposures）；HFL 官方文档记录 EXT 语料约 5.4B 词，RoFormerV2 官方记录约 280GB 无监督数据并追加约 20GB 有监督多任务数据。单位不能直接换算，但训练资源明显不匹配。

**在明显更小的训练数据与个人训练预算下，HCAM 仍在普通真实文本 MLM 和长距离上下文利用上达到了相对于这些成熟基线并不弱、部分指标处于同一量级的能力。** 例如 REAL-MLM Top-5 为 88.38%，与 Chinese-RoBERTa-WWM-ext 的 88.50% 仅差 0.12 个百分点；REAL-LONG 的 Cue 帮助率为 95.00%，同样与两个成熟基线接近。Rare Span 和最终精确恢复率仍有差距，但这类能力高度依赖长尾语料覆盖，因此其中很大一部分差距很可能来自训练数据规模、覆盖度与训练预算，而不能直接归因于混合架构。

结合此前自回归缩放实验、MLM 结果和这次 REAL-LONG 测试，**目前未发现 HCAM 相比纯 Transformer 存在可明确归因于混合卷积-注意力架构本身的系统性性能差异或明显精度损失**。换句话说，现有结果没有显示“将大部分注意力层替换为门控扩张卷积”本身造成了明显能力退化；严格区分架构效果仍需要未来进行同数据、同参数量、同训练步数的控制实验。

详细方法、完整分桶结果和基线健康检查见 [`REAL_BENCHMARK.md`](REAL_BENCHMARK.md)。

---

2. “仓库文件 / Repository files”增加：

- `REAL_BENCHMARK.md` — 真实文本 MLM、稀有跨度与长距离依赖对比 / real-text MLM, rare-span and long-range benchmark

3. “已知限制 / Known limitations”建议把这两条：

- 只有 3 层全局注意力，这是效率与全局交互能力之间的折中。
- 词/跨度压力测试明显更难。

替换为：

- 只有 3 层使用全局注意力；REAL-LONG 已确认 400+ 字远端 Cue 可稳定影响预测，但密集、多轮全局交互能力仍需要更多控制实验评估。 / Only 3 layers use global attention; REAL-LONG confirms reliable use of 400+ character distant cues, while dense multi-step global interaction still needs controlled evaluation.
- 稀有词/连续跨度恢复仍弱于成熟大语料基线；由于训练资源不匹配，目前不能将该差距直接归因于架构。 / Rare-word and contiguous-span recovery remains below mature large-corpus baselines; because training resources are unmatched, this gap cannot currently be attributed directly to architecture.
