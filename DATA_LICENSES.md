# Training Data Sources and License Notes / 训练数据来源与许可说明

> This is a provenance/release note, not legal advice. / 本文件是来源与发布审计说明，不构成法律意见。

HCAM 100M does **not** redistribute the raw pretraining corpora. The training mixture targeted approximately 50% Ultra-FineWeb Chinese, 32% SkyPile-150B Chinese, 14% Wikipedia, and 4% custom/user corpus. If Ultra-FineWeb or Wikipedia was short of its target allocation, the shortfall was filled from SkyPile rather than by repeating earlier text.

HCAM 100M **不在本仓库重新分发原始预训练语料**。预训练目标混合比例约为：Ultra-FineWeb 中文 ≤50%、SkyPile-150B 中文 32%、Wikipedia 14%、自有/用户语料 4%。当 Ultra-FineWeb 或 Wikipedia 不足目标比例时，缺口由 SkyPile 回补，而不是重复较早文本。

## 1. Ultra-FineWeb

- Source / 来源: https://huggingface.co/datasets/openbmb/Ultra-FineWeb
- The dataset card marks the project as Apache-2.0, **but explicitly asks users to inspect the licenses of each component dataset because Ultra-FineWeb is built from multiple sources**.
- 数据集页面标记项目为 Apache-2.0，但同时明确提醒：由于 Ultra-FineWeb 由多个数据集构成，使用者仍应检查各组成数据集自身许可。

## 2. SkyPile-150B

- Source / 来源: https://huggingface.co/datasets/Skywork/SkyPile-150B
- The dataset card states that community use requires the **Skywork Community License** and that commercial use must comply with that license as well as Apache-2.0.
- 数据集页面说明社区使用需要遵守 **Skywork Community License**，商业用途也需同时遵守相关条款与 Apache-2.0。
- Because SkyPile is a substantial part of the pretraining mix, downstream users should read the current upstream license rather than assuming the released weights are free of all upstream conditions.
- 由于 SkyPile 在本模型预训练混合中占比较高，下游使用者不应假定模型权重可以消除所有上游许可条件，应自行阅读其最新许可。

## 3. Wikipedia / Wikimedia text

- Terms / 条款: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en
- Most Wikimedia text is made available under CC BY-SA 4.0 and GFDL, with attribution requirements; reuse of source text may also trigger ShareAlike obligations.
- 大多数 Wikimedia 文本使用 CC BY-SA 4.0 与 GFDL，并包含署名要求；直接再利用源文本时还可能涉及 ShareAlike。
- HCAM does not redistribute Wikipedia article dumps in this repository.
- 本仓库不重新分发 Wikipedia 原始文章数据。

## 4. Custom / user corpus (4%)

- This portion was private/internal project data during training and is **not redistributed here**.
- 该部分在训练阶段属于项目内部/自有语料，**本仓库不重新分发**。
- Its item-level provenance and redistribution rights have not been independently audited in this release package. If a future release publishes any of those texts, each item should be reviewed first.
- 本次开源包没有对其逐条来源和再分发权进行独立审计。若未来计划公开其中原文，应先逐条确认来源与许可。

## Release position / 本项目的发布立场

The repository code is Apache-2.0. Rights held by the author in the model weights are offered under CC BY 4.0 with attribution required. These grants do not relicense upstream datasets or erase third-party terms.

仓库代码采用 Apache-2.0；作者对模型权重拥有并能够授予的权利采用 CC BY 4.0，要求署名。上述许可不会重新许可上游数据集，也不会消除第三方条款。
