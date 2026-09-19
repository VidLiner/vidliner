# VidLiner 首发文案

以下内容可直接作为 GitHub 初版仓库的发布素材；所有表述均以当前图像 MVP 的实际能力为边界。

## GitHub 仓库

**简介**

> 以验证为先的合成训练数据流水线：生成、重新分割、验证并导出带标签与证据的图像样本。

**建议 Topics**

`synthetic-data` · `data-augmentation` · `computer-vision` · `machine-learning` · `dataset-quality` · `image-processing` · `pydantic` · `python`

**社交预览图**

在 GitHub 仓库 Social preview 设置中直接上传已备好的 1280×640 [`vidliner-social-card.png`](../assets/vidliner-social-card.png)。可编辑源文件为 [`vidliner-social-card.svg`](../assets/vidliner-social-card.svg)。

## 首个 Release

**标题**

> v0.1.0-alpha.1 — 图像流水线 MVP

**正文**

> VidLiner 现已开源：这是一条以对象为中心的合成训练数据流水线，遵循一条简单规则——生成图像在标签和证据通过验证前，不是训练数据。
>
> 此 Alpha 提供完整的图像工作流：数据发现、目标对象选择、通过可替换 backend 生成、生成后重新检测与重新分割、质量门槛、标注重建、溯源、复核报告、缓存/恢复及 COCO/YOLO 导出。
>
> 默认本地 profile 刻意保持无外部依赖，适合演示与测试。任何真实训练集使用前，都应绑定生产级的检测、分割、生成与评估 backend。
>
> 视频跟踪、时序生成和视频导出将在后续阶段实现。

## 中文社媒帖

> 发布 VidLiner：可证明的合成数据。
>
> VidLiner 将图像转化为被接受，或带明确拒绝原因的训练样本。它会在生成后重新检测并分割目标，测量质量门槛，重建标签，并为每个通过的样本导出溯源记录。
>
> 开源图像流水线 MVP；视频能力正在规划，暂不夸大承诺。

## 面向社区的一段话

> 生成一张图像不难，难的是确定它是否可以安全用于训练。VidLiner 是一个开源、以对象为中心的图像增强流水线，将标签、质量证据、数据集 split 安全与溯源作为一等产物。被拒绝的候选与诊断信息同样会被保留，因此团队可以调整质量策略，而不会静默丢失证据。首个 Alpha 仅支持图像，并且刻意保持模型无关。

## 维护者回复模板

> VidLiner 仍处于早期 Alpha。默认 backend 的目标是让整条流水线无需付费 API 就可复现、可测试；这并不代表其具备生产级视觉理解能力。我们欢迎 backend 集成、真实失败样本，以及对验收策略的反馈。
