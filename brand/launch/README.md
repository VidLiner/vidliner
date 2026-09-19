# VidLiner Launch Copy

These are factual, release-ready starting points for the first public repository release.

## GitHub repository

**Description**

> Verification-first synthetic training data pipeline — generate, re-segment, validate, and export image samples with labels and evidence.

**Topics**

`synthetic-data` · `data-augmentation` · `computer-vision` · `machine-learning` · `dataset-quality` · `image-processing` · `pydantic` · `python`

**Social preview**

Upload the ready-to-use 1280×640 [`vidliner-social-card.png`](../assets/vidliner-social-card.png) in the GitHub repository social-preview settings. The editable source is [`vidliner-social-card.svg`](../assets/vidliner-social-card.svg).

## First release

**Title**

> v0.1.0-alpha.1 — Image Pipeline MVP

**Body**

> VidLiner is now public: an object-centric synthetic training-data pipeline built around a simple rule — a generated image is not training data until its label and evidence pass verification.
>
> This alpha ships a complete image workflow: discovery, object selection, generation through replaceable backends, post-generation re-detection and re-segmentation, quality gates, annotation rebuilding, provenance, review reports, cache/resume, and COCO/YOLO export.
>
> The default local profile is intentionally dependency-free and designed for demos and tests. Bind production-grade detection, segmentation, generation, and evaluation backends before using generated output in a real training set.
>
> Video tracking, temporal generation, and video export are planned for a future phase.

## X / LinkedIn

> Introducing VidLiner: synthetic training data you can prove.
>
> VidLiner turns images into accepted—or explainably rejected—training samples. It re-detects and re-segments generated targets, measures quality gates, rebuilds labels, and exports provenance with every accepted sample.
>
> Open-source image pipeline MVP. Video is planned, not claimed.

## Product Hunt / community post

**Headline**

> VidLiner — verification-first synthetic data for computer vision

**One paragraph**

> Generating an image is easy; knowing whether it is safe to train on is the hard part. VidLiner is an open-source, object-centric image augmentation pipeline that treats labels, quality evidence, split safety, and provenance as first-class outputs. It keeps rejected candidates and their diagnostics, so quality policy can improve without silently losing evidence. The first alpha is image-only and intentionally model-agnostic.

## Maintainer reply

> VidLiner is early alpha software. The default backends exist to make the pipeline reproducible and runnable without a paid API; they are not a claim of production-grade visual understanding. We welcome backend integrations, real-world failure cases, and feedback on the acceptance-policy design.
