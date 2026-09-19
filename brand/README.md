# VidLiner Brand Kit

VidLiner is the verification-first pipeline for synthetic training data.

> **Synthetic data you can prove.**

The brand should make one idea memorable: VidLiner does not merely generate an image; it produces
a candidate, validates it, rebuilds its label, and preserves the evidence needed to trust or reject
it.

This is an **image-pipeline MVP**. Video tracking, temporal generation, and video export are
planned capabilities, not release claims.

## Positioning

| Element | Approved language |
| --- | --- |
| Category | Verification-first synthetic training-data pipeline |
| Primary audience | ML, computer-vision, and data-platform engineers |
| Core promise | Turn existing images into accepted or explainably rejected training samples |
| Differentiator | Labels, provenance, and quality evidence are first-class outputs |
| Short descriptor | Object-centric synthetic data augmentation with mandatory verification |
| Tagline | **Synthetic data you can prove.** |

### Message hierarchy

1. **Trust, before scale.** Every candidate is measured against explicit acceptance gates.
2. **Labels follow pixels.** VidLiner re-detects and re-segments the generated target before
   rebuilding its annotation.
3. **Evidence travels with the sample.** Accepted data includes provenance and quality evidence;
   rejected candidates retain diagnostics.
4. **Model freedom.** Recipes describe intent while runtime profiles bind the concrete backends.

## Voice

Write like a careful engineer: precise, direct, calm, and evidence-led.

| Do | Avoid |
| --- | --- |
| “Export only candidates that pass explicit gates.” | “Generate perfect datasets automatically.” |
| “Image pipeline MVP; video is planned.” | “Video generation framework” as a present-tense claim. |
| “Measured background preservation.” | “Zero-hallucination AI.” |
| “Works with replaceable backends.” | Naming untested models as supported. |

Useful verbs: **generate, verify, measure, rebuild, trace, reject, export**.

## Visual system

The mark combines a `V` with a proof line: a candidate becomes useful only after it is verified.
Keep its corners, line weight, and green proof point intact. Do not stretch, recolour, rotate, add
shadows, or put the dark wordmark on a dark surface.

### Assets

| Asset | Use |
| --- | --- |
| [Mark](./assets/vidliner-mark.svg) | Avatar, favicon, compact UI locations |
| [Dark wordmark](./assets/vidliner-logo-dark.svg) | Light backgrounds, documents |
| [Light wordmark](./assets/vidliner-logo-light.svg) | Dark backgrounds, slides |
| [Social card PNG](./assets/vidliner-social-card.png) | GitHub release, Open Graph, social posts |
| [Social card SVG source](./assets/vidliner-social-card.svg) | Editable source for the social card |

### Colour tokens

| Token | Hex | Role |
| --- | --- | --- |
| `vl-night` | `#0B1020` | Primary dark surface |
| `vl-ink` | `#10182C` | Primary text on light surfaces |
| `vl-cyan` | `#35D2FF` | Data flow and active emphasis |
| `vl-blue` | `#4B8CFF` | Supporting accent |
| `vl-verify` | `#57E4A7` | Acceptance, proof, success |
| `vl-cloud` | `#F4F8FF` | Text on dark surfaces |
| `vl-muted` | `#A8B7D0` | Secondary text on dark surfaces |

Use green only for proof, acceptance, or a confirmed state. Blue/cyan convey execution and data
flow; they are not a substitute for “passed.”

### Type

Use a modern system sans serif for product and marketing headings. Use a monospace face for recipes,
metrics, backend IDs, and verification facts. The supplied SVG files use broadly available system
font fallbacks, so they render without bundling a font.

## Repository use

- Use the mark in GitHub’s repository avatar and the social card as the social-preview source.
- Use the short description below as the GitHub repository description.
- Preserve the `Alpha` label until a production-grade backend profile and real-world validation
  suite are available.

The ready-to-paste GitHub and social copy is in [launch/README.md](./launch/README.md).
