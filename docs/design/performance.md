# Performance architecture

VidLiner uses a split runtime rather than rewriting the whole system in a lower-level language.

## Keep in Python

Python remains the control plane: recipes, DAG assembly, capability binding, state persistence,
cache policy, provenance, review reports, and the production export guard. These paths are I/O- and
policy-heavy, not arithmetic hot loops, and keeping them in Python preserves fast iteration for
research operators and backends.

## Accelerate in Rust

The native video core targets the pure data plane:

1. project boxes for every frame of every track;
2. project word/event time spans;
3. discover one-axis contrast pairs across large variant manifests.

These operations have no side effects and can use Rayon safely. Results are collected in stable
input order and pairs are sorted after parallel discovery, so changing the worker count cannot
change IDs, manifests, or cache keys.

The first implementation lives in `native/` as a normal Rust crate. A future PyO3/maturin wheel can
replace the Python functions opportunistically; if the extension is absent or fails to load, the
reference Python implementation remains correct. Model inference and ffmpeg encoding are not moved
into this crate: they have different accelerator and process-level bottlenecks and should stay
behind explicit backend contracts.
