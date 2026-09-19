# VidLiner native video core

This Rust crate is the performance-oriented companion to the Python package. It targets the
embarrassingly parallel, renderer-independent operations that run once per frame or once per
variant:

- affine/placement projection of boxes;
- source-to-output time-span projection;
- one-axis contrast-pair discovery.

The Python DAG, cache, backend registry, provenance, and production guard remain authoritative.
The crate is intentionally free of model, filesystem, and credential code. It can later be exposed
as an optional PyO3/maturin extension (`vidliner_native`) without changing the Python contracts;
until then, the Python implementation remains the portable fallback.

## Check and benchmark

```bash
cargo test --manifest-path native/Cargo.toml
cargo test --manifest-path native/Cargo.toml --release
```

The parallel APIs retain deterministic ordering regardless of Rayon worker count, which is required
for reproducible manifests and cache keys.
