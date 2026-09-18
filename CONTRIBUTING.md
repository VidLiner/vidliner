# Contributing

[简体中文](./CONTRIBUTING.zh-CN.md)

Thanks for helping. This document covers how to work on VidLiner and, more importantly, the rules
that keep it honest.

---

## Set up

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Before every commit:

```bash
ruff check . && ruff format --check .   # or: ruff format .
ty check vidliner
pytest -q
```

All three must pass. They are cheap, and a change that breaks one is a change that will break
someone else's afternoon.

---

## The rules that matter

### 1. Unimplemented behaviour fails loudly

```python
raise NotImplementedError("video replacement is a phase-2 capability")
```

Never a placeholder return, never a `pass` body, never a "for now" that silently succeeds. A
component that quietly does nothing produces a dataset that looks complete and is not, which is the
most dangerous possible bug in this project.

### 2. No silent exception swallowing

```python
# No.
try:
    ...
except Exception:
    pass

# Yes, when the case is genuinely expected and you say so.
with contextlib.suppress(TimeoutError):
    await asyncio.wait_for(event.wait(), timeout=delay)
```

`tests/unit/test_architecture.py` catches the syntactic case; catching and continuing into a
fabricated success is on you.

### 3. Core stays vendor-free

Nothing outside `vidliner/backends/` may import a model runtime or a vendor SDK. A test enforces it.
If you need a capability, define the protocol in `capabilities/` and implement it in `backends/`.

### 4. Layer direction

```
cli → pipeline → operators → capabilities → domain/core → (storage, control, runtime) → backends
```

A lower layer never imports a higher one. A test enforces the direction for `domain`, `core`,
`capabilities`, `quality`, and `annotations`.

### 5. Determinism is a feature

* never call module-level `random` or `numpy.random` global state — take the seed the context gives
  you;
* derive identities from content, not from insertion order or the clock;
* if you add a fan-out, key it by explicit identity (`object:<id>`, `candidate:<key>`), never by list
  position, or you will silently invalidate the cache for unrelated work.

### 6. No credentials, ever

```python
credentials:
  api_key: { source: env, name: EDIT_KEY }   # a reference
```

Never a value in a recipe, a profile inside the repository, a manifest, a log line, or a test. The
resolver in `runtime/secrets.py` is the only place a secret becomes a string, and it redacts what it
knows from every event.

### 7. Type hints

Every public function has annotations and a docstring saying *why*, not *what*. The docstring rule is
enforced for public API by a test.

### 8. Docstrings explain the decision

```python
def quantise_box(...):
    """...quantised before hashing so a one-pixel jitter does not create a new identity..."""
```

Say why the code is the way it is. The code already says what it does.

---

## Clean-room policy

VidLiner is an independent implementation. Do not copy source, identifiers, file contents, DSLs,
command vocabularies, comments, documentation, tests, manifests, assets, or licence text from another
project. General architecture ideas — a dependency graph, ports and adapters, content-addressed
storage — are common practice and fair to use; a specific expression of them is not.

If a feature can only be achieved by copying, re-derive it from the requirement instead. If that is
not possible yet, raise `NotImplementedError` and note it in `docs/design/milestones.md`.

See `docs/design/decisions.md` (ADR-001) for the full statement.

---

## Adding things

### A backend

1. Implement the protocol in `vidliner/backends/` (or your own package).
2. Declare `backend_id`, `backend_version`, `capabilities`, `determinism`, `safe_to_retry`,
   `blocking`, and `declared_capabilities`.
3. Bind it in a runtime profile.
4. Add contract tests: probe, happy path, every legitimate refusal.
5. Document it in `docs/backends.md` if it is generic enough to ship.

### An operator

1. Create a module in `vidliner/operators/` with an `OperatorSpec`, an input/config Pydantic model,
   and a `register_all(registry)` function.
2. Wire it into `build_default_registry`.
3. Add a stage to `StageName` if it introduces one, and to `_STAGE_ORDER` for deterministic ordering.
4. Add it to the assembler if it belongs in the default graph.
5. Test it directly, and add a pipeline test if it changes the graph.

### A quality metric

1. Add the name to `MetricName` and its metering to the evaluate operator.
2. Add the default reason code in `quality/gates.py` and the code itself to `ReasonCode`.
3. Add it to the documentation table in `docs/quality.md`.
4. Test both a passing and a failing case.

### An architecture rule

Add it to `tests/unit/test_architecture.py`. A rule the build enforces survives; a rule that lives
only in a document does not.

---

## Commits and pull requests

* one logical change per commit, with a message that says what changed and why;
* a test for every behaviour change, including the failure path;
* update the documentation the change makes stale — `README.md`, `ARCHITECTURE.md`, and the relevant
  `docs/*.md`;
* if the change is a decision someone might later question, record it in
  `docs/design/decisions.md` rather than leaving it in a comment.

### Review checklist

- [ ] Does an annotation stay correct? (priority 1)
- [ ] Does the dataset stay internally consistent? (priority 2)
- [ ] Is the result reproducible from the recorded seed and inputs? (priority 3)
- [ ] Is provenance still complete, and still free of credentials? (priority 4)
- [ ] Does quality still reject what it should? (priority 5)
- [ ] Is the core still independent of vendors? (priority 6)

The order is the priority order. Performance and UI come last on purpose: 100 wrongly-labelled
generated samples are worse than 10 good ones.

---

## Reporting a bug

Include:

* the command you ran and the recipe (redact any path or value you consider sensitive);
* `vidliner job show <id> --json` and `vidliner report events <id> --json`;
* which backend was bound, from `vidliner backend check`;
* what you expected to happen to the *dataset*.

A bug report that says which samples should have been rejected and were not is worth ten that say a
run failed.
