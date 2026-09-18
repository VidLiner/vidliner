# VidLiner — Naming and Vocabulary

This file exists so that the codebase has one answer to "what do we call this". Every term below
was chosen for VidLiner from its requirements. None of it is inherited from another project's
vocabulary, DSL, command set, package namespace, or directory layout.

## Product vocabulary

| Term | Meaning |
| --- | --- |
| **Recipe** | Declarative YAML description of an augmentation job's intent. |
| **Operator** | One domain operation with declared inputs, outputs, config schema, and capability needs. |
| **Capability** | A named, versioned ability the pipeline requires, e.g. `vision.object_detection.v1`. |
| **Backend** | A concrete implementation that serves one or more capabilities. |
| **Runtime profile** | The document that says which backend serves which capability on this machine. |
| **Binding** | The mapping entry inside a runtime profile from a capability to a backend name. |
| **Job** | One execution of a compiled recipe against a dataset. |
| **Node** | One operator instance inside a job's operation graph. |
| **Artifact** | Any digest-addressed output of a node. |
| **Candidate** | One generated replacement variant for one target object. |
| **Scene context** | Measured facts about the target object and its surroundings. |
| **Gate** | One acceptance threshold evaluated against one metric. |
| **Acceptance policy** | The set of gates a candidate must pass to enter the dataset. |
| **Decision** | The outcome of the gate stage: accepted, rejected, or needs review. |
| **Assessment** | A single evaluator's measured evidence about a candidate or sample. |
| **Evidence** | The recorded execution facts that justify a decision. |
| **Provenance** | The link from an output sample back to its source, recipe, seeds, and backends. |
| **Lineage** | The parent/child relation between a source sample and its augmentation descendants. |

## Module layout (each name is owned by this project)

```
vidliner/
  domain/        typed objects, reasons, policy models
  core/          graph engine, identity, canonical serialization, results, errors, randomness
  pipeline/      recipe loading, compiler, planning facade, run orchestration
  operators/     one module per domain operation
  capabilities/  protocol definitions (the abstract ports)
  backends/      concrete adapters (the only place vendor code may appear)
  runtime/       profile loading, registry, engine, profiles, estimator
  quality/       metrics, gates, acceptance policy engine
  annotations/   internal IR converters (COCO, YOLO)
  storage/       workspace, content-addressed artifact store, SQLite state
  control/       dataset discovery, splits, duplicates, reporting
  cli/           Typer application
```

Deliberately avoided: any `augforge` identifier (the specification's placeholder name is not the
product), any single-letter or truncated module name borrowed from another project, and any
one-to-one directory correspondence with a project that is not this one.

## Identifier conventions

| Kind | Convention | Example |
| --- | --- | --- |
| Modules | `snake_case` | `scene_analysis.py` |
| Classes | `PascalCase` | `ReplacementPlanner` |
| Public functions | verb-first `snake_case` | `assemble_operation_graph` |
| Capability names | dotted, versioned | `quality.background_preservation.v1` |
| Backend names | `snake_case` | `heuristic_detector` |
| Operator names | dotted verb | `generate.replacement` |
| Prefixes for derived ids | short, documented | `j_`, `s_`, `o_`, `c_`, `n_` |
| Reason codes | `SCREAMING_SNAKE` | `BACKGROUND_CHANGED` |
| Env variables | `VIDLINER_` prefix | `VIDLINER_WORKSPACE` |

Id prefixes: `j_` job, `s_` sample, `o_` object, `c_` candidate, `n_` node, `a_` artifact alias
(artifacts themselves are addressed by digest), `r_` run.
