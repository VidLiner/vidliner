# Workflow canvas exchange

VidLiner can export a compiled recipe as a portable canvas document, validate its connections,
and produce an interactive, standalone HTML preview. This makes a plan inspectable before any
generation runs and supports editing layout and operator configuration drafts.

The canvas uses the existing `OperationGraph`, operator registry and execution contracts.
It does not introduce another scheduler. Backend names in the document are planning hints;
production readiness and acceptance still have to be checked when the recipe runs.

## Quickstart

From an initialized workspace containing `data/source` and a valid recipe:

```sh
vidliner workflow export recipe.yaml --workspace . --limit 1 --output workflow.json
vidliner workflow validate workflow.json
vidliner workflow preview workflow.json --output workflow.html
```

Open `workflow.html` in a browser. Search by operator, node ID or lineage, drag the background to
pan, use the mouse wheel or buttons to zoom, and select a node to inspect its config, ports,
connections, capability requirements, backend hints, retry policy and concurrency limit.
The page works offline and loads no external libraries. `Edit layout` enables node dragging with
connected edges following the node. The selected node's configuration can be edited as a JSON
object. `Save JSON` downloads the draft; run `vidliner workflow validate` on it before using it.
Browser JSON syntax checks do not replace operator schema validation. Editing does not run a model.

Export defaults to one sample to keep the first canvas manageable. `--limit 0` includes all
augmentable samples. Neither export nor preview overwrites an existing file unless `--force`
is supplied. Export discovers the dataset and resolves advertised bindings without instantiating
backends; an unbound capability is retained in the document rather than preventing inspection.

## Document contract

`schema_version` is `vidliner.workflow/v1`. Unknown fields and unsupported schema versions are
rejected. Get the JSON Schema or discover the node palette with:

```sh
vidliner workflow schema
vidliner workflow catalog
```

| Field | Meaning |
| --- | --- |
| `nodes` | Registered operator instances with version, stage, config, outputs and execution policy. |
| `nodes[].position` | Canvas coordinates; moving a node does not change execution semantics. |
| `nodes[].inputs` | External sample/artifact roles or JSON constants, including explicit `null`. |
| `edges` | One source output port connected to one target input port. |
| `nodes[].selects` | Explicit tuple selection indices for per-object branches. |
| `job_lineage`, `nodes[].lineage` | Original identity paths used by the graph and seed derivation. |
| `recipe_hash` | The originating recipe's identity; not a proof that an edited document still matches it. |
| `capability_bindings`, `unmet_capabilities` | Planning hints; credentials and runtime profiles stay outside the document. |

Validation checks duplicate node IDs, unknown nodes or output ports, multiply bound inputs,
cycles, missing or undeclared input ports, port type compatibility, operator versions and config
schemas. Outputs, stages, determinism and capability requirements must agree with the registered
operator contract. Credential-like fields and non-finite coordinates are rejected.

Python consumers use the same interface:

```python
from vidliner.pipeline.workflow import graph_from_workflow, workflow_from_graph

document = workflow_from_graph(plan.graph, name=plan.recipe.name)
restored_graph = graph_from_workflow(document)
```

`load_workflow(path)` reads and validates JSON against the built-in catalogue. Embedders with
their own explicitly registered operators can pass `registry=` to `graph_from_workflow` or
`operator_catalogue`. Constants and configs in this interchange format contain JSON values,
not arbitrary Python objects. Node order is normalized to the graph's topological order.

## Scope

This release provides canvas interchange, a typed palette, and offline plan inspection and editing. Recipe
execution remains `vidliner run recipe.yaml`. The document is a graph snapshot, not a replacement
for the recipe's dataset and acceptance policy. It has no standalone execution command.

Local video rendering is available through `video probe/generate/render`; it is not yet wired to
canvas execution. Structural graph editing, media asset browsing, live job status and an authenticated
server remain future work. See the [Toonflow assessment](design/toonflow-assessment.md) for the
design rationale and the relationship to the existing video planner and Rust kernels.
