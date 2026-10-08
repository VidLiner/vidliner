# VidLiner workflow enhancement: reference assessment

## Evidence and project understanding

The reviewed VidLiner baseline is `47cd0ec`: recipes compile into typed `OperationGraph` nodes;
the runtime resolves capabilities, schedules nodes, caches artifacts and records evidence;
the runner applies acceptance and production export checks. Video support currently includes
deterministic variant planning and geometry/time projection, with optional parallel Rust kernels.
Full video generation, rendering and tracking are still outside the implemented pipeline.

The workspace's graft graph was used to locate the planner, executor, runner, backend registry,
and production readiness checks. These relationships were checked against current source and
tests. AOCI was initialized and scanned, and its MCP rules and Overview were inspected. Its
formal code index has **zero entries** and reports governance unaligned. The initial AOCI assets
are kept locally for completing its governed authoring workflow; they are not published as a
finished project index. Source inspection remains necessary for the conclusions below.

The external reference is [Toonflow-app](https://github.com/HBAI-Ltd/Toonflow-app), inspected at
commit [`72a895c`](https://github.com/HBAI-Ltd/Toonflow-app/tree/72a895c26aab3f54c5a914517615362208fa6008).
Its README describes a creative video platform combining script, asset and generation workflows
on an infinite canvas. The relevant implementation evidence includes:

- [`apps/web/src/stores/workspace.ts`](https://github.com/HBAI-Ltd/Toonflow-app/blob/72a895c26aab3f54c5a914517615362208fa6008/apps/web/src/stores/workspace.ts): validated workspace selection and persistent project navigation.
- [`packages/nodeScaffold/src/connection.ts`](https://github.com/HBAI-Ltd/Toonflow-app/blob/72a895c26aab3f54c5a914517615362208fa6008/packages/nodeScaffold/src/connection.ts): declared node handles and connection type compatibility.
- [`packages/nodeScaffold/src/workspaceFiles.ts`](https://github.com/HBAI-Ltd/Toonflow-app/blob/72a895c26aab3f54c5a914517615362208fa6008/packages/nodeScaffold/src/workspaceFiles.ts): node-associated media file organization.
- [`packages/nodeScaffold/src/nodeTools.ts`](https://github.com/HBAI-Ltd/Toonflow-app/blob/72a895c26aab3f54c5a914517615362208fa6008/packages/nodeScaffold/src/nodeTools.ts): discovering and invoking tools belonging to current canvas nodes.

These are conceptual references. No Toonflow source, assets or dependencies are included in
VidLiner's implementation.

## Decision

Add a versioned canvas document around the existing DAG, plus a typed operator catalogue and
an offline preview. Preserve every graph execution field, external input binding and tuple
selection. Canvas coordinates belong to presentation; capability binding names are hints.
Import validates the document against the current operator registry without running backends.

Roundtrip tests exposed two existing undeclared `source_image` inputs. The generator now
declares its optional input while retaining its sample-context fallback; refinement declares
its required original image. This aligns the palette with inputs the assembled graph already
passes and the implementations already consume.

## What transfers well

| Toonflow idea | VidLiner application |
| --- | --- |
| Canvas nodes with typed handles | Export the compiled DAG as typed ports and explicit edges. |
| Discoverable node tools | Expose registered operator metadata and config JSON Schemas for a future palette. |
| Project-bound media assets | Build future asset browsing over existing content-addressed artifacts and lineage. |
| Provider-independent generation | Preserve capability/runtime-profile separation and show unresolved bindings. |
| Visual inspection during creation | Inspect the plan offline before running generation or spending on a provider. |

VidLiner produces verified training data. A future canvas must therefore retain re-detection,
annotation regeneration, split integrity, provenance and production export checks. A visually
successful media generation node alone is insufficient for accepted dataset output.

## Next increments

1. Persist authored projects containing recipe references, layout and selected artifacts; reuse
   the current workspace and state store for execution.
2. Add editable ports/config forms driven by the catalogue, with server-side graph validation.
3. Add artifact previews, candidate quality evidence and human review decisions.
4. Add actual video render/generation capabilities and temporal evidence before promoting
   video plans to video dataset export.

Provider accounts, plugin installation, a desktop/web monorepo and distributed workers require
separate product and runtime decisions. This change establishes their graph integration boundary.
