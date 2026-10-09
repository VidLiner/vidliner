# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Inferred from the current workflow canvas and CLI: video-data creators and technical artists who
compose provider-neutral generation, transformation, validation, and export workflows.

## Product Purpose

VidLiner lets a user construct a typed video workflow, execute local or remote generation steps,
inspect durable task evidence, and continue work without silently resubmitting an uncertain provider
request. Success means a workflow is understandable, editable, repeatable, and honest about what has
or has not been verified.

## Positioning

The product combines a visual graph editor with the same durable runtime contracts used by the CLI:
provider-specific APIs stay behind capability bindings, while graph edits, task handles, status, and
provenance remain inspectable in one workflow surface.

## Operating Context

The primary surface is a local loopback web canvas used beside provider consoles, code, and media
review tools. Users add operators, connect typed ports, edit JSON configuration, run a reviewed graph,
watch task progress, and inspect generated media or execution evidence.

## Capabilities and Constraints

The existing canvas supports typed graph editing, keyboard and pointer connections, layout persistence,
bounded execution, Runway/fal/Bifrost-style asynchronous video APIs, task refresh and cancellation,
and local provider-free planning. Credentials stay in the host runtime and are never exposed to the
browser. Remote media is unverified until a separate acceptance/export policy approves it.

## Brand Commitments

The name VidLiner is confirmed by the repository. No additional visual identity, typeface, logo asset,
or marketing claim is confirmed yet.

## Evidence on Hand

The repository contains the workflow canvas implementation in `vidliner/reports/workflow.py`,
`vidliner/web/canvas.js`, and `vidliner/web/canvas.css`, plus runnable examples under `examples/video`.
There are no confirmed customer studies, production-quality benchmarks, or approved brand assets in
the repository; future UI work must not fabricate them.

## Product Principles

- Make the graph the primary working surface.
- Show execution state without hiding uncertainty.
- Keep provider differences behind explicit runtime contracts.
- Preserve user edits and remote task evidence across restarts.
- Prefer inspectable, keyboard-accessible controls over gesture-only interactions.

## Accessibility & Inclusion

The canvas must remain usable with keyboard focus, visible focus rings, text alternatives for graph
nodes and connections, pointer and keyboard paths for wiring, readable contrast, and a mobile layout
that keeps the graph visible while secondary controls move below it.

<!-- The audience and visual preferences above are inferred from the explicit redesign request and
repository evidence; confirm them before treating them as permanent product strategy. -->
