# VidLiner Canvas Design

<!-- impeccable:design-schema 1 -->

## Surface

The workflow canvas is an Operate surface for building and executing typed video workflows.

## Visual World

**Route-map workbench.** The canvas reads like a technical orienteering map: the graph is terrain,
nodes are control points, the active execution path is purple, and yellow marks attention or a state
that needs review. The visual world is useful because it makes topology and state legible without
turning the editor into a decorative diagram.

## Composition

- A dark ink masthead anchors the product name and current node/connection count.
- A paper toolbar holds search, route fit, zoom, layout, export, execution, and seed controls.
- The graph is the largest region and uses a quiet square grid as its coordinate field.
- The right control-point panel is a fixed desktop inspector for configuration, evidence, operator
  palette, and execution trace; it moves below the graph on narrow screens.

## Tokens

```css
--ink: #181b24;
--paper: #fffdf7;
--sand: #ece8dd;
--line: #d8d2c5;
--muted: #6e7b70;
--route: #9848d8;
--signal: #f1c84b;
```

Use square or 2px corners, thin neutral rules, generous paper space, and no gradients. Purple is
reserved for the active route, selected node, primary action, and generation status; yellow is reserved
for attention and review. Do not introduce a competing accent family without revisiting this contract.

## Interaction

Pointer and keyboard both connect typed ports. Dragging moves nodes only in Edit layout mode. Fit and
zoom own the canvas gesture; the page remains scrollable on mobile. The primary action is Execute;
remote cancellation stays explicit and visually secondary. Selected nodes expose configuration and
evidence in the right inspector rather than opening a blocking modal.

## Responsive Contract

At 1440px the graph and inspector sit side by side. At 390px the toolbar wraps, the graph remains the
first evidence region with a minimum 360px height, and the inspector follows below it. There is no
horizontal page overflow. Reduced-motion users receive the same final states without transitions.

## Content And Evidence

Keep runtime state vocabulary literal: queued, running, succeeded, failed, cancelled, interrupted.
Never imply that a provider URL is verified training data. Generated media and task evidence remain
secondary to the graph and are presented in the execution trace.
