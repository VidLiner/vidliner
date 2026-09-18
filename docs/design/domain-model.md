# VidLiner — Domain Model

Every object below is a Pydantic v2 model in `vidliner.domain`. Values are frozen (immutable)
unless noted: the pipeline produces new objects instead of mutating shared ones.

---

## 1. Media and samples

| Object | Purpose | Key fields |
| --- | --- | --- |
| `MediaAsset` | One immutable source file, never modified. | `path` (workspace-relative), `kind` (`image`/`video`), `digest` (SHA-256 of bytes), `size_bytes`, `media_type`, `shape` (`ImageShape`), `video` (`VideoSpec` \| None), `exif` |
| `VideoSpec` | Video container facts. Reserved for Phase 2, already modelled. | `fps`, `frame_count`, `duration_s`, `codec` |
| `FrameRef` | Addresses one frame inside a media asset. | `asset_digest`, `frame_index`, `timestamp_s` |
| `DatasetSample` | One training unit: a media asset plus its annotations plus lineage. | `sample_id`, `asset`, `in_split`, `annotation`, `lineage`, `origin` |
| `SampleLineage` | Leakage control. | `root_digest`, `ancestors`, `augmentation_depth`, `parent_sample_id` |
| `DatasetRef` | A discovered dataset directory and its manifest if present. | `root`, `samples`, `format`, `splits` |

`sample_id` is derived deterministically: `sha256(root_digest + ":" + relative_path)[:16]` prefixed
with `s_`. Two copies of the same file at different paths are the same asset (same digest) but
different samples; the lineage validation uses the digest, so leakage is still caught.

---

## 2. Objects in a frame

| Object | Purpose | Key fields |
| --- | --- | --- |
| `BoundingBox` | Pixel-space box, `xyxy`, clipped to the image, plus `normalized` helper. | `x_min`, `y_min`, `x_max`, `y_max` |
| `PolygonMask` | One or more closed rings in pixel space. | `rings: list[list[Point]]` |
| `BinaryMask` | RLE-free compact mask reference: a `MaskRef` pointing at an artifact plus shape and area. | `mask`, `shape`, `area_px`, `coverage` |
| `MaskRef` | Digest-addressed pointer to a stored mask artifact. | `artifact` (`ArtifactRef`), `encoding` (`png-l8`/`npy-bool`/`rle`) |
| `DepthMap` | Reserved; digest-addressed depth artifact plus scale. | `depth`, `min_depth`, `max_depth`, `metric` |
| `ObjectInstance` | One detected/segmented object in one frame. | `object_id`, `class_name`, `bbox`, `score`, `mask_ref`, `polygon`, `attributes`, `source_frame`, `track_id`, `area_px` |
| `ObjectTrack` | Cross-frame association of one instance. Reserved for Phase 2. | `track_id`, `class_name`, `observations`, `first_frame`, `last_frame`, `visibility`, `reentry_count` |
| `TrackObservation` | One frame's evidence for a track. | `frame_index`, `bbox`, `mask_ref`, `visibility`, `score` |
| `SceneContext` | Environment facts used by the planner. | `object_category`, `orientation_deg`, `scale_class`, `light_direction_deg`, `light_intensity`, `shadow_polygon`, `depth_order`, `occluders`, `background_summary`, `ground_contact` |

`object_id` is `sha256(asset_digest + frame_index + class_name + rounded bbox + score bucket)[:12]`
prefixed with `o_`, so re-running detection on the same image yields the same object identity and
therefore the same node identity (ADR-016).

---

## 3. Replacement

| Object | Purpose | Key fields |
| --- | --- | --- |
| `ReplacementIntent` | *What the user wants*, independent of any model. | `target_object`, `replacement_category`, `replacement_description`, `preserve_pose/scale/position/lighting/occlusion`, `allowed_geometry_change`, `mode` (`strict`/`creative`), `seed` |
| `ReplacementPlan` | The planner's decision for one target object. | `intent`, `mask_expansion_px`, `depth_expansion_px`, `positive_prompt`, `negative_constraints`, `expected_category`, `geometry_budget`, `candidate_specs`, `plan_seed`, `planner_id` |
| `CandidateSpec` | One variation to generate. | `candidate_key`, `description`, `category`, `seed`, `reference_artifact` |
| `GenerationRequest` | The uniform request every generation backend must accept. | `source_image`, `target_mask` (`MaskRef`), `context_crop`, `positive_prompt`, `negative_constraints`, `candidate_key`, `seed`, `reference_image`, `expected_category`, `parameters` |
| `GenerationOutcome` | The uniform reply. | `image` (`ArtifactRef`), `backend_id`, `model_id`, `seed`, `duration_ms`, `cost_estimate`, `raw_metadata` |
| `ReplacementCandidate` | One concrete generated candidate and its lifecycle. | `candidate_id`, `sample_id`, `target_object_id`, `spec`, `state`, `generated`, `refined`, `quality`, `annotation`, `decision`, `artifacts` |
| `RefinementRecipe` / `RefinementStep` | Declarative description of post-processing. | `steps`, each `step` names a refinement operator and its config |
| `AugmentedSample` | An accepted (or reviewable) output sample. | `sample_id`, `source_sample_id`, `candidate_id`, `image`, `annotation`, `lineage`, `quality_summary`, `provenance` |

`candidate_id` is `sha256(sample_id + target_object_id + candidate_key)[:16]` prefixed with `c_`.

---

## 4. Annotations (internal IR)

| Object | Purpose |
| --- | --- |
| `AnnotationBundle` | Format-neutral annotation set for one sample: `image_shape`, `categories`, `objects`, `source_format`, `attributes` |
| `AnnotationObject` | One labelled object: `object_id`, `category_id`, `bbox`, `polygon`, `mask_ref`, `score`, `iscrowd`, `track_id`, `attributes` |
| `CategorySpec` | `category_id`, `name`, `supercategory` |
| `ExportProfile` | Requested output format and options: `format`, `path`, `copy_images`, `min_area_px`, `include_score` |

The IR is the only thing operators read and write. COCO and YOLO support is implemented as
converters around it (`vidliner.annotations`), so adding a format never touches the pipeline.

---

## 5. Job, evidence, provenance

| Object | Purpose | Key fields |
| --- | --- | --- |
| `JobManifest` | Immutable record of one job attempt. | `job_id`, `recipe_name/hash/snapshot`, `runtime_snapshot`, `seed`, `created_at`, `finished_at`, `state`, `counters`, `node_count`, `backend_ids`, `operator_versions`, `git_revision` |
| `OperatorEvidence` | One node execution's evidence. | `node_id`, `operator`, `operator_version`, `backend_id`, `input_digests`, `output_digests`, `started_at`, `duration_ms`, `status`, `attempt`, `cache_hit`, `error_code` |
| `QualityReport` | Per-candidate metric evidence. | `candidate_id`, `metrics: list[MetricOutcome]`, `overall_score`, `decision`, `reason_codes`, `evaluated_at`, `policy_hash`, `auxiliary` |
| `MetricOutcome` | One metric's measured value vs threshold. | `metric`, `value`, `threshold`, `comparison`, `severity`, `status`, `reason_codes`, `evaluator_id`, `detail` |
| `AcceptanceDecision` | `state` (`accepted`/`rejected`/`needs_review`), `reason_codes`, `policy_hash`, `failed_hard_gates` |
| `ProvenanceRecord` | The cross-artifact link written per accepted sample. | source sample/digest, recipe hash, job id, target object, intent, seeds, operator versions, backend/model ids, generation params, QA scores, decision, output digest, timestamp |
| `ReasonCode` | Closed enum of machine-readable rejection/review reasons. | see `vidliner/domain/reasons.py` |

---

## 6. Runtime configuration objects

| Object | Purpose |
| --- | --- |
| `RuntimeProfile` | `name`, `device`, `backends: dict[str, BackendSpec]`, `bindings: dict[CapabilityName, BackendName]`, `concurrency`, `storage`, `estimates` |
| `BackendSpec` | `use` (import path), `options` (free-form validated by the backend), `credentials` (references), `capacity` |
| `CredentialRef` | `source` (`env`/`file`/`value`), `name`, plus resolution in the backend layer only |
| `CapacityPolicy` | `limit`, optional `period_s` (rate budget), `weight` |
| `Estimate` | `unit_cost`, `unit_seconds`, `currency`, `external` — used by `plan`/`--dry-run` |

---

## 7. Identity and determinism rules

1. Every digest is lowercase hex SHA-256 of raw bytes.
2. Every derived object id is `prefix + sha256(canonical-json(payload))[:n]`.
3. Canonical JSON: sorted keys, no whitespace, `None` omitted, floats formatted with `repr`-stable
   shortest round-trip. Implemented in `vidliner.core.canonical`.
4. `LineageKey` strings are colon-joined, never re-ordered: `sample:<id>`, `object:<id>`,
   `candidate:<key>`.
5. Seed derivation: `seed(parent, label) = int(sha256(f"{parent}|{label}").hexdigest()[:16], 16)`.
   The job seed comes from the recipe or the CLI; nothing else introduces entropy.
