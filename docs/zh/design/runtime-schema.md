# VidLiner —— Runtime Profile Schema

[English](../../design/runtime-schema.md)

Runtime profile 只回答一个问题：*在这台机器上，哪项能力由哪个 backend 提供？*
它与 recipe 分离，因此同一份 recipe 既能对本地模型、也能对本地 HTTP 推理服务或云端生成服务运行，而无需改动。

文件：工作区根目录的 `runtime.yaml`，可用 `--runtime <path>` 或环境变量 `VIDLINER_RUNTIME` 覆盖。

```yaml
profile: workstation          # 必填，slug
device: cpu                   # cpu | cuda | mps | auto（提示性；backend 可覆盖）

storage:
  root: .                     # 工作区根；相对路径相对 profile 文件解析
  keep_rejected: true         # 保留被拒产物用于诊断
  max_artifact_mb: 256        # 拒绝大于该体积的产物

concurrency:
  workers: 4                  # 全局有界并行
  per_backend: 2              # 每 backend 默认上界
  external_requests: 3        # 网络型 backend 的独立上界

backends:
  local_detector:
    use: vidliner.backends.heuristic.detector:HeuristicDetectorBackend
    options:
      score_floor: 0.3
    capacity: { limit: 2 }
    estimates: { unit_seconds: 0.4 }
    demo_only: true             # 显著性启发式，不是检测器

  local_segmenter:
    use: vidliner.backends.heuristic.segmentation:HeuristicSegmentationBackend
    options: { dilate_px: 2 }
    estimates: { unit_seconds: 0.6 }
    demo_only: true

  local_scene:
    use: vidliner.backends.heuristic.scene:HeuristicSceneBackend
    demo_only: true

  local_planner:
    use: vidliner.backends.heuristic.planner:RuleBasedPlannerBackend
    demo_only: true

  http_replacement:
    use: vidliner.backends.http_replacement:HttpReplacementBackend
    options:
      endpoint: http://127.0.0.1:8080/v1/edit
      request_timeout_s: 120
      max_response_mb: 32
      allowed_media_types: [image/png, image/jpeg, image/webp]
      response_image_field: image_base64
    credentials:
      api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
    capacity: { limit: 1, period_s: 6 }
    estimates: { unit_cost: 0.02, unit_seconds: 9, external: true }

  local_evaluator:
    use: vidliner.backends.local.evaluator:LocalMetricEvaluatorBackend
    demo_only: true

bindings:
  vision.object_detection.v1: local_detector
  vision.instance_segmentation.v1: local_segmenter
  vision.scene_analysis.v1: local_scene
  planning.replacement.v1: local_planner
  generation.object_replacement.v1: http_replacement
  quality.semantic_match.v1: local_evaluator
  quality.artifact_detection.v1: local_evaluator
```

## 规则

1. `use` 是 `module:Attribute` 或 `module.Attribute` 形式的导入路径。Backend 类必须至少实现一个能力协议，并暴露 `backend_id`、`backend_version`、`capabilities`、`probe()`。
2. 编译后的图需要、但 profile 未绑定的能力，构成**预检校验失败**：`vidliner plan` 会报告，`vidliner run` 拒绝启动。
3. 当恰好只有一个已配置 backend 声明某能力时，binding 可省略；当有多个时，必须显式指定——解析器从不猜测。
4. 仓库内的 profile 中，`credentials` 一律是引用而非明文值。`{source: value, name: "..."}` 只在工作区边界之外的 profile 中被接受，且会被 `backend check` 标记。
5. `capacity.limit` 限制进入该 backend 的并发调用；`period_s` 增加可补充的速率预算（用于遵守云端服务的限流策略）。
6. `estimates` 只供 `plan`/`--dry-run` 使用，永不影响执行。
7. `device` 是提示性的：backend 可以声明设备要求，无法满足的 profile 会在预检阶段以 `DEVICE_UNAVAILABLE` 失败。
8. `demo_only` 把 backend 标记为占位实现。它只改变一件事：当 `PRODUCTION_CAPABILITIES` 中的某项能力解析到被标记的 backend 时，除非 recipe 设置 `acceptance.allow_demo_backends`，数据集不得导出。Backend 类自己也可以声明该标记，且以类声明为准。检查只读这个标记——绝不导入、实例化或探测 backend，因此在规划阶段零成本。
9. 由流水线自行测量的能力（`BUILTIN_CAPABILITIES`，当前为 `quality.background_preservation.v1`）**不得**出现在 `bindings` 中——它们不需要 backend。

## 脱敏

任何被序列化的 profile（进入 job manifest、plan 产物或日志事件）都会经过
`vidliner.runtime.secrets.redact_profile`，把解析后的凭据值替换为字面量 `"<redacted>"`。
端到端测试 `test_no_credential_material_reaches_the_manifest_or_dataset` 断言这一点。

## 预检（`vidliner backend check`）

对每个已配置 backend，命令行打印：`backend_id`、`backend_version`、所声明能力、探测结果
（`ready` / `degraded` / `unavailable` 及原因）、声明的确定性、`safe_to_retry`、设备，以及是否存在估算。
预检绝不执行生成调用：它只调用 `probe()`。内置测量型能力会显示为 `builtin` 已满足，不计入未覆盖清单。
