# Runtime Profile（运行时配置）

[English](../runtime.md)

Recipe 说**要什么**，runtime profile 说**这台机器上谁来干**。正是这一分离，让同一份 recipe 今天跑内置启发式、明天跑本地模型、下周跑云端服务，而无需改动 recipe。

---

## profile 从哪里来

1. 命令行 `--runtime <path>`；
2. 环境变量 `$VIDLINER_RUNTIME`；
3. 工作区根目录的 `runtime.yaml`；
4. 以上都没有时，使用内置的本地 profile。

`vidliner init` 会写出本地 profile——这就是新克隆的仓库无需任何下载就能跑通全流程的原因。

---

## 结构

```yaml
profile: workstation
device: cpu                       # cpu | cuda | mps | auto

storage:
  root: .
  keep_rejected: true
  max_artifact_mb: 256
  state_filename: state.db

concurrency:
  workers: 4                      # 最大并发节点数
  per_backend: 2                  # 每个 backend 的默认并发上限
  external_requests: 3            # 预留给网络型 backend

backends:
  local_detector:
    use: vidliner.backends.heuristic.detector:HeuristicDetectorBackend
    options: { score_floor: 0.30 }
    capacity: { limit: 2 }
    estimates: { unit_seconds: 0.4 }
    demo_only: false              # 占位实现写 true；见下文"演示 backend"

  http_replacement:
    use: vidliner.backends.http_replacement:HttpReplacementBackend
    options:
      endpoint: http://127.0.0.1:8080/v1/edit
      request_timeout_s: 120
      max_response_mb: 32
      allowed_media_types: [image/png, image/jpeg, image/webp]
    credentials:
      api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
    capacity: { limit: 1, period_s: 6 }
    estimates: { unit_cost: 0.02, unit_seconds: 9, external: true }

bindings:
  vision.object_detection.v1: local_detector
  generation.object_replacement.v1: http_replacement
```

---

## `use`

导入路径，形式为 `package.module:ClassName` 或 `package.module.ClassName`。该类必须暴露 `backend_id`、`backend_version`、`capabilities`、`probe()`，并至少实现一个能力协议。

Backend 可以用类属性 `declared_capabilities` 声明能力，也可以写在 spec 的 `options.declared_capabilities` 里。声明的好处是：`vidliner plan` 能在**不导入 backend 模块**的前提下解析绑定——这正是规划阶段又快又无依赖的原因。

---

## 演示 backend

`demo_only: true` 把一个 backend 标记为**占位实现**：它能驱动整条流水线，却没有真正测量其能力所声称之事。随附的本地 profile 把它所有的感知、生成与评估 backend 都标成了这样，而这些类自己也会声明——类属性 `demo_only = True` 即使被 profile 条目漏掉，也依然会被捕获。

这条标记只有一个后果。有四项能力——检测、实例分割、对象替换、语义匹配——产出的是最终进入数据集的数据；其中任何一项由演示 backend 提供，job 照常运行，但导出会被拒绝，错误码为 `DEMO_BACKEND_NOT_ALLOWED`。recipe 可以用 `acceptance.allow_demo_backends: true` 覆盖它，而这个事实会被记进 manifest，因此产出的数据集事后仍可审计。其余能力出现占位实现不会阻塞导出：演示级 planner 或 refiner 改变的是样本怎么做出来的，而不是它的标签是否描述了画面。

`vidliner plan` 会在生成任何东西之前报告该状况：

```bash
vidliner plan recipe.yaml
# DEMONSTRATION stack: the export would be refused
#   vision.object_detection.v1                   -> builtin_detector (demonstration stand-in)
#   ...
```

完整规则见[编写 Backend](./backends.md#演示-backend-与生产防护)，能力清单本身在 `vidliner/capabilities/names.py` 的 `PRODUCTION_CAPABILITIES`。

---

## 绑定解析

按顺序进行，绝不猜测：

1. 显式 `bindings[capability]` 优先；
2. 否则使用声明能力中包含该能力的 backend，前提是**恰好一个**匹配；
3. 再否则实例化所有已启用 backend 并询问它们各自提供什么；
4. 零个候选 = 能力未绑定；多个候选 = 歧义错误，并列出候选名。

`vidliner plan` 会报告未绑定的能力并以退出码 `1` 结束；`vidliner run` 会拒绝启动。

---

## 容量控制

`capacity.limit` 限制进入该 backend 的并发调用数。`capacity.period_s` 增加一个可补充的速率预算：在任意长度为该值的窗口内，最多只允许 `limit` 次调用**开始**——这是在不把客户端串行化的前提下尊重云端服务"每分钟请求数"策略的方式。

`blocking: true` 的 backend（本地模型、解码器、子进程）在 worker 线程中执行，从而让 job 的事件循环继续调度其他节点；网络型 backend 直接 await，并从独立预算中取用。

---

## 凭据

凭据在仓库内一律是**引用**，绝不是值：

```yaml
credentials:
  api_key: { source: env,  name: VIDLINER_EDIT_API_KEY }
  token:   { source: file, name: /run/secrets/edit-token }
```

`source: value`（内联明文）在 profile 位于带 VCS 标记的目录内时会被直接拒绝，其他情况也会被 `vidliner backend check` 标记。

实际的解析只发生在 `vidliner/runtime/secrets.py`。解析出的值只在运行期驻留内存，会在每一条日志与每一份序列化 profile 中被脱敏，并且永远不会写进 recipe、manifest、plan、日志或溯源记录。端到端测试会断言这一点。

---

## 估算

`estimates` 只服务于 `vidliner plan`：

| 字段 | 含义 |
| --- | --- |
| `unit_cost` / `currency` | 每次调用的金额。 |
| `unit_seconds` | 每次调用的耗时。 |
| `external` | 计入"外部调用"数。 |
| `accelerator` | 计入"加速器操作"数。 |

若某个 backend 什么都没声明，该阶段仍会被计数，但成本与耗时按 0 计，且 plan 会在备注里如实说明，而不是编造一个平均值。

---

## 预检

```bash
vidliner backend list    --workspace ./playground     # 当前配置了什么
vidliner backend check   --workspace ./playground     # 探测并报告覆盖率
vidliner backend list --capabilities                  # 能力词汇表
```

由流水线自行测量的能力（当前为 `quality.background_preservation.v1`）会显示为 `builtin` 已满足；它们不需要 backend，也不应出现在 `bindings` 中。

`check` 只调用 `probe()`。探测必须是廉价的，且绝不能发起付费或生成调用：它只检查凭据是否存在、端点是否可达、模型文件与设备是否可用。报告会列出每个 backend 的 `backend_id`、`version`、健康状态（`ready`、`degraded`、`unavailable`）、所声明能力、确定性、`safe_to_retry`、设备，最后列出完全没有 backend 的能力。

---

## 确定性与重试

Backend 要声明自身的可复现程度（`deterministic`、`seeded`、`nondeterministic`）以及重试同一请求是否安全。引擎两者都会用：

* 只有确定性或 seeded 的节点才会被缓存；
* 除非 backend 声明 `safe_to_retry`，生成类操作不会被重试——因为重试可能产出不同图像并产生又一次付费调用。

---

## 替换内置 backend

要拿到生产级精度，最快的路径是保留流水线、只换四项能力：

```yaml
backends:
  my_detector:
    use: my_project.detector:DetectorBackend
    options: { model_path: /models/yolo.onnx, device: cuda }
    estimates: { unit_seconds: 0.05, accelerator: true }
  my_segmenter:
    use: my_project.segmenter:SegmenterBackend
    options: { model_path: /models/sam.onnx }
  my_editor:
    use: my_project.editor:EditorBackend
    credentials: { api_key: { source: env, name: EDIT_KEY } }
  my_judge:
    use: my_project.judge:SemanticJudgeBackend

bindings:
  vision.object_detection.v1: my_detector
  vision.instance_segmentation.v1: my_segmenter
  generation.object_replacement.v1: my_editor
  quality.semantic_match.v1: my_judge
```

其余一切不变，导出也不再被拒绝。契约见 `docs/zh/backends.md`。
