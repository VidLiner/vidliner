# VidLiner —— DAG 设计

[English](../../design/dag.md)

## 1. 为什么是图

流水线不是一条直线：一张图里可能有多个目标对象，一个对象会产出多个候选，每个候选被独立精修、评估、重建标注并判定。写成嵌套循环会得到一个无法规划、无法缓存、无法恢复、无法估算的程序；写成图，这四项性质自然产生。

## 2. 节点契约

```python
@dataclass(frozen=True)
class OperationNode:
    node_id: str            # 确定性身份（ADR-016）
    operator: str           # 已注册 operator 名
    operator_version: str
    stage: PipelineStage    # ingest | detect | segment | scene | plan | generate |
                            # refine | verify | evaluate | annotate | gate | export
    inputs: Mapping[str, PortBinding]   # 端口名 -> 绑定
    outputs: Mapping[str, PortType]     # 声明的输出端口
    config: Mapping[str, Any]           # 已校验、可规范化序列化
    needs: tuple[str, ...]              # 运行期必须绑定的能力
    retry: RetryPolicy
    timeout_s: float | None
    determinism: Determinism            # deterministic | seeded | nondeterministic
    cacheable: bool
    max_parallelism: int
    selects: tuple[tuple[str, int], ...] # 需要从生产者元组中取第几个元素
    lineage: tuple[str, ...]            # 用于种子派生与展示的身份路径
```

`PortBinding` 有四类：`NodeOutput(node_id, port)`、`SampleSource(sample_ref)`、`ArtifactSource(role)`、`Constant(value)`。图校验器会拒绝指向不存在节点/端口的引用，以及端口 `PortType` 与消费方声明不符的边。端口类型有 `image`、`image_ref`、`mask_ref`、`instances`、`scene_context`、`plan`、`candidate_ref`、`quality_report`、`annotation`、`dataset_slice`、`metrics`、`flag`、`any`。

## 3. MVP 图形状

装配器（`vidliner.pipeline.assemble`）构建如下结构。`*` 表示扇出点。

```
每个样本 S:
  ingest:S            (media_asset)                ── 无能力需求
  detect:S            (instances)                  ── 需要 vision.object_detection.v1
  select:S            (筛选后的 instances)          ── 纯函数
  segment:S           (instances + masks)          ── 需要 vision.instance_segmentation.v1

  每个被选中的对象 O: *
    scene:S:O         (scene_context)              ── 需要 vision.scene_analysis.v1
    plan:S:O          (plan)                       ── 需要 planning.replacement.v1

    plan 的每个候选 C: **
      generate:S:O:C      (candidate_ref)          ── 需要 generation.object_replacement.v1
      refine:S:O:C        (refined image/mask)     ── 无能力需求（纯精修）
      verify:S:O:C        (item + mask + found)    ── 需要 vision.object_detection.v1
                                                      与 vision.instance_segmentation.v1
      evaluate:S:O:C      (quality_report)         ── 需要 quality.* 评估能力
      annotate:S:O:C      (annotation)             ── 纯函数（用 verify 的 mask 重建标注）
      export:S:O:C        (per-sample evidence)    ── 纯函数（落盘证据）
```

扇出键是显式身份而不是位置：`O` 是 `object:<object_id>`，`C` 是 `candidate:<candidate_key>`。因此往 recipe 里增加一个候选值，绝不会改变已有候选的节点 ID——这正是跨运行缓存复用能够成立的原因。

`object:N` 分支数是按 `target.max_per_sample` **预留**的。样本实际对象少于预留数时，多出的分支连同其全部下游节点被标记为 `skipped`（而不是 failed），这就是"一张固定形状的图能服务所有样本"的机制。每个节点通过 `selects` 显式声明"从 `items` 元组取第 index 个"。

## 4. 引擎职责

| 关注点 | 机制 |
| --- | --- |
| 校验 | `OperationGraph.validate()` —— id 唯一、无环、端口类型一致；能力可用性在 `plan`/`run` 前对着 runtime profile 做预检 |
| 调度 | Kahn 式入度计数；依赖一满足即派发 |
| 并行 | `asyncio`；全局 worker 信号量 + 每 backend 信号量；阻塞型 backend 走 `asyncio.to_thread` |
| 重试 | `RetryPolicy(attempts, backoff_s, jitter, retry_on)`；backend 声明 `safe_to_retry`，生成类默认 `attempts=1` |
| 超时 | 每次尝试用 `asyncio.wait_for` 包裹 |
| 取消 | 单个 `asyncio.Event`；派发前检查，长等待中也会被取消 |
| 恢复 | 从 `state.db` 载入已完成节点；只有当记录的输出摘要仍然存在且可验证时才跳过 |
| 缓存 | 见 ADR-005 的缓存键；命中即跳过并记 `cache_hit` |
| 失败隔离 | 节点失败只影响其后代；兄弟候选继续；`--fail-fast` 时首个失败即停 |
| 质量拒绝 | 永不作为节点失败；`evaluate`/`gate` 正常完成并产出 `rejected` 决策 |
| 日志 | 每行一个 JSON 事件；`events.jsonl` + `state.db` 每个节点一行 |

## 5. 状态机

Job：

```
CREATED → PLANNED → RUNNING → (WAITING) → FINALIZING → SUCCEEDED
                        └──────────────────────────────→ FAILED
                        └──────────────────────────────→ CANCELLED
```

`WAITING` 用于"所有可运行节点都在等外部异步 backend（submit/poll）"的情形；下一次轮询成功后回到 `RUNNING`。`FINALIZING` 覆盖导出与报告写出。

Candidate：

```
GENERATED → EVALUATING → ACCEPTED
                       → REJECTED
                       → REVIEW        （需人工判定；未判定前不导出）
```

状态迁移在 `candidate_events` 中只追加不修改；当前状态另有一列以加速查询。界面永不推断状态（需求 §15）。

## 6. Dry run

`vidliner plan --dry-run` 会构建图、解析绑定并打印：

* 样本数、目标对象数、候选数；
* 每项能力选中的 backend；
* 预估 GPU 操作数（绑定 backend 声明了 `cpu`/`cuda` 的节点）；
* 预估外部 API 调用数与成本（来自 profile 中的 `Estimate`）；
* 按拓扑阶段排序的节点数统计。

`plan` 期间**从不**调用任何生成 backend——它会检查能力*解析*，但不调用生成能力。`plan` 还会把计划写入 `runs/<job>/plan.json`，因此 `run` 可以执行被审阅过的同一张图。
