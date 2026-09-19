# 架构

本文说明 VidLiner 的组织方式及其原因，目标是让新贡献者不必读完整个代码库就能找到改动该落在哪一层。

单项决策的理由见 `docs/zh/design/decisions.md`。

---

## 1. 问题的形状

训练数据生产的成功标准与图像生成不同：

* **执行单元**是跨帧跟踪的对象实例（tracked object instance），不是帧；
* **交付物**是图像 + 正确标签 + 证据；
* **真正致命的失败**是"看起来合理、但标签错了或背景变了"的样本——因为它教给模型错误的东西。

因此 VidLiner 被构建为**一条验证流水线，生成只是其中的一步**，而不是"一次生成调用 + 事后补几个检查"。

---

## 2. 分层

包是分层的，依赖方向由测试强制（`tests/unit/test_architecture.py`）：下层不得导入上层，且 `backends/` 之外不得导入任何厂商 SDK。

```
cli/                 用户界面（Typer）
  │
pipeline/            编排：recipe → 图 → job → 导出
  │
operators/           领域操作（每个阶段一个模块）
  │
capabilities/        抽象端口：能力名 + 协议
  │
domain/  core/       带类型的数据     与   基元（身份、错误、图、结果）
  │
storage/  control/   内容寻址产物 + SQLite 状态      数据集发现、split、报告
  │
runtime/             profile 加载、backend 注册表、引擎、日志
  │
backends/            唯一允许出现厂商 SDK 或模型运行时的地方
```

| 层 | 该放什么 | 不该放什么 |
| --- | --- | --- |
| `domain` | Pydantic 模型、枚举、校验规则 | I/O、模型调用、文件系统访问 |
| `core` | 规范 JSON、身份、种子、错误、图、结果 | 领域专有知识 |
| `capabilities` | 能力名、backend 协议、`PipelineContext` | 任何具体 backend |
| `operators` | 一个领域操作及其声明的端口与配置 | 厂商导入、直接文件系统访问 |
| `backends` | 适配器：启发式、fake、本地模型、HTTP 服务 | 流水线策略 |
| `runtime` | profile、注册表、容量、事件日志、**引擎** | 领域逻辑 |
| `pipeline` | recipe 编译、图装配、job runner、导出、策略 | operator 内部实现 |
| `storage` | 工作区布局、产物存储、SQLite 状态 | 领域决策 |
| `control` | 发现、split、去重、分布、复核 HTML | 执行 |
| `cli` | 参数解析、渲染、退出码 | 业务逻辑 |

`core` 只允许引用 `domain` 的两个基元模块（`enums`、`shapes`）；它**不**导入 operator 目录——那个依赖是注入进去的（见下方"图是设计中心"）。

---

## 3. 图是设计的中心

Recipe 被编译成由 `OperationNode` 组成的 `OperationGraph`。每个节点声明：

* **inputs** —— 带类型的端口，绑定到另一节点的输出、样本值、job 级产物或常量；
* **outputs** —— 带类型的端口，图校验器会逐个比对消费方；
* **config** —— 由 operator 的 Pydantic 模型在**规划期**校验，因此配置错误绝不会拖到 job 中途才暴露；
* **capabilities** —— 运行它必须绑定的能力；
* **determinism**、**cacheability**、**retry policy**、**timeout**、**parallelism**；
* **lineage** —— 身份路径（`sample → object → candidate`），用于种子与报告。

MVP 图形状（`*` 表示扇出）：

```
每个样本 S
  ingest:S ──▶ detect:S ──▶ select:S
                              │
                              ├─▶ 每个对象 O:  segment ─▶ scene ─▶ plan
                              │                        │
                              │                        └─▶ 每个候选 C:
                              │                              generate ─▶ refine ─▶ verify ─▶ evaluate ─▶ annotate ─▶ export
                              └──────────────────────────────────────────────────────────────────────────────┘
```

扇出由**显式身份**（`object:<id>`、`candidate:<key>`）决定，绝不依赖列表位置。因此往 recipe 里加一个候选值不会重新编号已有候选——这正是跨运行缓存复用能够正确的前提。

`verify` 是让流程闭合的阶段：它在**生成图**上重新检测对象并重新分割，因此几何指标、边界指标与导出的标注，描述的都是真实存在的那个对象，而不是交给生成器的那个 mask。详见 ADR-018。

**不存在的分支是"跳过"，不是"失败"。** 装配器会预留 `target.max_per_sample` 个对象分支，以保证节点身份稳定；只有 1 个对象的样本在其余分支上本就无事可做，于是该分支及其全部下游节点都被标记为 skipped。

---

## 4. 执行

`runtime/engine.py` 是单进程 `asyncio` 调度器。它只管六件事，其余全部委派：

| 关注点 | 机制 |
| --- | --- |
| 调度 | Kahn 式入度计数；依赖一满足就立即派发 |
| 并行 | 每轮派发受全局 worker 预算约束，另有每节点信号量 |
| 重试 | 节点自身策略，并受 backend 是否声明"可安全重试"进一步限制 |
| 超时 | 每次尝试独立超时 |
| 取消 | 单个 `asyncio.Event`，派发前与等待中都会检查 |
| 缓存与恢复 | 被解析出的节点，其端口值在**其下游被派发之前**完成回填 |

引擎从不直接与 backend 通信。它让 `NodeRunner` 执行单个节点；生产实现是 `pipeline/executor.py`，它负责：

1. 解析节点的输入绑定，并施加按分支的选取；
2. 构造 `ExecutionContext`：节点、种子、产物 I/O、样本值、backend 解析器、取消事件；
3. 调用 operator；
4. 把 operator 的返回值编码进 `NodeResult`，其 payload 能跨进程重启存活。

**质量拒绝不是失败。** 门槛是数据：`pipeline/policy.py` 把 recipe 变成 `AcceptancePolicy`，再套用到质量证据上，产出带 reason code 的 `AcceptanceDecision`。`vidliner qa` 对已存证据运行同一个函数——这就是为什么改阈值不花一分钱。

---

## 5. 能力、backend 与 profile

三份文档决定实际跑什么：

* **recipe** 说想要什么，且从不指名厂商；
* **runtime profile** 说这台机器上谁来提供每项能力；
* **backend 契约** 说一个 backend 必须能做到什么。

```yaml
# runtime.yaml
bindings:
  vision.object_detection.v1: local_detector
  generation.object_replacement.v1: http_replacement
```

解析顺序（绝不猜测）：显式 binding 优先；否则使用声明能力中包含它的 backend，前提是**恰好一个**匹配；再否则实例化已启用的 backend 并询问。零个或多个候选都构成预检失败，并给出明确信息。

Backend 拿到的是 `PipelineContext`——job id、node id、seed、device、config 与 `ArtifactIO`。它无法解析工作区路径、无法打开状态数据库、无法读取 recipe。

**词汇表中有一项能力刻意不由 backend 提供**：`quality.background_preservation.v1` 是流水线自己做的确定性像素比较。它留在词汇表里（recipe 的门槛会引用它），但 profile 不得绑定它，`vidliner backend check` 会把它报告为 `builtin` 已满足。

Backend 还可以声明 `demo_only`，随附 profile 用这个标记标出了自己的启发式、合成生成器与颜色评估器。这条标记喂给**生产防护**：检测、实例分割、对象替换、语义匹配这四项能力产出的正是最终进入数据集的数据，因此其中任何一项由占位实现提供时，job 照常运行，但**导出会被拒绝**——除非 recipe 设置 `acceptance.allow_demo_backends`。检查只读这个标记，所以在规划阶段零成本；它会跑两次：一次在生成任何东西之前的预检，一次在导出时，且后者依据存储的 manifest 而非当前 profile。参见 ADR-019。

阻塞型 backend（本地模型、解码器）声明 `blocking = True`，并把同步实现放在 `self._sync`；运行时会在线程中执行它。原生异步 backend（HTTP 客户端）直接 await，并从独立的容量预算中取用。

---

## 6. 存储

```
<workspace>/
  runtime.yaml
  state.db                      SQLite：job、node、candidate、metric、cache、lineage、duplicate
  artifacts/<aa>/<digest>.<ext> 内容寻址产物存储
  runs/<job_id>/
    plan.json  events.jsonl  job-summary.json
    samples/<sample_id>/
      source.png                  原始帧，用于对比
      refined-<candidate>.png     精修后的候选
      annotation-<candidate>.json 重建后的标注
      quality-<candidate>.json    质量报告
      decision-<candidate>.json   带 reason code 的决策
      candidate-<candidate>.json  生成载荷（意图、seed、backend 元数据）
    review/index.html           静态复核报告
    dataset/                    导出数据集（由 `vidliner export` 生成）
```

像素以 SHA-256 为键存在文件系统上，状态存在 SQLite 中。缓存或恢复的结果只有在产物被验证确实存在后才会被采用——因此损坏的缓存永远不会被误当成有效结果。

样本级证据是**按候选**存放的：一个样本产出三个候选，就保留三套证据，后写入的不会覆盖先写入的。

---

## 7. 扩展点

| 想加什么 | 怎么做 |
| --- | --- |
| 检测 / 分割 / 生成模型 | 写一个实现协议的 backend，然后在 `runtime.yaml` 绑定能力 |
| 新的导出格式 | 在 `annotations/` 加转换器，并在 `ANNOTATION_FORMATS` 里登记 |
| 新的质量指标 | 加入 `MetricName`，在质量阶段计算，并在 recipe 门槛中引用 |
| 新的精修步骤 | 加能力名、加 operator、加 recipe step——装配器会自动接上 |
| 视频支持 | 实现跟踪与视频替换能力；数据模型、引擎扇出、时序指标都已就绪 |
| 不同的执行后端 | 实现 `NodeRunner`；引擎对此无感 |

---

## 8. MVP 刻意不做的事

没有分布式 worker、消息队列、GPU 调度器、对象存储、Web UI，也没有多用户认证。上述扩展点就是它们未来接入的缝，阶段规划记录在 `docs/zh/design/milestones.md`。
