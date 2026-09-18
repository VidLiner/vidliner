# 测试

测试套件是本项目的主要安全网，它有一条不可妥协的性质：**绝不触碰网络、付费 API 或下载的模型**。一切都跑在进程内生成的合成 fixture 上。

```bash
pytest -q                       # 全部
pytest tests/unit -q            # 基元与模型
pytest tests/contract -q        # backend 协议
pytest tests/pipeline -q        # 端到端 job
ruff check .                    # 静态检查
ruff format --check .           # 格式
ty check vidliner               # 类型检查
```

当前规模：**348 个测试**，全部通过；`ruff` 与 `ty` 均无输出。

---

## 目录

```
tests/
  conftest.py            fixture：工作区、session、合成数据集、recipe 构造器
  unit/                  规范序列化与身份、领域模型、引擎、存储、质量、标注、架构规则
  contract/              通过协议逐个验证每个 backend
  pipeline/              端到端、导出、恢复与缓存、CLI
```

---

## Fixture

`vidliner/fixtures.py` 用 numpy 渲染合成场景：带纹理的背景、柔和的垂直渐变（让光照测量有东西可测）、以及一个或多个带明暗与底部暗边的对象。`build_dataset` 能按任一受支持布局把它们写成数据集。

`tests/conftest.py` 把它们组合成测试真正想要的东西：

| Fixture | 提供什么 |
| --- | --- |
| `workspace` | 已初始化、带默认本地 profile 的工作区 |
| `session` | 指向该工作区的已打开 `Session` |
| `dataset_root` | 10 张合成图 + 对应 COCO 标注文件 |
| `synthetic_dataset` | 数据集对象本身，供需要检查绘制内容的测试使用 |

`make_recipe(...)` 为这份数据集构造合法 recipe。它的默认门槛刻意宽松：想测试**拒绝**的用例会显式收紧门槛，因此通用流水线测试不会被它并不打算测试的启发式质量分卡住。

---

## Backend 测试替身

`RecordingRunner`（在 `tests/unit/test_engine.py`）是一个 `NodeRunner`，会记录调用，并可被指示失败、阻塞或抛出特定错误。调度器就是这样被隔离测试的：这里失败意味着**引擎**有问题，而不是某个 operator。

`tests/contract/test_backends.py` 用真实的 `ArtifactStore` 与真实 `PipelineContext`，通过协议逐个验证 backend。同样的断言适用于启发式实现、确定性替身与 HTTP 适配器。

---

## 刻意覆盖的内容

| 领域 | 代表性测试 |
| --- | --- |
| 身份与确定性 | 量化后的 object id、新增候选时种子树的稳定性、忽略 estimates 的 recipe 哈希 |
| 规范序列化 | 键排序、`-0.0`、枚举、UTC 时间戳、拒绝 NaN 与未知类型 |
| 图校验 | 重复 id、未知依赖、未知端口、环、端口类型一致性 |
| 调度 | 线性图、扇出、重试策略、默认不重试、拒绝不安全 backend、超时、取消、下游跳过、兄弟分支存活、fail-fast |
| 缓存与恢复 | 仅输入完全一致才命中缓存、取消后恢复、产物消失则缓存失效 |
| 存储 | 摘要校验、体积与尺寸护栏、路径穿越拒绝、候选生命周期、lineage、泄漏检测 |
| 质量 | 门槛顺序、硬门槛压过高总分、复核带、警告门槛、缺失指标策略 |
| 标注 | COCO 与 YOLO 往返、畸变行、缺少类别名、空 bundle、跨候选类别保持独立 |
| 流水线 | 10 图端到端、COCO 与 YOLO 导出、溯源完整性、凭据不泄漏、split 安全、多目标扇出、跳过语义 |
| 架构 | `backends/` 之外无厂商 SDK、无向上层导入、无静默吞异常、公开 API 均有文档 |

---

## 需求中特别点名的边界情况

| 情况 | 对应测试 |
| --- | --- |
| 空 mask | `test_fake_generator_refuses_an_empty_mask`、`test_segmentation_needs_a_prompt` |
| 框越界 | `test_annotation_validation_reports_out_of_bounds` |
| 对象缺失 | `test_class_filter_produces_no_targets_and_still_succeeds` |
| 替换类别错误 | `test_semantic_evaluator_flags_a_mismatch` |
| 背景被改 | `test_background_measurements_detect_a_recoloured_background`、`test_quality_rejection_is_not_a_job_failure` |
| 重复生成 | `test_duplicate_index_detects_an_exact_copy`、`test_near_duplicate_outputs_are_excluded` |
| 视频帧缺失 | `test_detector_refuses_video_input` |
| backend 超时 | `test_engine_enforces_a_node_timeout` |
| 进程重启 | `test_resume_after_an_interrupted_job`（状态从 SQLite 重新加载） |
| job 取消 | `test_engine_honours_cancellation`、`test_job_cancel_marks_the_job` |
| 标注非法 | `test_export_with_a_tiny_minimum_area_drops_samples` |
| 数据集泄漏 | `test_exporter_refuses_a_leaking_dataset`、`test_assert_no_leakage_reports_every_offending_edge` |
| 能力绑定悬挂 | `test_default_profile_has_no_dangling_binding` |

---

## 如何新增一个测试

1. 优先使用合成 fixture，而不是提交一个二进制样例文件——无法被审阅的 fixture 等于没有 fixture。
2. 断言**契约**，而不是实现。断言"某个内部 helper 被调用过"的测试，在 helper 改名时就会失败，却永远抓不到真实缺陷。
3. 新增 backend 时，补一个探测测试、一个正常路径测试，以及每个合法拒绝路径各一个测试。
4. 新增门槛时，通过用例与失败用例都要有，并断言 reason code。
5. 如果改动是架构性的（新增一层、引入新的厂商依赖），先在 `tests/unit/test_architecture.py` 里把规则写下——规则比评审意见便宜得多。
