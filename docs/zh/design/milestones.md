# VidLiner —— MVP 里程碑

[English](../../design/milestones.md)

每个里程碑都是一个可工作的增量，并配有验证命令。里程碑在检查项通过时完成，而不是在文件存在时完成。

## M0 —— 骨架与设计

* 交付物：设计文档、仓库布局、工具链（ruff、ty、pytest）。
* 验证：`pytest -q` 可运行，`ruff check .` 干净，`vidliner --help` 打印命令列表。

## M1 —— 领域与核心基元

* `vidliner.domain`：媒体、对象、track、替换、标注、质量、job 模型。
* `vidliner.core`：规范 JSON、摘要/身份派生、种子树、错误分类、结果、图、状态枚举。
* 验证：`pytest tests/unit/test_canonical.py tests/unit/test_domain.py tests/unit/test_engine.py`。

## M2 —— 存储

* 内容寻址产物存储（写入/校验/查找、MIME 与尺寸校验、体积上限）。
* `storage-schema.md` 中的 SQLite 状态库与迁移。
* 工作区布局创建。
* 验证：`pytest tests/unit/test_storage.py`。

## M3 —— 能力与 backend

* 每项能力的协议定义。
* Backend 注册表：惰性导入、绑定解析、基于探测的预检、容量预算。
* fake backend（确定性、无外部调用）与内置本地 backend：启发式检测/分割/场景/规划/评估、
  PIL 合成器与协调器、phash。
* 一个通用 HTTP 替换 backend。
* 验证：`pytest tests/contract`；`vidliner backend check`。

## M4 —— Operator 与质量

* MVP 图中的每个 operator，各自带类型化端口与配置 schema。
* 指标评估器、验收策略引擎、reason code。
* 验证：`pytest tests/unit/test_quality.py tests/contract`。

## M5 —— 标注

* 内部 IR 转换器：COCO detection/instance、YOLO detection/segmentation。
* 往返测试，含边界情况（空、越界、crowd、缺失 image 条目、畸变行）。
* 验证：`pytest tests/unit/test_annotations.py`。

## M6 —— 引擎与 job 编排

* 图校验、拓扑调度、有界并行、重试、超时、取消、恢复、节点缓存、结构化 JSON 日志、job 汇总。
* 验证：`pytest tests/pipeline`，其中包含恢复、取消、缓存与重启测试。

## M7 —— CLI 与端到端流程

* `init`、`inspect`、`recipe validate|schema|init|show`、`plan`、`run`、`jobs`、
  `job show|resume|cancel`、`qa`、`export`、`backend list|check`、`report review|summary|events`。
* Dry run 打印估算且不调用生成 backend。
* 静态 HTML 复核报告。
* 验证：`pytest tests/pipeline/test_cli.py`，并手动走一遍 `docs/zh/getting-started.md`。

## M8 —— 数据集完整性

* 感知 lineage 的 split 校验、去重（phash + embedding 钩子）、分布报告。
* 验证：`pytest tests/pipeline/test_export.py`。

## M10 —— 闭合验证回路（MVP 之后，路线图第 1 项）

* 在精修与评估之间加入 `verify.redetect`：在生成图上重新检测被替换的对象并重新分割，下游一切描述的都是这个对象。
* 导出的标注由重建 mask 生成；交给生成器的 mask 只用于界定被授权改变的区域。
* "没找到对象"是一种结果（`OBJECT_NOT_FOUND`），而不是节点失败。
* 验证：`pytest tests/unit/test_verification.py tests/pipeline/test_verification_loop.py`，后者用一个"删掉对象"的 backend 做端到端验证。

## M9 —— 文档与 Definition of Done

* README、ARCHITECTURE、CONTRIBUTING 与 `docs/`（中英文），以及可运行的示例。
* 用 fake backend 在 10 张合成图的 10 图数据集上验证下方清单。

## 验证记录

MVP 已完成。撰写本文时：

| 检查项 | 命令 | 结果 |
| --- | --- | --- |
| 格式 | `ruff format --check .` | 137 个文件已格式化 |
| 静态检查 | `ruff check .` | 干净 |
| 类型检查 | `ty check vidliner` | 干净 |
| 安装 | `pip install -e .` 后 `vidliner --help` | 在干净环境中可用 |
| 测试 | `pytest -q` | 367 个测试通过 |
| 示例 | `python examples/car-swap/demo.py` | 端到端运行，无需联网 |

规模：包代码约 23.4k 行、105 个模块；测试约 4.9k 行；中文文档约 3.0k 行，与英文文档一一对应。

## Definition of Done（产品需求 §38）

| 需求 | 在何处被证明 |
| --- | --- |
| 读取 10 图数据集 | `tests/pipeline/test_end_to_end.py` |
| 检测目标对象 | M4 + 端到端 |
| 建立 mask | M4 + 端到端 |
| 针对对象生成替换候选 | M4 + 端到端 |
| 保存溯源 | `tests/pipeline/test_end_to_end.py::test_provenance_records_everything_the_requirement_lists` |
| 自动 QA 并拒绝低质量 | `tests/unit/test_quality.py`、端到端 |
| 重新计算 bbox/mask | `test_annotations_are_regenerated_not_copied` |
| 导出 COCO | `test_run_exports_a_coco_dataset` |
| 导出 YOLO | `test_run_exports_yolo` |
| 数据集报告 | `test_export_report_records_the_distribution` |
| 中断后恢复 job | `tests/pipeline/test_resume_and_cache.py::test_resume_after_an_interrupted_job` |
| 重复执行利用 cache | `test_rerunning_the_same_job_reuses_the_cache` |
| 全流程零付费 API | 所有端到端测试均使用 fake/本地 backend |
| pytest / 类型检查 / 静态检查通过 | M9 验证记录 |
| Core 与厂商 SDK 解耦 | `tests/unit/test_architecture.py` |
| 每个候选保持独立类别 | `test_each_candidate_keeps_its_own_category` |

## MVP 明确不在范围内

视频替换、跟踪执行、分布式 worker、队列、云对象存储、Web UI、多用户认证与插件市场。
领域模型与引擎已经带有扩展点（`ObjectTrack`、`FrameRef`、`temporal_consistency`、
按样本扇出），Phase 2/3 计划记录在英文版 `docs/design/milestones.md` 中。
