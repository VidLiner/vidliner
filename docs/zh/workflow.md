# 工作流画布交换

VidLiner 现在可以把编译后的 recipe 导出为画布文档，校验接线，并生成可离线打开的交互式
HTML 编辑器。它让任务在生成前就能被查看，并支持编辑节点布局和配置草稿。

画布复用 `OperationGraph`、算子注册表及其执行契约。文档里的 backend 名称只是规划提示；
执行 recipe 时仍必须重新解析能力绑定并检查生产导出与验收策略。

## 使用方式

在已有 `data/source` 数据集、runtime profile 和有效 recipe 的工作区中运行：

```sh
vidliner workflow export recipe.yaml --workspace . --limit 1 --output workflow.json
vidliner workflow validate workflow.json
vidliner workflow preview workflow.json --output workflow.html
```

用浏览器打开 `workflow.html`：按算子、节点 ID 或 lineage 搜索，拖动背景平移，使用滚轮或按钮
缩放，选择节点查看配置、端口、连接、能力需求、backend 提示、重试和并行限制。
页面无需联网，也不会调用模型。点击 `Edit layout` 后可拖动节点，连接线同步更新；选中节点
后可编辑配置 JSON 对象。点击 `Save JSON` 下载草稿，再用 `workflow validate` 校验。
浏览器只检查 JSON 基本形式，不替代算子 Schema 校验。

### 本地化

离线预览与 hosted 画布共用内置的 `en-US`、`zh-CN`、`ja-JP`、`ko-KR` 和 `es-ES` 资源。
使用工具栏中的“语言”选择器即可切换；选择会写入浏览器 local storage，并在下次打开时恢复。
文档中的算子 ID、JSON 阶段名和 API payload 在不同语言之间保持稳定。语言资源直接嵌入 HTML，
因此离线页面也能切换语言。

hosted 画布另提供 GitHub 仓库和 Star 历史记录入口。Star 数来自 GitHub 公开接口，设置四秒
超时与一小时会话缓存；网络不可用或触发限流时只保留仓库链接，不显示虚构计数。请求不携带
画布 token 或供应商凭据，离线预览不会请求 GitHub。

导出默认只规划一个样本，`--limit 0` 包含全部可增强样本。已有输出文件需要 `--force` 才能
覆盖。导出会发现数据集并解析声明的能力绑定，不实例化 backend；未绑定的能力仍会保留，
方便查看尚不具备执行条件的计划。

## 文档契约

格式版本为 `vidliner.workflow/v1`。未知字段及不支持的版本会被拒绝。获取 Schema 和算子面板：

```sh
vidliner workflow schema
vidliner workflow catalog
```

| 字段 | 含义 |
| --- | --- |
| `nodes` | 算子实例，保留版本、阶段、配置、输出及执行策略。 |
| `nodes[].position` | 画布坐标，移动节点不会改变执行语义。 |
| `nodes[].inputs` | 样本角色、任务级 artifact 角色或 JSON 常量，保留显式 `null`。 |
| `edges` | 生产者输出端口到消费者输入端口的连接。 |
| `nodes[].selects` | 对象分支使用的显式元组选择索引。 |
| `job_lineage`、`nodes[].lineage` | 原有身份路径，保留 seed 派生上下文。 |
| `recipe_hash` | 来源 recipe 的身份；不能证明编辑后的文档仍与 recipe 一致。 |
| `capability_bindings`、`unmet_capabilities` | 规划提示；凭据及 runtime profile 不进入文档。 |

校验覆盖节点 ID 重复、未知节点/输出端口、输入多重绑定、循环、缺失/未声明输入、端口类型、
算子版本和配置 Schema。输出、阶段、确定性和能力声明必须与注册算子一致。
文档还会拒绝凭据字段及非有限坐标。

Python 接口：

```python
from vidliner.pipeline.workflow import graph_from_workflow, workflow_from_graph

document = workflow_from_graph(plan.graph, name=plan.recipe.name)
restored_graph = graph_from_workflow(document)
```

`load_workflow(path)` 读取 JSON 并对照内置注册表校验。自定义算子的嵌入方可以向
`graph_from_workflow` 或 `operator_catalogue` 显式传入 `registry=`。
交换格式中的常量和配置限定为 JSON 值，不支持任意 Python 对象；节点顺序按拓扑序标准化。

## 当前边界

本版提供画布交换、类型化算子面板与离线计划编辑。执行入口仍是 `vidliner run recipe.yaml`。
画布文档是图快照，不能替代 recipe 的数据集及验收策略，也没有独立执行命令。

`video probe/generate/render` 已支持 FFmpeg 本地变体渲染，尚未接入画布执行。
`workflow serve` 已接通图结构编辑手势、执行按钮、任务状态及视频预览；媒体资产浏览和公网多用户宿主仍待实现。
图结构修改事务与画布执行宿主接口已预留，详见[视频与画布接口说明](video-integrations.md)。
[Toonflow 借鉴评估](../design/toonflow-assessment.md) 说明了选型依据及后续方向。
