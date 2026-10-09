# 视频模型与画布接入

项目提供库、CLI 与本地网页应用。完整契约及示例见
[英文接入文档](../video-integrations.md)，模型配置见
[runtime-ai.yaml](../../examples/video/runtime-ai.yaml)。

AI 视频复用现有 `RuntimeProfile`、`BackendRegistry`、`CapabilityBroker`、凭据解析和并发限流。
新增 `generation.video_generation.v1` 能力，以及 `video.submit`、`video.status`、`video.cancel`
节点。请求与任务句柄有独立 JSON Schema；提交、查询、取消分开执行，不在提交超时后自动重试。

已有适配器支持 Runway 原生文生/图生视频、fal 队列协议，以及 Bifrost 的 OpenAI 兼容视频网关；示例包含 Runway Gen-4.5、Kling、Wan、Veo。

Bifrost 是 API 网关，不是本地模型运行时。可复制 `examples/video/runtime-bifrost.yaml`，设置 `BIFROST_API_KEY`，并把
`model_id` 写成网关可解析的模型标识。VidLiner 只调用 `/v1/videos` 及其任务查询/删除接口，网关负责供应商路由、回退和供应商密钥。
时长、分辨率、音频等模型参数保留供应商的数据类型，通过 `parameters` 传入。新增原生服务或本地模型
可以按 `VideoGenerationBackend` 协议实现适配器，通过 runtime 的 `use` 导入，无需另建调度器。
这些路径已使用模拟 HTTP 响应测试，尚未进行真实模型付费生成验证。

图结构编辑通过 `WorkflowEditRequest` 提交原子批次，可增删节点、连线/断线、改配置、移动节点、
绑定/解绑输入。`expected_digest` 用于识别过期草稿；最终状态必须通过已有算子、端口及无环验证。
宿主需用同一摘要实现存储层的原子比较与替换。CLI 可通过 `workflow patch` 使用相同逻辑。

画布执行通过 `build_workflow_engine()` 复用 `OperationEngine`，提供执行、状态、取消和事件回调。
它在执行前检查草稿摘要、真实 runtime 绑定、backend 可用性与外部输入，且拒绝训练导出节点。
宿主负责用户认证、费用授权、持久任务状态、任务去重及远程状态轮询；`allow_external` 必须由可信宿主设置。
`workflow serve` 已提供本地 HTTP 服务、网页执行按钮、算子面板、拖拽/键盘连线和断线操作。
新增未接线的必要输入保存为 artifact 占位绑定，执行时必须补齐连接；本地宿主暂不注入数据集上下文或外部资产。

生成结果的 `verified` 固定为 `false`，标签变换默认是 `synthesized`。视频链接不代表已入库的产物
或已验收训练样本；训练导出继续走既有验收及生产 backend 守卫。

```sh
vidliner video request-schema
vidliner video task-schema
vidliner workflow catalog
vidliner workflow edit-schema
vidliner workflow execution-schema
vidliner workflow digest workflow.json
vidliner workflow patch workflow.json edits.json --output edited.json
```

启动无需模型凭据的规划画布：

```sh
vidliner workflow serve examples/video/workflow-canvas.json --workspace ./canvas-workspace
```

打开 `http://127.0.0.1:8767`，点击 Execute 可执行规划节点。新增节点后，将输出圆点拖到输入圆点；
也可以先聚焦输出端口按 Enter，再聚焦输入端口按 Enter。点选连线后可断线，选中节点后可删除。
配置与布局自动保存，Save JSON 导出副本。服务使用 SQLite 原子保存草稿及任务记录，不覆盖原始 JSON。
生成节点成功提交后由宿主每 5 秒查询一次，最多 10 分钟；之后可以手动刷新或取消已保存的任务。
视频结果直接在页面播放供应商链接；链接可能过期，视频仍需训练验收。

同一任务 ID 的重复请求返回已有任务；改变草稿、runtime 或 seed 后不能复用旧 ID。
服务重启不会重发生成请求。Cancel job 停止本地调度，并请求取消已保存的远程任务；确认收到取消请求
不代表远程任务已终止。提交超时或在拿到句柄前崩溃可能留下未知任务，需要人工核对供应商记录。

服务只绑定回环地址，API 校验会话令牌和同源请求，默认禁用外部生成。一次只运行一个宿主操作该草稿；
它是本机单进程应用，不提供公网多用户认证。保留 workspace/cache 中的私有数据库，以保留任务证据和去重状态。

Runway 真实生成：先安装 `vidliner[http]`，为启动进程配置 `RUNWAYML_API_SECRET`，或在私有 runtime
副本里把 `api_key` 改为 `{source: file, name: /绝对路径/runway-key}`。不要把密钥提交到仓库。
下面的示例请求会使用账户额度生成 5 秒 Gen-4.5 视频；仅打开页面或加载配置不会提交任务。

```sh
vidliner workflow serve examples/video/workflow-runway.json \
  --workspace ./runway-workspace --runtime examples/video/runtime-runway.yaml --allow-external

vidliner video verify examples/video/request-runway.json \
  --workspace ./runway-verification --runtime examples/video/runtime-runway.yaml \
  --job-id runway-smoke-001 --timeout 600
```

`video verify` 同样复用画布服务：保存请求意图、句柄与节点结果，有界轮询，重复使用同一 `--job-id`
只刷新已有任务。退出码 0 表示供应商生成成功，2 表示校验失败、生成失败、中断或待完成。
默认只打印证据数据库位置，`--json` 会包含私有任务句柄及视频链接。自动测试使用免费模拟响应；
真实验证只有在收到实际终态结果后才能报告成功，且不等于画质或训练数据验收通过。
