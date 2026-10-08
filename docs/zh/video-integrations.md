# 视频模型与画布接入

本次提供库和 CLI 接口，便于后续网页或桌面宿主接入。完整契约及示例见
[英文接入文档](../video-integrations.md)，模型配置见
[runtime-ai.yaml](../../examples/video/runtime-ai.yaml)。

AI 视频复用现有 `RuntimeProfile`、`BackendRegistry`、`CapabilityBroker`、凭据解析和并发限流。
新增 `generation.video_generation.v1` 能力，以及 `video.submit`、`video.status`、`video.cancel`
节点。请求与任务句柄有独立 JSON Schema；提交、查询、取消分开执行，不在提交超时后自动重试。

已有适配器支持 Runway 原生文生/图生视频和 fal 队列协议；示例包含 Runway Gen-4.5、Kling、Wan、Veo。
时长、分辨率、音频等模型参数保留供应商的数据类型，通过 `parameters` 传入。新增原生服务或本地模型
可以按 `VideoGenerationBackend` 协议实现适配器，通过 runtime 的 `use` 导入，无需另建调度器。
这些路径已使用模拟 HTTP 响应测试，尚未进行真实模型付费生成验证。

图结构编辑通过 `WorkflowEditRequest` 提交原子批次，可增删节点、连线/断线、改配置、移动节点、
绑定/解绑输入。`expected_digest` 用于识别过期草稿；最终状态必须通过已有算子、端口及无环验证。
宿主需用同一摘要实现存储层的原子比较与替换。CLI 可通过 `workflow patch` 使用相同逻辑。

画布执行通过 `build_workflow_engine()` 复用 `OperationEngine`，提供执行、状态、取消和事件回调。
它在执行前检查草稿摘要、真实 runtime 绑定、backend 可用性与外部输入，且拒绝训练导出节点。
宿主负责用户认证、费用授权、持久任务状态、任务去重及远程状态轮询；`allow_external` 必须由可信宿主设置。
本次未增加 HTTP 服务、网页执行按钮或图结构编辑手势，前端可继续消费上述接口和算子目录。

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
