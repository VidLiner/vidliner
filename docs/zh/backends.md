# 编写 Backend

**能力（capability）**是流水线需要的一项本领；**backend** 是它的一种具体实现。流水线只引用能力名，因此替换模型只是一次配置改动。

---

## 词汇表

| 能力 | 契约方法 | 使用阶段 |
| --- | --- | --- |
| `vision.object_detection.v1` | `detect` | detect 阶段**与**生成后验证 |
| `vision.instance_segmentation.v1` | `segment` | segment 阶段**与**生成后验证 |
| `vision.scene_analysis.v1` | `analyse` | scene |
| `vision.object_tracking.v1` | `track` | 预留（视频） |
| `vision.depth_estimation.v1` | `estimate_depth` | 预留（视频） |
| `planning.replacement.v1` | `plan` | plan |
| `generation.object_replacement.v1` | `replace` | generate |
| `generation.image_compositing.v1` | `composite` | refine |
| `generation.image_harmonization.v1` | `harmonize` | refine |
| `generation.mask_refinement.v1` | `refine_mask` | refine |
| `generation.video_replacement.v1` | — | 预留（视频） |
| `quality.semantic_match.v1` | `assess_semantics` | evaluate |
| `quality.background_preservation.v1` | — | **内置**：由流水线自行测量，永不绑定 |
| `quality.artifact_detection.v1` | `evaluate` | evaluate |
| `quality.vision_evaluation.v1` | `evaluate` | evaluate |
| `quality.embedding.v1` | `embed` | 去重 |

`vidliner backend list --capabilities` 会打印完整清单及每项的一句话说明。

绝大多数能力由 backend 提供，只有一项例外：`quality.background_preservation.v1` 是流水线在 `vidliner/quality/metrics.py` 中自己做的确定性像素比较（"非目标像素是否变化"是像素比对，不是模型判断）。它留在词汇表里是因为 recipe 的验收门槛会引用它，但 profile **不得绑定**它，`vidliner backend check` 会把它报告为 `builtin`。

---

## 最小契约

```python
class MyDetector:
    backend_id = "my_detector"            # 稳定标识，出现在 manifest 与缓存键中
    backend_version = "1.0.0"             # 适配器版本，不是模型版本
    capabilities = (CAP_OBJECT_DETECTION,)
    determinism = Determinism.DETERMINISTIC
    safe_to_retry = True
    blocking = False                      # 本地 CPU/GPU 工作设为 True

    def __init__(self, spec, options, credentials=None):
        self.options = options

    async def probe(self) -> BackendProbe:
        return BackendProbe(
            backend_id=self.backend_id,
            version=self.backend_version,
            health="ready",
            capabilities=self.capabilities,
            determinism=self.determinism,
            message="loaded yolo11n.onnx",
        )

    async def detect(self, request: DetectionRequest, context: PipelineContext) -> DetectionResult:
        image = context.require_io().image(request.image)     # 得到 Pillow 图像
        ...                                                   # 你的推理
        return DetectionResult(instances=instances, backend_id=self.backend_id)
```

一个 backend 可以提供多项能力：全部写进 `capabilities`，并实现对应方法。

---

## 两种形态

**原生异步**（HTTP 客户端、远程服务）。写 `async def` 方法并在内部 `await`，运行时直接 await 它们。

**阻塞型**（本地模型、解码器、子进程）。设置 `blocking = True`，并把同步实现按协议方法名挂在 `self._sync` 上：

```python
class LocalDetector:
    blocking = True

    def __init__(self, spec, options, credentials=None):
        self._sync = _Sync(self)

    async def detect(self, request, context):
        return self._sync.detect(request, context)      # 运行时会在线程中执行它

class _Sync:
    def __init__(self, backend):
        self._backend = backend

    def detect(self, request, context):
        ...
```

运行时会在 worker 线程中调用 `self._sync.detect`。切勿在运行时直接 await 的 `async def` 里做阻塞操作。

---

## Backend 能访问什么

`PipelineContext` 携带：

| 字段 | 用途 |
| --- | --- |
| `job_id`、`node_id` | 日志与诊断 |
| `seed` | 该节点派生出的种子——**请用它，而不是任何全局随机数** |
| `device` | profile 给出的设备提示 |
| `config` | 该节点已校验的配置 |
| `metadata` | profile 选项与估算 |
| `io` | `ArtifactIO`：`load`、`load_json`、`image`、`load_mask`、`save_image`、`save_mask`、`save_json` |

Backend 无法解析工作区路径、无法打开状态数据库、无法读取 recipe，也无法越过自己的客户端访问网络。这是刻意的：它让 backend 可替换、可测试。

**确定性是硬要求。** 同一节点、同一种子的两次运行必须产出相同结果，否则缓存、恢复与溯源全都失去意义。

---

## 请求与结果

请求携带的是**产物引用**，不是像素，因此可缓存、可序列化。唯一例外是 `SceneAnalysisRequest`：它携带已解码的 mask 与亮度，因为构造它的 operator 本来就必须解码一次。

结果对象携带"是谁产出的"：

```python
DetectionResult(instances=..., backend_id=..., model_id=..., duration_ms=..., raw={...})
GenerationOutcome(image=..., backend_id=..., model_id=..., seed=..., cost_estimate=..., raw_metadata={...})
```

`raw_metadata` 会被记录以供审计，但它**永远不会**成为质量门槛的输入——生成模型对自身输出的评价不是证据。

---

## 错误

请抛出 `vidliner.core.errors` 中的结构化错误：

| 错误 | 含义 |
| --- | --- |
| `BackendFailure(..., safe_to_retry=...)` | 不可达、超时、限流，或返回了不可用的东西 |
| `OperatorFailure` | 请求本身无法满足（例如该 backend 不支持文本 prompt） |
| `ValidationFailure` | 输入违反契约 |

只有在重试同一请求是幂等时，才设 `safe_to_retry=True`。生成类调用通常不是。

---

## 内置 backend

它们的存在是为了让新克隆的仓库立刻能跑，并且对自身定位很诚实：

| Backend | 能力 | 说明 |
| --- | --- | --- |
| `heuristic_detector` | detection | 基于显著性；只报告它认识**且**数据集标注过的类别 |
| `heuristic_segmenter` | segmentation | 由 box / point prompt 生长 mask；拒绝纯文本 prompt |
| `heuristic_scene` | scene analysis | 测量朝向、光照、接地、阴影、遮挡 |
| `rule_based_planner` | planning | 确定性；**从不**调用任何模型 |
| `fake_replacement` | replacement | 确定性合成图像，标记 `synthetic: true` |
| `local_metric_evaluator` | semantic / artifact / embedding | 颜色外观启发式 + 伪影测量 |
| `local_refiner` | mask / blend / harmonise | 确定性 numpy 运算 |
| `perceptual_hash` | embedding | 64 位 pHash，以向量形式暴露 |
| `http_replacement` | replacement | 通用 JSON + base64 适配器，可对接任意编辑服务 |

每个内置探测在"只是基线而非训练模型"时会报告 `degraded`，因此 `vidliner backend check` 不会夸大就绪程度。

---

## HTTP 适配器

`HttpReplacementBackend` 可对接任何"接受含 base64 图片与 prompt 的 JSON 请求、返回含 base64 图片的 JSON 响应"的服务：

```yaml
http_replacement:
  use: vidliner.backends.http_replacement:HttpReplacementBackend
  options:
    endpoint: http://127.0.0.1:8080/v1/edit
    request_image_field: image_base64
    request_mask_field: mask_base64
    request_prompt_field: prompt
    request_negative_field: negative_prompt
    response_image_field: image_base64
    response_image_encoding: base64
    allowed_media_types: [image/png, image/jpeg, image/webp]
    max_response_mb: 32
    request_timeout_s: 120
    extra_fields: { model: my-editor-v2 }
  credentials:
    api_key: { source: env, name: VIDLINER_EDIT_API_KEY }
```

响应被视为不可信输入：限制体积、声明的 MIME 必须与解码后的字节一致且在白名单内、尺寸经过校验，并且图像最终通过产物存储写入（存储会再校验一次）。

---

## 契约测试

`tests/contract/test_backends.py` 用合成 fixture 通过协议逐个验证内置 backend。新 backend 应当获得同样待遇：一个探测测试、一个正常路径测试，以及每个合法拒绝路径各一个测试。

任何 backend 都值得断言两条性质：

* **确定性** —— 同一请求两次得到相同的产物摘要（针对 `seeded` 与 `deterministic`）；
* **封闭性** —— 生成类 backend 只改变它被要求改变的区域。
