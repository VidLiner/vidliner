# 示例

## `car-swap` —— 完整流水线，无需下载模型、无需付费 API

```bash
python examples/car-swap/demo.py [输出目录]
```

示例会：

1. 创建一个使用默认本地 profile 的工作区；
2. 渲染 10 张带标注的合成道路场景；
3. 依次运行 `plan`（免费）、`run`（生成 → 验证 → 重建标注 → 验收 → 导出）、`qa`（不重新生成的重评分），
   并读取数据集报告；
4. 打印一条溯源记录。

它打印的全部内容都来自库 API —— `Session.plan`、`Session.run`、`Session.re_evaluate` ——
正是 CLI 调用的同一批函数。如果你更想用命令行：

```bash
WS=/tmp/vidliner-cli
vidliner init $WS
python - <<'PY'
import sys; sys.path.insert(0, "examples/car-swap")
from demo import _write_coco, _write_scene_dataset
from pathlib import Path
root = Path("/tmp/vidliner-cli/data/source")
_write_scene_dataset(root / "train", count=10)
_write_coco(root, root / "train")
PY
vidliner inspect $WS/data/source --format coco-instance
vidliner plan    examples/car-swap/car-swap.yaml --workspace $WS
vidliner run     examples/car-swap/car-swap.yaml --workspace $WS
vidliner qa      latest --workspace $WS --show 5
vidliner export  latest --workspace $WS --format yolo-segmentation --path ./data/yolo
vidliner report  review latest --workspace $WS
```

### 该看什么

| 产物 | 为什么重要 |
| --- | --- |
| `plan` 的节点数与绑定 | 图在任何生成调用发生之前就被构建并估算 |
| `run` 的计数器 | accepted / rejected / review 与 failed 是分开计数的 |
| `qa` 的输出 | 阈值可以套用到已存证据上，无需重新生成 |
| `dataset-report.json` | 替换分布均匀覆盖 `sedan`、`suv`、`pickup` |
| `manifest.json` | 计数、类别、种子树与逐样本评分 |
| `provenance/*.json` | 源摘要、recipe 哈希、种子、backend、评分、决策 |
| `review/index.html` | 落在复核带内的候选所对应的人工复核界面 |

### 验证阶段对示例的影响

示例现在跑的是闭合的回路，因此它的通过率是一次真实测量，而不是形式上的通过。`verify.redetect` 在每张生成图上找到对象并重新分割；替换结果比它替换掉的区域明显更小、或合成外观与目标类别不符的候选，会以 `GEOMETRY_SCALE_CHANGE` 或 `SEMANTIC_MISMATCH` 被拒绝，而不是靠着"和它自己拿到的 mask 比对"蒙混过关。

内置生成器只是按种子涂一块纯色，因此少数候选确实会落在检测器的显著性范围之外，以 `OBJECT_NOT_FOUND` 被拒绝。这是流水线在正常工作而不是故障：你看到的通过率，就是这套内置 demo 栈的真实产出能力。

### 为什么会有候选被拒绝

内置生成器产出的是确定性合成图像，而不是照片级编辑；内置评估器也是启发式的。因此真正有意思的行为不是"它能通过"，而是它**会拒绝**那些测得的语义、背景保持或几何不成立的候选。

把 `car-swap.yaml` 里的 `quality.hard_gates` 调严，再跑一次 `vidliner qa`：同一批候选会基于已存证据被重新判定，reason code 会解释每一个。

这正是整个项目的立足点：流水线宁可自己过不了门槛，也不产出一个看起来完整、实际有问题的数据集。

### 换成真实模型

recipe 一个字都不用改，只在工作区的 `runtime.yaml` 里把三项能力指向真实 backend：

```yaml
backends:
  my_detector: { use: my_project.detector:DetectorBackend, options: { model_path: /models/yolo.onnx } }
  my_editor:   { use: my_project.editor:EditorBackend, credentials: { api_key: { source: env, name: EDIT_KEY } } }

bindings:
  vision.object_detection.v1: my_detector
  generation.object_replacement.v1: my_editor
```

契约见 `docs/zh/backends.md`，profile 说明见 `docs/zh/runtime.md`。
