# 参与贡献

感谢你的帮助。本文说明如何在 VidLiner 上工作，以及更重要的——那些让项目保持可信的规则。

---

## 环境准备

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

每次提交前：

```bash
ruff format --check .    # 或 ruff format .
ruff check .
ty check vidliner
pytest -q
```

四项都必须通过。它们很便宜，而破坏其中一项的改动，就是会破坏别人一整个下午的改动。

若 `pip install` 的构建隔离阶段因网络失败：

```bash
pip install -e . --no-build-isolation
```

---

## 真正重要的规则

### 1. 未实现的行为必须响亮地失败

```python
raise NotImplementedError("video replacement is a phase-2 capability")
```

不要占位返回，不要 `pass` 函数体，不要"暂时先这样"的静默成功。一个悄悄什么都不做的组件，产出的数据集会"看起来完整、其实不是"——这是本项目中最危险的 bug。

### 2. 不得静默吞掉异常

```python
# 不行
try:
    ...
except Exception:
    pass

# 可以：当该情形确实在预期内，并且你说明了原因
with contextlib.suppress(TimeoutError):
    await asyncio.wait_for(event.wait(), timeout=delay)
```

`tests/unit/test_architecture.py` 会抓语法层面的情况；至于"捕获后继续并伪造成功"，那由你负责。

### 3. 核心保持无厂商依赖

`vidliner/backends/` 之外不得导入任何模型运行时或厂商 SDK，测试会强制。需要新能力时，在 `capabilities/` 定义协议、在 `backends/` 实现。

### 4. 分层方向

```
cli → pipeline → operators → capabilities → domain/core → (storage, control, runtime) → backends
```

下层不得导入上层。测试对 `domain`、`core`、`capabilities`、`quality`、`annotations` 强制这条方向。

`core` 只允许引用 `domain` 的 `enums` 与 `shapes` 两个基元模块。需要 operator 目录时，把那依赖**注入**进去（见 `OperationGraph.validate_against_operators` 的 `describe` 参数），而不是 import。

### 5. 确定性是一项功能

* 绝不调用模块级 `random` 或全局 `numpy.random` 状态——用 context 给你的种子；
* 身份从内容派生，而不是从插入顺序或时钟；
* 新增扇出时用显式身份（`object:<id>`、`candidate:<key>`），绝不用列表位置，否则你会静默让无关工作的缓存失效。

### 6. 绝不出现凭据

```python
credentials:
  api_key: { source: env, name: EDIT_KEY }   # 引用
```

绝不在 recipe、仓库内的 profile、manifest、日志行或测试中写明文。`runtime/secrets.py` 是唯一把凭据变为字符串的地方，并且它会把已知的密钥从每个事件中脱敏。

### 7. 类型标注

每个公开函数都要有注解，以及说明**为什么**（而不是"做什么"）的 docstring。公开 API 的 docstring 要求由测试强制。

### 8. Docstring 解释决策

```python
def quantise_box(...):
    """...在哈希之前量化，使一像素抖动不会创造出新身份..."""
```

说明代码为什么是这样。代码本身已经说明了它在做什么。

### 9. 证据不能丢

被拒绝的候选必须保留其产物、指标行与复核报告条目——否则就再也无法在不重新生成的情况下调整阈值。记录失败时也不要丢掉原因：`QualityReject` 是正常结果，静默丢弃不是。

---

## Clean-room 政策

VidLiner 是独立实现。不要从其他项目复制源代码、标识符、文件内容、DSL、命令体系、注释、文档、测试、manifest、资源或许可文本。通用架构思想——依赖图、端口与适配器、内容寻址存储——属于业界通行做法，可以借鉴；但它们的**具体表达**不可以。

如果某项功能只能靠复制获得，就从需求重新推导。若暂时做不到，就 `raise NotImplementedError` 并在 `docs/zh/design/milestones.md` 中记录。

完整声明见 `docs/zh/design/decisions.md`（ADR-001）。

---

## 如何新增东西

### 一个 backend

1. 在 `vidliner/backends/`（或你自己的包）中实现协议。
2. 声明 `backend_id`、`backend_version`、`capabilities`、`determinism`、`safe_to_retry`、
   `blocking`、`declared_capabilities`。
3. 在 runtime profile 中绑定它。
4. 补契约测试：探测、正常路径、每条合法拒绝路径。
5. 如果足够通用，在 `docs/zh/backends.md` 与 `docs/backends.md` 中记录。

### 一个 operator

1. 在 `vidliner/operators/` 新建模块，包含 `OperatorSpec`、输入/配置 Pydantic 模型与
   `register_all(registry)`。
2. 接进 `build_default_registry`。
3. 若引入新阶段，加入 `StageName` 与 `_STAGE_ORDER`（保证确定性排序）。
4. 若属于默认图，加进装配器。
5. 直接测它；若改变了图形状，再加一个流水线测试。

### 一个质量指标

1. 把名称加入 `MetricName`，在 evaluate operator 中实现测量。
2. 在 `quality/gates.py` 加默认 reason code，并在 `ReasonCode` 中登记该码。
3. 在 `docs/zh/quality.md` 与 `docs/quality.md` 的表格中补充。
4. 通过用例与失败用例都要有。

### 一条架构规则

加进 `tests/unit/test_architecture.py`。构建能强制的规则会活下来；只活在文档里的规则不会。

---

## 提交与 PR

* 每个提交只做一件逻辑改动，提交信息说明改了什么、为什么；
* 每处行为改动都有测试，包括失败路径；
* 同步更新因此过时的文档——`README.md`、`ARCHITECTURE.md`、`docs/*.md` 及各中文版本；
* 如果这个改动是"将来可能被质疑的决策"，把它记进 `docs/zh/design/decisions.md`，而不是留在注释里。

### 评审清单

- [ ] 标注是否仍然正确？（优先级 1）
- [ ] 数据集是否仍然内部一致？（优先级 2）
- [ ] 结果能否从记录的种子与输入复现？（优先级 3）
- [ ] 溯源是否仍然完整，且不含凭据？（优先级 4）
- [ ] 质量是否仍然拒绝该拒绝的？（优先级 5）
- [ ] 核心是否仍然与厂商解耦？（优先级 6）
- [ ] 中英文文档是否同时更新？（避免两种语言的文档漂移）

顺序即优先级。性能与 UI 刻意排在最后：100 个标签错误的生成样本，比 10 个高质量样本更糟。

---

## 报告缺陷

请包含：

* 你运行的命令与 recipe（对敏感路径或取值做脱敏）；
* `vidliner job show <id> --json` 与 `vidliner report events <id> --json`；
* 绑定了哪个 backend（来自 `vidliner backend check`）；
* 你**期望数据集**发生什么。

一份说明"哪些样本本该被拒绝却没有"的缺陷报告，价值十倍于"某次运行失败了"。
