## DAT-former
手写公式识别（Handwritten formula recognition）实现与训练工程。

本仓库实现了基于 Transformer 的公式识别训练/评估流水线，并集成了常用的数据预处理与评估工具（参考 CROHME / LgEval）。

## 目录概览

以下为仓库中重要文件/目录（非穷尽）：

- `config.yaml` — 默认配置文件（训练/数据/模型等参数）。
- `train.py` — 训练入口脚本。
- `requirements.txt` — Python 依赖。
- `dat_formmer/` — 模型、数据模块、工具代码。
	- `datamodule/` — 数据集加载与变换代码。
	- `model/` — encoder/decoder/transformer 等模型实现。
	- `utils/` — beam search、生成工具等。
- `convert2symLG/` — 数据格式转换与工具脚本（mml/lg/symlg 等）。
- `lgeval/` — LgEval 评估工具集（用于与标准评测脚本配合计算指标）。
- `lightning_logs/` — 训练时的日志与 checkpoint（由 PyTorch Lightning 产生）。

## 快速开始（快速复现）

下面给出一个最小快速开始流程，假设你在 Windows 下使用 PowerShell：

1) 创建并激活虚拟环境（可选但推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装依赖

```powershell
pip install -r requirements.txt
```

3) 准备数据（示例）

仓库附带 `data.zip`（或请自行放置 CROHME 数据）。默认使用 CROHME 2014 数据集。如果你使用的是 2016/2019 或自定义数据集，请在 `config.yaml` 中修改路径与数据选项。

（假设解压后数据路径为 `data/crohme2014/`，可在 `config.yaml` 中将 dataset.path 指向该路径。）

4) 启动训练

```powershell
python train.py --config config.yaml
```

训练过程会在 `lightning_logs/` 下保存日志与 checkpoint（取决于 `config.yaml` 中的 logger/checkpoint 配置）。

5) 评估

仓库提供了评估脚本 `eval_all.sh`（UNIX shell）和 `lgeval/` 工具。Windows 用户可使用 WSL 或在 PowerShell 下运行相应的脚本。

```powershell
sh eval_all.sh
```

或使用 `lgeval` 中的工具对输出进行更细粒度的评估。

## 数据说明

- 支持的数据集：CROHME（2014/2016/2019）。仓库默认配置使用 2014 数据集。
- 数据格式：仓库内提供若干转换工具（`convert2symLG/`），用于将不同来源的标注（MML、LG、SYMLG 等）互转，配合 `lgeval/` 进行评测。

注意：如果你的数据不是严格的 CROHME 格式，需要通过 `convert2symLG` 中的脚本进行转换，或者修改 `dat_formmer/datamodule/dataset.py` 中的数据读取逻辑以匹配你的标注结构。

假设/约定：
- 假设训练脚本期望的数据目录在 `config.yaml` 中以 `dataset.path` 指定，内部 datamodule 根据该路径查找训练/验证/测试文件。

如果该默认约定不符合你的数据布局，请在 `dat_formmer/datamodule/dataset.py` 中查看和调整数据加载器的实现，或在 `config.yaml` 中修改路径参数。

## 配置说明（`config.yaml`）

`config.yaml` 包含模型、优化器、数据路径、训练超参等。

常见字段（示例说明）：
- `dataset.path`：数据根目录。
- `trainer.max_epochs`：训练轮数。
- `optimizer.lr`：初始学习率。
- `model.*`：模型结构相关参数（编码器层数、隐藏维度、注意力头数等）。

在开始训练前请打开并检查 `config.yaml`，确保 `dataset.path` 与本地数据一致。

## 检查点与日志

- 默认使用 PyTorch Lightning 保存训练日志与 checkpoint，目录为 `lightning_logs/`。
- checkpoint 文件通常包含模型权重与训练状态，可用于恢复训练或推理。

恢复训练示例：在 `config.yaml` 或 `train.py` 中指定 `resume_from_checkpoint`（或传入命令行参数）为相应的 checkpoint 路径。

## 推理与导出

仓库当前以训练与评估为主；若需单独的推理脚本，可在 `dat_formmer/` 下添加一个小脚本，加载 checkpoint 并对单张图片或一批图片执行预测。建议实现参数：`--checkpoint`, `--input`, `--output`。

示例（伪代码）:

```python
# load model from checkpoint
# preproc image
# run model.predict
# decode output to LaTeX / MathML / LG
```

如果你希望，我可以为这个仓库添加一个简单的推理脚本（包含 CLI），并编写相应的使用示例。

## 评估指标

- 本仓库使用 LgEval 工具集来计算结构相似度和编辑距离等 CROHME 常用指标。评估脚本位于 `lgeval/` 和 `scripts/`。
- 评估流程通常：将模型输出转换为标准 LG/MML 格式 -> 使用 LgEval 的 `evaluate` 工具生成分数。

## 常见问题与故障排查

- 若在导入依赖时报错，请确认 Python 版本 >= 3.8 且已经激活虚拟环境并安装了 `requirements.txt`。
- 若 GPU 不可见：检查 CUDA / cuDNN 是否正确安装、torch 是否匹配 GPU 版本。
- 若数据路径找不到：打开 `config.yaml` 并确认 `dataset.path` 指向正确的解压后数据目录。

## 开发建议 / 后续改进（可选）

- 增加一个 `scripts/infer.py` 用于单张图片推理与可视化。  
- 增加示例数据子集与下载说明，便于快速复现（当前仓库仅提供 `data.zip` 占位）。  
- 提供 Windows 下的评估替代脚本，便于不使用 WSL 的用户运行 `eval_all.sh`。

---

如果你愿意，我可以：
- 添加 `scripts/infer.py` 并写出推理示例；
- 或根据你的本地数据结构，帮助修改 `config.yaml` 与 datamodule 的数据读取逻辑以实现“一键训练”。

联系方式／作者信息请见仓库元数据。

License: 请查看仓库顶层或 `cc_license` 目录中的许可证信息。