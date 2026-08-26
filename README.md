## DAT-former
This repository contains the source code, configuration files, and evaluation scripts for the revised DAT-Former experiments on handwritten mathematical expression recognition (HMER).

The package is intended to reproduce the results reported in the submitted manuscript, including the main comparison tables, ablation studies, and CROHME evaluation results. The exact release tag, checkpoints, logs, and supplementary artifacts used for the manuscript should also be archived on Zenodo.


<table align="center">
  <tr>
    <!-- 左图 -->
    <td>
      <img src="image.png" width="500">
    </td>
    <!-- 右侧两张上下排列 -->
    <td>
      <img src="image-1.png" width="500"><br><br>
      <img src="image-2.png" width="500">
    </td>
  </tr>
</table>


## Abstract
Handwritten Mathematical Expression Recognition (HMER) is a crucial technology for converting handwritten formulas into machine-readable formats, with wide applications in digital education and scholarly communication. While the dominant CNN-Transformer architecture has shown promise, it suffers from two fundamental bottlenecks: the inefficiency of CNNs in modeling long-range dependencies due to their local receptive fields, and signal conflicts in coverage mechanisms caused by heterogeneous attention. To overcome these dual challenges, this paper introduces DAT-Former, a novel architecture featuring two synergistic innovations: a globally-aware hybrid encoder with a task-adaptive 2D Rotary Position Embedding (2D-RoPE) to explicitly capture spatial topology, and an Adaptive Gated Coverage Module (AGCM) that uses a data-driven gate to resolve attention conflicts. Extensive experiments demonstrate state-of-the-art recognition rates of 63.86%, 60.51%, and 64.89% on CROHME 2014/16/19, respectively, and exceptional generalization on HME100K. This work highlights that a carefully designed sequence-to-sequence paradigm can rival more complex tree-based approaches, setting a new benchmark for robust HMER.

## 方法概述


## 目录概览

以下为仓库目录：

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

## Environment

The revised codebase targets the following environment:

```text
Python: 3.9
PyTorch: 2.8.0
TorchVision: 0.23.0
Lightning: 2.5.2
CUDA: compatible with the installed PyTorch build
GPU used in our experiments: NVIDIA RTX 5880 Ada Generation / equivalent CUDA GPU
```

Install the package and dependencies:

```bash
conda create -n datformer python=3.9
conda activate datformer

pip install torch==2.8.0 torchvision==0.23.0
pip install lightning==2.5.2
pip install "jsonargparse[signatures]==4.49.0"
pip install -r requirements.txt
pip install -e .
```

For official CROHME evaluation, Perl and the CROHME evaluation tools are required:

```bash
perl --version
```

If TensorBoard raises a NumPy compatibility error, use a TensorBoard/NumPy combination compatible with the installed PyTorch environment.

## Data Preparation

The expected data archive is:

```text
data.zip
```

After extraction, the directory should contain the official CROHME training data and the CROHME 2014/2016/2019 test sets:

```text
data/
├── train/
├── 2014/
├── 2016/
└── 2019/
```

If the dataset cannot be redistributed, download it from the official CROHME source and place it in the format above. Before release, provide checksums for the local archive used in the manuscript, for example:

```bash
sha256sum data.zip
```

## Experimental Protocol

To avoid test-set reuse, model selection and hyperparameter tuning are performed only on the official CROHME training set.

The default revised protocol is:

```text
1. Split the official CROHME training set into 90% training and 10% validation data.
2. Use the validation split for checkpoint selection and hyperparameter tuning.
3. Keep CROHME 2014/2016/2019 only as final test sets.
4. Report the final test results using the official CROHME evaluation scripts.
```

The split is controlled by:

```yaml
data:
  val_ratio: 0.1
  split_seed: 7
```

All ablation variants should use the same split, seed, augmentation, training schedule, checkpoint selection metric, and evaluation script.

## Training

Run DAT-Former with:

```bash
python train.py fit --config config.yaml
```

The current configuration uses LightningCLI. Therefore, `fit`, `validate`, `test`, or `predict` must be provided as the subcommand.

Important configuration entries:

```yaml
seed_everything: 7

trainer:
  accelerator: gpu
  devices: [0, 1]
  strategy: ddp_find_unused_parameters_true
  max_epochs: 301
  check_val_every_n_epoch: 5

model:
  learning_rate: 0.08
  dropout: 0.3
  beam_size: 10
  max_len: 100

data:
  zipfile_path: ./data.zip
  train_batch_size: 32
  eval_batch_size: 16
  scale_aug: true
  val_ratio: 0.1
  split_seed: 7
```

Before training, make sure that `CUDA_VISIBLE_DEVICES` and `trainer.devices` refer to the same visible GPUs.

For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py fit --config config.yaml
```

## Evaluation

Evaluate one trained version on a selected CROHME test set:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test/eval.sh <version> 2014 4
CUDA_VISIBLE_DEVICES=0 bash scripts/test/eval.sh <version> 2016 4
CUDA_VISIBLE_DEVICES=0 bash scripts/test/eval.sh <version> 2019 4
```

For example, if the checkpoint is stored under `lightning_logs/version_best/`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test/eval.sh best 2014 4
```

Evaluate all CROHME test sets:

```bash
bash eval_all.sh <version>
```

The result files are written to:

```text
lightning_logs/version_<version>/2014.txt
lightning_logs/version_<version>/2016.txt
lightning_logs/version_<version>/2019.txt
```


## Random Seeds and Hardware Details

The default seed is:

```yaml
seed_everything: 7
```

When reporting ablation studies, use the same seed and data split across all variants. If multiple seeds are reported, include the seed list and report mean and standard deviation.

Record the following details for each released run:

```text
GPU model and number of GPUs
CUDA version
Python version
PyTorch version
Lightning version
Training batch size
Evaluation batch size
Random seed
Validation split seed
Checkpoint selection metric
Training time
```

## Checkpoints and Logs

Pretrained checkpoints should be provided through the GitHub release or Zenodo archive. A reproducible release should include:

```text
checkpoint.ckpt
config.yaml
hparams.yaml
training log
evaluation outputs
result.zip
2014.txt / 2016.txt / 2019.txt
```

Large checkpoints should be stored in the release assets or Zenodo rather than committed directly to the Git repository.

## Citation

If you use this repository, please cite the manuscript and the archived Zenodo release:

```bibtex
@article{datformer,
  title   = {DAT-Former: <replace with final title>},
  author  = {<replace with authors>},
  journal = {<replace with journal>},
  year    = {2026}
}
```

## License

A clear open-source license should be included in the repository before public release. The recommended option is to add a `LICENSE` file at the repository root.

The CROHME datasets may have separate redistribution restrictions. If redistribution is not explicitly permitted by the dataset license, do not redistribute CROHME data through this repository. Instead, provide official download instructions and checksums.

