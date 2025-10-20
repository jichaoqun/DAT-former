# DAT-former
Handwritten formula recognition

## project structure

DAT-formmer/
├── config.yaml           # 配置文件
├── train.py              # 训练入口脚本
├── data.zip              # 数据文件
├── dat_formmer/          # 核心模型代码
│   ├── datamodule/       # 数据处理模块
│   ├── model/            # 模型定义
│   │   ├── dat.py        # DAT模型
│   │   ├── encoder.py    # 编码器
│   │   └── decoder.py    # 解码器
│   └── utils/            # 工具函数
├── scripts/              # 评估和测试脚本
└── lgeval/               # 评估工具

## dataset

./date.zip: Store the crohme dataset file, including 2014, 2016 and 2019

## train    
```
python train.py
```
Note: The default dataset is the 2014 dataset, and the 2016 and 2019 datasets are not used by default. If you want to use the 2016 and 2019 datasets, you need to modify the config.yaml file and add the dataset path.

## eval
```
sh eval.sh
```