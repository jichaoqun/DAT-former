# DAT-former
Handwritten formula recognition

## project structure

DAT-formmer/
├── config.yaml        # 配置文件
├── train.py           # 训练入口脚本
├── data.zip           # 数据文件
├── dat_formmer/
│   ├── datamodule/
│   ├── model/
│   │   ├── dat.py
│   │   ├── encoder.py
│   │   └── decoder.py
│   └── utils/
├── scripts/
└── lgeval/


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