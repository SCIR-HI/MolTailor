# MolTailor
文章[MolTailor: Tailoring Chemical Molecular Representation to Specific Tasks via Text Prompts](https://arxiv.org/abs/2401.11403)的源码（发表在AAAI 2024）。

<div style="background-color: #ffffcc; padding: 10px; border-left: 6px solid #ffeb3b;">
  <strong>注意：</strong>在本项目中MolTailor被命名为DEN
</div>

![MolTailor](./src/overall.svg)

## 文件结构
```bash
.
├── mt-mtr-build # 用于构建MT-MTR预训练预料的代码
│   ├── 01-get-smiles.py # 合并来自drugbank与chebi的smiles
│   ├── 02-calculate-descriptors.py # 使用RDKit计算分子的描述符
│   ├── 03-generate-descriptions.py # 使用GPT-3.5生成任务描述
│   ├── 04-build-mt-mtr.py # 构建MT-MTR预料
│   └── data
│       ├── temporary # 存储中间数据
│       ├── mt-mtr.pt # MT-MTR
│       ├── mt-mtr-clean.pt # 去除与8个来自MoleculeNet数据集重复分子的MT-MTR
│       ├── mt-mtr-origin.jsonl # 包含全部分子属性的MT-MTR
│       └── mt-mtr-origin-clean.jsonl
├── pretrain # 预训练代码
│   ├── data # 存储预训练预料mt-mtr.pt
│   ├── workspace # checkpoint和log文件
│   ├── models # 模型结构
│   │   ├── bert_uncased # backbone模型代码
│   │   ├── ...
│   │   ├── bert.py # 基于bert代码，构建Multimodal T-Encoder
│   │   ├── config.py # MolTailor配置文件
│   │   ├── den.py # MolTailor的具体实现
│   │   ├── load.py # 加载backbone
│   │   └── multitask.py # 预训练任务代码
│   ├── data_collator.py
│   ├── data_modules.py
│   ├── dataset.py
│   ├── debug.py
│   ├── main.py # 入口文件
│   ├── train.py
│   └── tune.py # batch_size与学习率的搜索
├── linear-probe-moleculenet # 下游任务代码
│   ├── models # MolTailor及Baseline模型的代码实现
│   ├── data # MoleculeNet数据集及相关预处理代码
│   │   ├── feature # 模型提取的分子的Embedding特征
│   │   ├── raw # 原始的MoleculeNet数据集
│   │   ├── utils
│   │   │   ├── preprocess.py # MoleculeNet预处理代码
│   │   │   └── feature-extract.py # 提取分子的Embedding特征
│   │   ├── prompt4molnet.json # 用于下游任务的MolTailor的文本提示
│   │   ├── bbbp.csv # 预处理后的MoleculeNet数据集
│   │   └── ...
│   ├── workspace # checkpoint和log文件
│   ├── callbacks.py # lightning的回调函数
│   ├── data_modules.py
│   ├── dataset.py
│   ├── main.py # 入口文件
│   ├── metrics.py # roc_auc与delta ap的方法实现
│   ├── multi_seeds.py
│   ├── split.py # 数据集random与scaffold分割函数实现
│   ├── train.py
│   └── tune.py # 使用optuna搜索学习率
├── linear-probe-moleculenet-lite # 移除Uni-Mol模型的下游任务代码，文件结构与上面相同
│   └── ...
├── models # 模型权重
│   ├── bert-base-uncased
│   ├── CHEM-BERT
│   ├── ChemBERTa-10M-MTR
│   ├── DEN # MolTailor在本项目中用DEN命名，故权重存储在DEN文件夹下
│   └── ...
├── readme
├── readme.md
├── requirements-lite.txt # 移除Uni-Mol模型的依赖
├── requirements.txt # 完整依赖
└── scripts
    ├── pretrain.sh # 预训练脚本
    ├── convert_ckpt.sh # 将预训练后的模型移动到models文件夹，并进行转换
    ├── linear-probe-molnet.sh # 下游任务脚本
    └── linear-probe-molnet-lite.sh
```