# ICAMGP: Intelligent Clustering Attention Model for Genomic Prediction

一个整合特征聚类、特征选择和基于注意力机制的深度学习模型的**基因组预测系统**。通过聚类-注意力框架提高复杂基因组数据的预测精度。

## 📋 项目概述

ICAMGP（Intelligent Clustering Attention Model for Genomic Prediction）是一套完整的基因组预测流水线，专门用于处理高维SNP特征数据。系统核心特点：

- **特征聚类**：支持多种聚类算法（HDBSCAN、KMeans、DBSCAN等），基于LD块原理或物理位置对SNP特征进行分组
- **注意力机制**：采用簇内（局部）和簇间（全局）两层注意力架构，捕捉特征间的复杂交互
- **特征选择**：通过p值和LD阈值进行特征筛选，降低维度
- **多模型支持**：包括CAM（聚类注意力模型）、MLP基线、BLUP等传统方法
- **消融研究**：支持woLocal、woGlobal、woCluster等变体，便于模型分析

---

## 🚀 快速开始

### 环境创建

```bash
# 创建 conda 环境
conda create -n cam python=3.12 numpy=1.26 pandas scikit-learn matplotlib scipy seaborn statsmodels
conda activate cam

# 安装 PyTorch（CUDA 12.8）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install pandas-plink hdbscan
```

### 基本使用

```python
from pipeline import run_pipeline

# 运行完整流水线
result = run_pipeline(
    dataset_name='CLMA',
    cluster_method='hdbscan_ldblock',
    cluster_params={'ld_threshold': 0.2, 'min_cluster_size': 5},
    enable_feature_selection=True,
    selection_params={'p_threshold': 0.05},
    model_type='CAM',
    model_params={'hidden_dim': 32, 'dropout': 0.1},
    training_params={'k1_fold': 5, 'num_restarts': 1},
    fit_method='torch_cv_fit',
    random_seed=42
)

# 获取预测结果和性能指标
predictions = result['predictions']
metrics = result['metrics']
print(metrics)
```

---

## 📁 项目结构

```
ICAMGP/
├── README.md                          # 项目说明文档
├── pipeline.py                        # 主流水线模块（端到端处理）
├── cam.py                            # 注意力模型实现
├── mynet.py                          # 模型训练和评估框架
├── feature_cluster.py                # 特征聚类算法集合
├── feature_select.py                 # 特征选择模块
├── rrblup.py                         # BLUP方法实现
├── dataset/
│   └── new_dataset.py               # 数据加载与处理
├── algorithm_comparison_results.csv  # 算法对比结果
├── ld_threshold_search_results.csv  # LD阈值搜索结果
└── p_threshold_search_results.csv   # p值阈值搜索结果
```

---

## 🔑 核心模块详解

### 1. **pipeline.py** - 端到端流水线

主函数 `run_pipeline()` 执行完整的基因组预测流程：

#### 流水线步骤：
1. **数据加载** - 从配置文件加载数据集
2. **特征聚类** - 按指定方法对SNP特征进行分组
3. **特征选择** - 基于p值和LD过滤特征
4. **模型训练** - 使用选定的模型进行交叉验证
5. **结果输出** - 返回预测结果和性能指标

#### 参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_name` | str | - | 数据集名称（配置文件中的key） |
| `cluster_method` | str | 'hdbscan_ldblock' | 聚类方法 |
| `model_type` | str | 'CAM' | 模型类型（MLP/CAM/woGlobal/woLocal/woCluster） |
| `fit_method` | str | 'torch_cv_fit' | 训练方法（torch_cv_fit/blup_fit/tradition_fit） |
| `k1_fold` | int | 5 | 交叉验证折数 |

#### 支持的模型类型：

- **CAM** - 完整聚类注意力模型（局部+全局注意力）
- **MLP** - 多层感知机基线
- **woLocal** - 消融：移除簇内注意力
- **woGlobal** - 消融：移除簇间注意力
- **woCluster** - 消融：所有特征视为单一簇
- **BLUP** - 传统贝叶斯方法
- **RandomForest/SVM/Ridge** - 传统机器学习方法

---

### 2. **cam.py** - 聚类注意力模型核心

**ClusterSparseAttentionModel** 是系统的核心深度学习模型：

#### 模型架构：

```
Input Features [B, F]
    ↓
[1] Feature Embedding & Gather → [B, C, L, H]
    ↓
[2] Local Attention (Intra-cluster) → [B, C, L, H]
    ↓
[3] Aggregation (Mean Pooling) → [B, C, H]
    ↓
[4] Global Attention (Inter-cluster) → [B, C, H]
    ↓
[5] Decoder → [B, output_dim]
```

#### 关键参数：

- `feature_class` - 每个特征所属的簇ID
- `hidden_dim` - 隐层维度（默认8）
- `num_heads` - 注意力头数（默认1）
- `ablation_type` - 消融类型（woLocal/woGlobal/woCluster）

#### 特色设计：

1. **分层注意力** - 先在簇内做局部attention，再在簇间做全局attention
2. **稀疏计算** - 利用聚类结构减少计算复杂度
3. **分块处理** - 大规模数据自动分块处理，避免OOM
4. **注意力可视化** - 支持返回attention权重用于解释性分析

---

### 3. **feature_cluster.py** - 多种聚类算法

提供13+种聚类方法供用户选择：

#### 主要聚类方法：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `by_chr` | 按染色体分组 | 简单快速的基线 |
| `by_chr_window` | 染色体+窗口划分 | 考虑物理位置 |
| `position_cluster_chr` | 每条染色体层次聚类 | 基于位置的精细聚类 |
| `kmeans` | K-means聚类 | 特征表达型聚类 |
| `agg` | 层次聚类 | 生成聚类树 |
| `hdbscan_ldblock` | LD块+HDBSCAN | **推荐**：结合遗传学和聚类理论 |
| `dbscan` / `dbscan_chr` | DBSCAN聚类 | 自适应密度聚类 |

#### 最推荐的方法：**hdbscan_ldblock**

```python
# 基于LD块的HDBSCAN聚类
cluster_with_hdbscan_ldblock(
    features,
    threshold=0.2,           # LD相关性阈值
    min_cluster_size=5,      # 最小簇大小
    assign_outliers=True     # 将离群点分配到最近簇
)
```

---

### 4. **feature_select.py** - 特征筛选

基于关联统计的特征选择模块：

```python
selected_features, selected_feature_class = feature_selector(
    dataset,
    feature_class=feature_class,
    p_threshold=0.05,        # p值阈值
    max_snp=10000            # 最多保留特征数
)
```

---

### 5. **mynet.py** - 模型训练框架

统一的模型训练和评估接口：

```python
net = MyNet(
    dataset=dataset,
    torch_selection=TorchModelType.CAM,
    feature_class=feature_class,
    k1_fold=5,
    model_params={'hidden_dim': 32, 'dropout': 0.1}
)

# 使用PyTorch进行交叉验证
df_predictions, df_metrics = net.torch_cv_fit(
    num_restarts=1,
    early_stopping_patience=5
)

# 或使用传统方法
df_predictions, df_metrics = net.tradition_fit(n_jobs=-1)
```

---

## 📊 输出结果

### 预测结果 (DataFrame)
```
Sample_ID  Predicted_Value  True_Value  Residual
S1         2.45            2.50        -0.05
S2         3.12            3.08        0.04
...
```

### 性能指标 (DataFrame)
```
Metric          Value
R²              0.852
RMSE            0.156
MAE             0.123
Correlation     0.923
```

---

## 🔬 实验配置示例

### 基础配置 - 快速测试

```python
result = run_pipeline(
    dataset_name='CLMA',
    model_type='CAM',
    cluster_params={'ld_threshold': 0.2},
    model_params={'hidden_dim': 16},
    training_params={'k1_fold': 3},
    fit_method='torch_cv_fit'
)
```

### 进阶配置 - 超参数调优

```python
# 网格搜索LD阈值
for ld_thresh in [0.1, 0.2, 0.3, 0.5]:
    result = run_pipeline(
        dataset_name='CLMA',
        cluster_params={'ld_threshold': ld_thresh},
        model_params={'hidden_dim': 32, 'dropout': 0.15},
        training_params={'k1_fold': 5, 'num_restarts': 3},
    )
    print(f"LD={ld_thresh}: {result['metrics']}")
```

### 消融实验

```python
# 对比不同的模型变体
for model_type in ['CAM', 'woLocal', 'woGlobal', 'woCluster', 'MLP']:
    result = run_pipeline(
        dataset_name='CLMA',
        model_type=model_type,
        fit_method='torch_cv_fit'
    )
    print(f"{model_type}: R²={result['metrics'].loc['R²', 'Value']}")
```

---

## 📈 性能对比

本仓库包含对比结果：

- `algorithm_comparison_results.csv` - 不同算法的性能对比
- `ld_threshold_search_results.csv` - LD阈值敏感性分析
- `p_threshold_search_results.csv` - p值阈值敏感性分析

---

## 🔍 注意力可视化

```python
# 获取注意力权重用于可解释性分析
model = ClusterSparseAttentionModel(feature_class, hidden_dim=32)
output, local_attn_dict, global_attn = model(features, return_attn=True)

# local_attn_dict: 每个簇的内部注意力权重
# global_attn: 簇间的全局注意力权重
```

---

## 📚 数据集格式

数据集应包含：
- **特征矩阵**：行为样本，列为SNP标记（格式: X{染色体}_{物理位置}_{等位基因}）
- **表型标签**：连续值或二分类标签
- **元数据**：可选的样本/特征注释

### 特征列名示例
```
X1_12345_A    X1_56789_G    X2_123456_T    ...
X2_654321_C   X3_111111_A   ...
```

---

## 🛠️ 依赖项

- Python 3.12+
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- scipy, matplotlib, seaborn, statsmodels
- hdbscan, pandas-plink

---

## 📖 使用流程总结

1. **数据准备** → 组织SNP特征数据和表型标签
2. **聚类配置** → 选择合适的聚类方法（推荐hdbscan_ldblock）
3. **模型选择** → 选择模型类型（推荐CAM）
4. **流水线执行** → 调用run_pipeline()
5. **结果分析** → 检查预测精度和注意力权重
6. **超参优化** → 迭代调整参数以改进性能

---

## 📝 引用

如果您使用本项目，请引用相关研究工作。

---

## 📞 联系方式

有问题或建议，欢迎提出Issue或Pull Request。

---

## 📄 许可证

[根据需要添��许可证信息]

---

**最后更新**: 2026年6月
