# ICAMGP: Intelligent Clustering Attention Model for Genomic Prediction

A comprehensive genomic prediction system that integrates feature clustering, feature selection, and attention-based deep learning models. Leveraging a clustering-attention framework to improve prediction accuracy on complex genomic data.

## 📋 Project Overview

ICAMGP (Intelligent Clustering Attention Model for Genomic Prediction) is a complete genomic prediction pipeline specifically designed for high-dimensional SNP feature data. Key features of the system:

- **Feature Clustering**: Supports multiple clustering algorithms (HDBSCAN, KMeans, DBSCAN, etc.), grouping SNP features based on LD blocks or physical positions
- **Attention Mechanism**: Employs a two-layer attention architecture with intra-cluster (local) and inter-cluster (global) attention to capture complex feature interactions
- **Feature Selection**: Filters features based on p-values and LD thresholds to reduce dimensionality
- **Multi-Model Support**: Includes CAM (Cluster Attention Model), MLP baseline, BLUP, and other traditional methods
- **Ablation Studies**: Supports variants like woLocal, woGlobal, woCluster for model analysis

---

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n cam python=3.12 numpy=1.26 pandas scikit-learn matplotlib scipy seaborn statsmodels
conda activate cam

# Install PyTorch (CUDA 12.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies
pip install pandas-plink hdbscan
```

### Basic Usage

```python
from pipeline import run_pipeline

# Run the complete pipeline
result = run_pipeline(
    dataset_name='Env1',
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

# Get predictions and performance metrics
predictions = result['predictions']
metrics = result['metrics']
print(metrics)
```

---

## 📁 Project Structure

```
ICAMGP/
├── README.md                          # Project documentation
├── pipeline.py                        # Main pipeline module (end-to-end processing)
├── cam.py                            # Attention model implementation
├── mynet.py                          # Model training and evaluation framework
├── feature_cluster.py                # Feature clustering algorithms
├── feature_select.py                 # Feature selection module
├── rrblup.py                         # BLUP method implementation
├── dataset/
│   └── new_dataset.py               # Data loading and processing
├── algorithm_comparison_results.csv  # Algorithm comparison results
├── ld_threshold_search_results.csv  # LD threshold search results
└── p_threshold_search_results.csv   # P-value threshold search results
```

---

## 🔑 Core Modules

### 1. **pipeline.py** - End-to-End Pipeline

The main function `run_pipeline()` executes the complete genomic prediction process:

#### Pipeline Steps:
1. **Data Loading** - Load dataset from configuration file
2. **Feature Clustering** - Group SNP features using specified method
3. **Feature Selection** - Filter features based on p-values and LD
4. **Model Training** - Train selected model with cross-validation
5. **Result Output** - Return predictions and performance metrics

#### Parameter Reference:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | - | Dataset name (key in config file) |
| `cluster_method` | str | 'hdbscan_ldblock' | Clustering method to use |
| `model_type` | str | 'CAM' | Model type (MLP/CAM/woGlobal/woLocal/woCluster) |
| `fit_method` | str | 'torch_cv_fit' | Training method (torch_cv_fit/blup_fit/tradition_fit) |
| `k1_fold` | int | 5 | Number of cross-validation folds |

#### Supported Model Types:

- **CAM** - Complete cluster attention model (local + global attention)
- **MLP** - Multi-layer perceptron baseline
- **woLocal** - Ablation: remove intra-cluster attention
- **woGlobal** - Ablation: remove inter-cluster attention
- **woCluster** - Ablation: treat all features as single cluster
- **BLUP** - Traditional Bayesian method
- **RandomForest/SVM/Ridge** - Traditional machine learning methods

---

### 2. **cam.py** - Cluster Attention Model

**ClusterSparseAttentionModel** is the core deep learning model of the system:

#### Model Architecture:

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

#### Key Parameters:

- `feature_class` - Cluster ID for each feature
- `hidden_dim` - Hidden dimension size (default: 8)
- `num_heads` - Number of attention heads (default: 1)
- `ablation_type` - Ablation type (woLocal/woGlobal/woCluster)

#### Design Features:

1. **Hierarchical Attention** - Local attention within clusters, then global attention between clusters
2. **Sparse Computation** - Leverage clustering structure to reduce computational complexity
3. **Chunked Processing** - Automatically process large-scale data in chunks to avoid OOM
4. **Attention Visualization** - Support returning attention weights for interpretability

---

### 3. **feature_cluster.py** - Clustering Algorithms

Provides 13+ clustering methods for user selection:

#### Main Clustering Methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `by_chr` | Group by chromosome | Fast baseline approach |
| `by_chr_window` | Chromosome + window division | Consider physical position |
| `position_cluster_chr` | Hierarchical clustering per chromosome | Position-based fine clustering |
| `kmeans` | K-means clustering | Feature expression clustering |
| `agg` | Agglomerative clustering | Generate clustering tree |
| `hdbscan_ldblock` | LD block + HDBSCAN | **Recommended**: combines genetics and clustering theory |
| `dbscan` / `dbscan_chr` | DBSCAN clustering | Adaptive density-based clustering |

#### Recommended Method: **hdbscan_ldblock**

```python
# HDBSCAN clustering based on LD blocks
cluster_with_hdbscan_ldblock(
    features,
    threshold=0.2,           # LD correlation threshold
    min_cluster_size=5,      # Minimum cluster size
    assign_outliers=True     # Assign outliers to nearest cluster
)
```

---

### 4. **feature_select.py** - Feature Selection

Feature selection module based on association statistics:

```python
selected_features, selected_feature_class = feature_selector(
    dataset,
    feature_class=feature_class,
    p_threshold=0.05,        # P-value threshold
    max_snp=10000            # Maximum number of features to retain
)
```

---

### 5. **mynet.py** - Training Framework

Unified model training and evaluation interface:

```python
net = MyNet(
    dataset=dataset,
    torch_selection=TorchModelType.CAM,
    feature_class=feature_class,
    k1_fold=5,
    model_params={'hidden_dim': 32, 'dropout': 0.1}
)

# PyTorch-based cross-validation
df_predictions, df_metrics = net.torch_cv_fit(
    num_restarts=1,
    early_stopping_patience=5
)

# Or use traditional methods
df_predictions, df_metrics = net.tradition_fit(n_jobs=-1)
```

---

## 📊 Output Results

### Predictions (DataFrame)
```
Sample_ID  Predicted_Value  True_Value  Residual
S1         2.45            2.50        -0.05
S2         3.12            3.08        0.04
...
```

### Performance Metrics (DataFrame)
```
Metric          Value
R²              0.852
RMSE            0.156
MAE             0.123
Correlation     0.923
```

---

## 🔬 Experiment Configuration Examples

### Basic Configuration - Quick Test

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

### Advanced Configuration - Hyperparameter Tuning

```python
# Grid search over LD thresholds
for ld_thresh in [0.1, 0.2, 0.3, 0.5]:
    result = run_pipeline(
        dataset_name='CLMA',
        cluster_params={'ld_threshold': ld_thresh},
        model_params={'hidden_dim': 32, 'dropout': 0.15},
        training_params={'k1_fold': 5, 'num_restarts': 3},
    )
    print(f"LD={ld_thresh}: {result['metrics']}")
```

### Ablation Study

```python
# Compare different model variants
for model_type in ['CAM', 'woLocal', 'woGlobal', 'woCluster', 'MLP']:
    result = run_pipeline(
        dataset_name='CLMA',
        model_type=model_type,
        fit_method='torch_cv_fit'
    )
    print(f"{model_type}: R²={result['metrics'].loc['R²', 'Value']}")
```

---

## 📈 Performance Comparison

This repository includes comparison results:

- `algorithm_comparison_results.csv` - Performance comparison across algorithms
- `ld_threshold_search_results.csv` - LD threshold sensitivity analysis
- `p_threshold_search_results.csv` - P-value threshold sensitivity analysis

---

## 🔍 Attention Visualization

```python
# Extract attention weights for interpretability analysis
model = ClusterSparseAttentionModel(feature_class, hidden_dim=32)
output, local_attn_dict, global_attn = model(features, return_attn=True)

# local_attn_dict: intra-cluster attention weights for each cluster
# global_attn: inter-cluster global attention weights
```

---

## 📚 Data Format

Dataset should contain:
- **Feature Matrix**: Rows are samples, columns are SNP markers (format: X{chromosome}_{physical_position}_{allele})
- **Phenotype Labels**: Continuous or binary classification labels
- **Metadata**: Optional sample/feature annotations

### Feature Column Name Example
```
X1_12345_A    X1_56789_G    X2_123456_T    ...
X2_654321_C   X3_111111_A   ...
```

---

## 🛠️ Dependencies

- Python 3.12+
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- scipy, matplotlib, seaborn, statsmodels
- hdbscan, pandas-plink

---

## 📖 Usage Workflow Summary

1. **Data Preparation** → Organize SNP feature data and phenotype labels
2. **Clustering Configuration** → Select appropriate clustering method (recommended: hdbscan_ldblock)
3. **Model Selection** → Choose model type (recommended: CAM)
4. **Pipeline Execution** → Call run_pipeline()
5. **Result Analysis** → Examine prediction accuracy and attention weights
6. **Hyperparameter Optimization** → Iteratively adjust parameters to improve performance

---

## 📝 Citation

If you use this project, please cite the relevant research work.

---

## 📞 Contact

For questions or suggestions, feel free to open Issues or submit Pull Requests.

---

## 📄 License

[Add license information as needed]

---

**Last Updated**: June 2026
