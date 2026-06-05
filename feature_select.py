import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List, Tuple
from dataset.new_dataset import MyDataset

def prepare_labels_for_selection(labels, strategy='single_column', target_col=0):
    """
    将 labels 预处理成特征选择可接受的一维向量
    支持策略: 'mean', 'argmax', 'single_column'

    Args:
        labels (np.ndarray or pd.DataFrame): [N, C]
        strategy (str): 处理策略，默认 'mean'
        target_col (int): 如果选用 'single_column'，指定第几列
    Returns:
        y (np.ndarray): [N]
    """
    if isinstance(labels, (pd.DataFrame, pd.Series)):
        labels = labels.values

    labels = np.array(labels)
    if labels.ndim == 1:
        return labels
    elif labels.ndim == 2:
        if labels.shape[1] == 1:
            return labels[:, 0]
        elif strategy == 'mean':
            return labels.mean(axis=1)
        elif strategy == 'argmax':
            return labels.argmax(axis=1)
        elif strategy == 'single_column':
            return labels[:, target_col]
        else:
            raise ValueError(f"不支持的 strategy: {strategy}")
    else:
        raise ValueError("labels 必须是一维或二维数组")

def feature_selector(dataset: MyDataset,
                     feature_class: List[int],
                     p_threshold: float = 0.05,
                     max_snp: int = 10000,
                     label_strategy='single_column') -> Tuple[pd.DataFrame, List[int]]:
    """
    基于皮尔逊相关系数进行特征选择，并同步过滤 feature_class。
    特征选择仅基于 SNP 与 标签 的相关性，与聚类结果无关。

    Args:
        dataset (MyDataset): 数据集实例
        feature_class (List[int]): 原始特征对应的聚类标签列表，长度需与 dataset.features 列数一致
        p_threshold (float): P 值显著性阈值
        max_snp (int): 最大保留的 SNP 数量，默认 10000
        label_strategy (str): 标签处理策略

    Returns:
        Tuple[pd.DataFrame, List[int]]: (筛选后的特征矩阵, 对应的聚类标签列表)
    """
    features = dataset.features
    labels = dataset.labels

    if not isinstance(features, pd.DataFrame):
        raise TypeError("dataset.features 必须为 pandas.DataFrame")
    
    if len(feature_class) != features.shape[1]:
        raise ValueError(f"feature_class 长度 ({len(feature_class)}) 与特征数量 ({features.shape[1]}) 不一致")
    
    # 准备标签
    y = prepare_labels_for_selection(labels, strategy=label_strategy)

    print(f"执行 Pearson 特征选择, p_threshold={p_threshold}...")
    
    selected_indices = []
    scores = []
    
    # 遍历所有特征计算 Pearson 相关系数
    X_values = features.values
    n_features = X_values.shape[1]
    
    for i in range(n_features):
        col_data = X_values[:, i]
        try:
            # pearsonr 返回 (correlation, p-value)
            corr, p_value = pearsonr(col_data, y)
        except Exception:
            # 处理可能的常数序列导致的错误
            corr, p_value = 0, 1.0
            
        if np.isnan(corr):
            corr = 0
        if np.isnan(p_value):
            p_value = 1.0
            
        if p_value < p_threshold:
            selected_indices.append(i)
            scores.append(abs(corr))
    
    # 按相关性分数倒序排列
    temp_scores = np.array(scores)
    sort_idx = np.argsort(temp_scores)[::-1]
    final_indices = [selected_indices[i] for i in sort_idx]

    # 限制最大 SNP 数量
    if len(final_indices) > max_snp:
        print(f"筛选后特征数量 ({len(final_indices)}) 超过 max_snp ({max_snp})，截断至前 {max_snp} 个。")
        final_indices = final_indices[:max_snp]

    # 构建结果
    selected_features = features.iloc[:, final_indices]
    selected_feature_class = [feature_class[i] for i in final_indices]
    
    print(f"特征选择完成: 原始 {n_features} -> 筛选后 {len(final_indices)}")

    return selected_features, selected_feature_class
