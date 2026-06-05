import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import Counter
from itertools import combinations
from sklearn.metrics import pairwise_distances
import hdbscan# 需要 pip install hdbscan
import matplotlib.pyplot as plt

import time

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录函数结束执行的时间
        print(f"function: {func.__name__}, cost_time: {end_time - start_time} s")
        return result
    return wrapper

def check_feature_column_format(columns):
    """
    检查特征列名格式是否符合规则：X<编号>_<位置>_...
    捕捉所有无法解析位置字段的列名

    Args:
        columns (List[str] or pd.Index): 特征列名

    Returns:
        List[str]: 有问题的列名
    """
    if isinstance(columns, pd.Index):
        columns = columns.tolist()

    error_columns = []

    for col in columns:
        parts = col.split('_')
        if len(parts) < 2:
            error_columns.append(f"❌ 少于两个分隔符: {col}")
            continue

        pos_str = parts[1]

        # 使用正则尝试提取数字或科学记数法
        if not re.match(r'^[\d\.eE+-]+$', pos_str):
            error_columns.append(f"❌ 非法位置字段: {col}")
            continue

        # 再次确认能否被 float 转换
        try:
            _ = float(pos_str)
        except:
            error_columns.append(f"❌ 无法转换为float: {col}")

    return error_columns

def extract_chromosome_classes(feature_names):
    """
    提取每个特征对应的染色体编号，例如 'X1_12345_A' → 1

    Args:
        feature_names (List[str] or pd.Index): 特征名列表或列名索引

    Returns:
        List[int]: 每个特征对应的染色体编号
    """
    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()

    chrom_classes = []
    for name in feature_names:
        match = re.match(r"X(\d+)_", name)
        if match:
            chrom_classes.append(int(match.group(1)))
        else:
            raise ValueError(f"无法从特征名 '{name}' 中提取染色体编号")
    return chrom_classes

def sort_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    根据特征列名中的“染色体编号 + 物理位置”对 DataFrame 的列进行排序。
    支持列名格式为 'X1_12345_A'。

    参数：
        features (pd.DataFrame): 每列为一个特征，列名格式为 'X1_12345_A'

    返回：
        pd.DataFrame: 按染色体+位置排序后的新 DataFrame
    """
    def parse_chr_pos(col: str):
        chr_, pos_str = col.split('_')[:2]
        chr_num = int(chr_.replace('X', ''))  # 如 X1 → 1
        pos = int(float(pos_str))
        return chr_num, pos

    sorted_cols = sorted(features.columns, key=parse_chr_pos)
    return features[sorted_cols]



def position_cluster_with_chr(columns, n_clusters_per_chr=3):
    """
    按染色体编号分别进行位点聚类（层次聚类），不同染色体的聚类编号保持唯一

    Args:
        columns (List[str] or pd.Index): 特征名（格式如 'X1_12345_A'）
        n_clusters_per_chr (int): 每条染色体内部聚类数

    Returns:
        List[int]: 每个特征所属的聚类编号
    """
    chr_pos_dict = {}  # key: 染色体编号，value: List of (index, position)
    for idx, col in enumerate(columns):
        chr_, pos = col.split('_')[:2]
        pos = int(float(pos))
        chr_pos_dict.setdefault(chr_, []).append((idx, pos))

    feature_class = [None] * len(columns)
    class_id_counter = 0

    for chr_, items in chr_pos_dict.items():
        indices, positions = zip(*items)
        positions = np.array(positions).reshape(-1, 1)

        if len(positions) < n_clusters_per_chr:
            labels = [0] * len(positions)  # 少于聚类数则全归一类
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters_per_chr)
            labels = model.fit_predict(positions)

        for i, idx in enumerate(indices):
            feature_class[idx] = class_id_counter + labels[i]
        class_id_counter += max(labels) + 1

    return feature_class


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

def knn_gaussian_cosine_net(features: pd.DataFrame, k=50, sigma= None):
    """
    基于 kNN + 余弦相似度 + 高斯核构建稀疏相似网络

    Args:
        features (pd.DataFrame): 输入特征矩阵 (样本 × 特征)，行=样本，列=特征
        k (int): kNN 邻居数
        sigma (float): 高斯核带宽，若 None 则取非零余弦距离的中位数

    Returns:
        H_df (pd.DataFrame): 特征稀疏相似矩阵 (p × p)，Hadamard 稀疏化结果
        sigma (float): 实际使用的高斯核带宽
    """
    X = features.values.T
    index = features.T.index

    # 1. 计算余弦相似度（or hamming距离）
    # ham_dist = pairwise_distances(X, metric="hamming")
    # cos_sim = 1 - ham_dist
    cos_sim = cosine_similarity(X)
    cos_sim = np.clip(cos_sim, -1, 1)

    # 2. 转换为距离并施加高斯核
    dist = 1 - cos_sim
    if sigma is None:
        sigma = np.median(dist[dist > 0])  # 非零部分的中位数
    K = np.exp(- dist**2 / (2 * sigma**2))

    # 3. kNN 邻接矩阵 (0/1)
    knn_graph = kneighbors_graph(
        X, n_neighbors=k, mode="connectivity", metric="cosine", include_self=False
    )
    # 2. mutual kNN
    knn_graph = knn_graph.minimum(knn_graph.T)
    # 3. 转成 dense 矩阵 (0/1 int)
    knn_mask = knn_graph.toarray().astype(int)
    np.fill_diagonal(knn_mask, 1)

    # 4. Hadamard 积（稀疏化）
    H = K * knn_mask

    print(f"构建了 kNN+高斯核稀疏相似图, sigma={sigma:.4f}, 平均度={knn_mask.sum(axis=1).mean():.2f}")

    return pd.DataFrame(H, index=index, columns=index)

@time_decorator
def ld_net(features: pd.DataFrame, threshold=0.2) -> pd.DataFrame:
    """
    基于LD block原理构建特征间的相关矩阵

    Args:
        features (pd.DataFrame): 输入特征矩阵 (样本 × 特征)，行=样本，列=特征
        threshold (float): 相关性阈值，低于该值的边被移除

    Returns:
        pd.DataFrame: 特征稀疏相似矩阵 (p × p)
    """
    # 计算相关系数矩阵
    X = features.values
    n_samples = X.shape[0]

    # 1. 标准化 (Z-score)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)  # ddof=1 对应样本标准差
    std[std == 0] = 1.0  # 防止除零
    Z = (X - mean) / std
    # 2. 矩阵乘法计算相关系数矩阵
    # Formula: R = (Z^T * Z) / n
    corr_matrix = np.dot(Z.T, Z) / (n_samples)
    # 3. 取绝对值并转为 DataFrame
    corr = pd.DataFrame(np.abs(corr_matrix), index=features.columns, columns=features.columns)
    # 4. 应用阈值过滤
    corr[corr < threshold] = 0
    
    return corr
    

from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import List, Tuple

def hierarchical_cluster_by_chr(features: pd.DataFrame, max_clusters=5, method='average') -> List[int]:
    def parse_chr(col):
        return col.split('_')[0]

    chr_dict = {}
    for i, col in enumerate(features.columns):
        chr_ = parse_chr(col)
        chr_dict.setdefault(chr_, []).append(i)

    cluster_labels = np.zeros(features.shape[1], dtype=int)
    offset = 0

    for chr_, indices in sorted(chr_dict.items()):
        if len(indices) <= 2:
            cluster_labels[indices] = offset
            offset += 1
            continue

        sub_features = features.iloc[:, indices].T
        corr = knn_gaussian_cosine_net(sub_features)
        distance = 1 - corr.abs()
        distance = np.clip(distance, 0, 1)  # 距离非负
        np.fill_diagonal(distance.values, 0.0)  # 显式设置对角线为0

        dist_array = squareform(distance.values, checks=False)
        Z = linkage(dist_array, method=method)
        labels = fcluster(Z, t=max_clusters, criterion='maxclust')

        for j, idx in zip(labels, indices):
            cluster_labels[idx] = j + offset
        offset = cluster_labels.max() + 1

    return cluster_labels.tolist()


from typing import List
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def auto_select_eps(distance_matrix, min_samples=2, plot=False):
    """
    利用 K-Distance 方法自动选择 eps，并可视化拐点
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np


    neigh = NearestNeighbors(n_neighbors=min_samples, metric='precomputed')
    neigh.fit(distance_matrix)
    distances, _ = neigh.kneighbors(distance_matrix)
    k_distances = np.sort(distances[:, -1])  # 取每个样本的第min_samples邻居距离并排序

    # 选择90%位置的距离作为近似拐点
    idx = int(len(k_distances) * 0.9)
    eps = k_distances[idx]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_distances, label=f"{min_samples}-NN Distance")
        plt.axvline(x=idx, color='r', linestyle='--', label=f"Point location (idx={idx})")
        plt.axhline(y=eps, color='r', linestyle='--', label=f"$\\varepsilon$ ≈ {eps:.4f}")
        plt.title("K-Distance Graph")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return eps



def cluster_by_chr_with_dbscan(features: pd.DataFrame, eps=None, min_samples=3, plot=True) -> List[int]:
    chr_dict = {}
    for i, col in enumerate(features.columns):
        chr_ = col.split('_')[0]
        chr_dict.setdefault(chr_, []).append(i)

    cluster_labels = np.full(features.shape[1], -1, dtype=int)
    offset = 0

    for chr_, indices in sorted(chr_dict.items()):
        if len(indices) <= 2:
            cluster_labels[indices] = np.arange(offset, offset + len(indices))
            offset += len(indices)
            continue

        sub_features = features.iloc[:, indices].T
        corr = knn_gaussian_cosine_net(sub_features)
        distance = 1 - corr.abs()
        distance = np.clip(distance, 0, 1)
        np.fill_diagonal(distance.values, 0.0)

        # 自动选择 eps
        eps_chr = eps
        if eps is None:
            eps_chr = auto_select_eps(distance.values, min_samples=min_samples, plot=plot)

        dbscan = DBSCAN(eps=eps_chr, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance.values)

        for idx, lbl in zip(indices, labels):
            if lbl == -1:
                cluster_labels[idx] = offset
                offset += 1
            else:
                cluster_labels[idx] = lbl + offset
        offset = cluster_labels.max() + 1

    return cluster_labels.tolist()

def cluster_with_dbscan(features: pd.DataFrame, eps=None, min_samples=3, plot=True) -> List[int]:
    """
    对所有特征整体进行 DBSCAN 聚类，不按染色体划分。

    参数：
        features: pd.DataFrame，行为样本，列为特征（SNP）
        eps: float，DBSCAN 的 epsilon 参数；若为 None，则调用 auto_select_eps 自动选择
        min_samples: int，DBSCAN 的最小样本数参数
        plot: bool，是否绘制自动选 eps 时的图

    返回：
        cluster_labels: List[int]，每列特征对应的聚类标签
    """
    # 计算相关性距离矩阵（特征之间的相关性）
    corr = knn_gaussian_cosine_net(features)
    distance = 1 - corr.abs()
    distance = np.clip(distance, 0, 1)
    np.fill_diagonal(distance.values, 0.0)

    # 自动选择 eps（如未指定）
    eps_val = eps
    if eps is None:
        eps_val = auto_select_eps(distance.values, min_samples=min_samples, plot=plot)

    # 进行 DBSCAN 聚类
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance.values)

    # 给未分类的特征分配唯一标签
    cluster_labels = np.array(labels)
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        max_label = cluster_labels[~noise_mask].max() + 1 if np.any(~noise_mask) else 0
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                cluster_labels[i] = max_label
                max_label += 1

    return cluster_labels.tolist()

def cluster_with_hdbscan(features: pd.DataFrame,
                         min_cluster_size=5,
                         min_samples=None,
                         assign_outliers=True):
    """
    使用 HDBSCAN 对稀疏网络进行聚类，并支持离群点归簇

    Args:
        features (pd.DataFrame): 输入特征矩阵 (样本 × 特征)，已转换为稀疏相似度/距离
        min_cluster_size (int): HDBSCAN 最小簇大小
        min_samples (int or None): HDBSCAN min_samples 参数
        assign_outliers (bool): 是否将离群点 (-1) 分配到最近簇

    Returns:
        labels (list[int]): 每个特征的簇标签
    """
    # 这里传入的是稀疏距离矩阵
    corr = knn_gaussian_cosine_net(features)  # 你之前的函数
    distance = 1 - corr.abs()
    distance = np.clip(distance, 0, 1)
    np.fill_diagonal(distance.values, 0.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed"
    ).fit(distance.values)

    labels = clusterer.labels_.copy()

    if assign_outliers:
        for i in range(len(labels)):
            if labels[i] == -1:
                # 找到该点到各簇的平均距离，分配到最近的簇
                dists = []
                for c in np.unique(labels):
                    if c == -1:
                        continue
                    cluster_points = np.where(labels == c)[0]
                    mean_dist = distance.values[i, cluster_points].mean()
                    dists.append((c, mean_dist))
                labels[i] = min(dists, key=lambda x: x[1])[0]

    return labels.tolist()


def cluster_by_chr_with_hdbscan(features: pd.DataFrame, min_cluster_size=5, min_samples=None) -> List[int]:
    """
    每条染色体单独做 HDBSCAN 聚类
    """
    chr_dict = {}
    for i, col in enumerate(features.columns):
        chr_ = col.split('_')[0]
        chr_dict.setdefault(chr_, []).append(i)

    cluster_labels = np.full(features.shape[1], -1, dtype=int)
    offset = 0

    for chr_, indices in sorted(chr_dict.items()):
        if len(indices) <= 2:
            cluster_labels[indices] = np.arange(offset, offset + len(indices))
            offset += len(indices)
            continue

        sub_features = features.iloc[:, indices].T
        corr = knn_gaussian_cosine_net(features)
        distance = 1 - corr.abs()
        distance = np.clip(distance, 0, 1)
        np.fill_diagonal(distance.values, 0.0)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='precomputed')
        labels = clusterer.fit_predict(distance.values)

        for idx, lbl in zip(indices, labels):
            if lbl == -1:
                cluster_labels[idx] = offset
                offset += 1
            else:
                cluster_labels[idx] = lbl + offset
        offset = cluster_labels.max() + 1
    return cluster_labels.tolist()

@time_decorator
def cluster_with_hdbscan_ldblock(features: pd.DataFrame, threshold=0.2,
                                 min_cluster_size=5,min_samples=None, assign_outliers=True) -> List[int]:
    """
    使用 HDBSCAN 对 LD block 网络进行聚类，并支持离群点归簇

    Args:
        features (pd.DataFrame): 输入特征矩阵 (样本 × 特征)
        min_cluster_size (int): HDBSCAN 最小簇大小
        min_samples (int or None): HDBSCAN min_samples 参数
        assign_outliers (bool): 是否将离群点 (-1) 分配到最近簇

    Returns:
        labels (list[int]): 每个特征的簇标签
    """
    # 构建 LD block 网络
    ld_corr = ld_net(features, threshold=threshold)
    distance = 1 - ld_corr
    distance = np.clip(distance, 0, 1)
    np.fill_diagonal(distance.values, 0.0)
    # 转换为 float64 类型以避免 HDBSCAN 警告
    distance_values = distance.values.astype(np.float64)
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed"
        ).fit(distance_values)
    except Exception as e:
        print("HDBSCAN 聚类时出错，可能是由于内存不足导致的OOM。错误信息：", e)
        raise e

    labels = clusterer.labels_.copy()

    if assign_outliers:
        for i in range(len(labels)):
            if labels[i] == -1:
                # 找到该点到各簇的平均距离，分配到最近的簇
                dists = []
                for c in np.unique(labels):
                    if c == -1:
                        continue
                    cluster_points = np.where(labels == c)[0]
                    mean_dist = distance.values[i, cluster_points].mean()
                    dists.append((c, mean_dist))
                labels[i] = min(dists, key=lambda x: x[1])[0]

    return labels.tolist()

@time_decorator
def cluster(features: pd.DataFrame, method='kmeans', n_clusters:int=10, window_size:int=100000, n_clusters_per_chr:int=3,
            eps=None, min_samples=3, min_cluster_size=5, ld_threshold=0.2) -> Tuple[List[int], List[int], dict]:
    """
    根据指定方法对特征进行聚类，支持按染色体/物理距离/表达模式聚类

    Args:
        features (pd.DataFrame): 输入特征矩阵，行为样本，列为特征
        method (str): 聚类方式，可选：
                      'by_chr'               - 按染色体编号分组
                      'by_chr_window'        - 染色体 + 窗口分段
                      'position_cluster'     - 所有特征按物理位置层次聚类（不区分染色体）
                      'position_cluster_chr' - 每条染色体内分别聚类
                      'kmeans'               - 特征表达型 KMeans 聚类
                      'agg'                  - 特征表达型层次聚类
                      'hierarchical_chr'     - 每条染色体层次聚类
                      'dbscan' / 'dbscan_chr' -  基于 DBSCAN 的聚类
                      'hdbscan' / 'hdbscan_chr' - 基于 HDBSCAN 的聚类
                      'hdbscan_ldblock' - 基于 LD block 的 HDBSCAN 聚类
        n_clusters (int): 聚类数量（用于 'kmeans'、'agg'、'position_cluster'）
        window_size (int): 窗口大小（用于 'by_chr_window'）
        n_clusters_per_chr (int): 每条染色体内聚类数量（用于 'position_cluster_chr'）
        eps (float): DBSCAN 的 eps 参数
        min_samples (int): DBSCAN / HDBSCAN 的 min_samples 参数
        min_cluster_size (int): HDBSCAN 的 min_cluster_size 参数

    Returns:
        Tuple[List[int], List[int], Dict[int,int]]:
            - feature_class: 每个特征所属的整数编号类（从0开始）
            - class_counts: 每一类中的特征数目
            - label_to_id: 原始标签 → 连续整数的映射
    """
    columns = features.columns

    # ✅ 预处理：修复列名中位置字段错误
    def fix_column_name(col):
        if isinstance(col, int):
            return col, False
        parts = col.split('_')
        if len(parts) < 2:
            return col, False

        chr_, pos = parts[0], parts[1]
        # 修复 '5.5e.07' → '5.5e7'
        fixed_pos = re.sub(r'e\.?0*(\d+)', r'e\1', pos, flags=re.IGNORECASE)

        try:
            float(fixed_pos)
        except:
            return col, False  # 修复失败

        new_parts = [chr_, fixed_pos] + parts[2:]
        return '_'.join(new_parts), (fixed_pos != pos)

    fixed_columns = []
    fix_count = 0
    for col in columns:
        fixed_col, fixed = fix_column_name(col)
        fixed_columns.append(fixed_col)
        if fixed:
            print(f" 修复列名：{col} → {fixed_col}")
            fix_count += 1
    if fix_count > 0:
        print(f" 共修复 {fix_count} 个列名错误")
        features.columns = pd.Index(fixed_columns)  # 覆盖列名

    columns = features.columns  # 重新取列名

    raw_class = []

    if method == 'by_chr':
        raw_class = [col.split('_')[0] for col in columns]

    elif method == 'by_chr_window':
        def parse_chr_pos(col):
            chr_, pos_str = col.split('_')[:2]
            pos = int(float(pos_str))  # 更鲁棒
            return chr_, pos
        chr_pos = [parse_chr_pos(col) for col in columns]
        raw_class = [f"{chr_}_{pos // window_size}" for chr_, pos in chr_pos]

    elif method == 'position_cluster':
        def extract_pos(col):
            return int(float(col.split('_')[1]))
        positions = np.array([extract_pos(col) for col in columns]).reshape(-1, 1)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        raw_class = model.fit_predict(positions)

    elif method == 'position_cluster_chr':
        raw_class = position_cluster_with_chr(columns, n_clusters_per_chr=n_clusters_per_chr)

    elif method == 'kmeans':
        X = features.T.values
        model = KMeans(n_clusters=n_clusters, random_state=42)
        raw_class = model.fit_predict(X)

    elif method == 'agg':
        X = features.T.values
        model = AgglomerativeClustering(n_clusters=n_clusters)
        raw_class = model.fit_predict(X)

    elif method == 'hierarchical_chr':
        raw_class = hierarchical_cluster_by_chr(features, max_clusters=n_clusters)

    elif method == 'dbscan_chr':
        raw_class = cluster_by_chr_with_dbscan(features, eps=eps, min_samples=min_samples)

    elif method == 'dbscan':
        raw_class = cluster_with_dbscan(features, eps=eps, min_samples=min_samples)

    elif method == 'hdbscan':
        raw_class = cluster_with_hdbscan(features, min_cluster_size=min_cluster_size, min_samples=min_samples)

    elif method == 'hdbscan_chr':
        raw_class = cluster_by_chr_with_hdbscan(features, min_cluster_size=min_cluster_size, min_samples=min_samples)

    elif method == 'hdbscan_ldblock':
        raw_class = cluster_with_hdbscan_ldblock(features, threshold=ld_threshold,
                                                min_cluster_size=min_cluster_size,
                                                min_samples=min_samples)


    else:
        raise ValueError(f"未知聚类方式：{method}")

    # 将 raw_class 映射为整数编号，保留原始出现顺序
    if isinstance(raw_class[0], (str, object)):
        unique_labels = list(dict.fromkeys(raw_class))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        feature_class = [label_to_id[label] for label in raw_class]
    else:
        feature_class = list(raw_class)
        label_to_id = {int(i): int(i) for i in set(feature_class)}  # 数值标签无需转换

    # 统计类别数量
    counter = Counter(feature_class)
    class_ids = sorted(counter.keys())
    class_counts = [counter[k] for k in class_ids]
    n_classes = len(class_ids)

    print(f" 聚类方式：{method}，共 {n_classes} 类")

    feature_class_series = pd.Series(feature_class)
    counts = feature_class_series.value_counts().sort_index()
    # 过滤掉只有 1 个特征的类
    counts = counts[counts > 1]
    plt.figure(figsize=(12, 6))
    ax = counts.plot(kind="bar", width=0.8, color='#4c72b0')
    
    n_clusters = len(counts)
    plt.title(f"Feature distribution across {n_clusters} clusters (size > 1)", fontsize=14)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.ylabel("Number of features", fontsize=12)
    
    # 智能调整 X 轴标签
    if n_clusters > 50:
        # 标签过多时，每隔一定间距显示一个标签，防止重叠
        step = max(1, int(n_clusters / 50))
        ticks = ax.get_xticks()
        tick_labels = counts.index
        plt.xticks(ticks[::step], tick_labels[::step], rotation=90, fontsize=8)
    else:
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


    return feature_class, class_counts, label_to_id

