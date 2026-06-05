import os
print("Starting pipeline.py execution...")
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from dataset.new_dataset import load_dataset, MyDataset
from feature_cluster import cluster
from feature_select import feature_selector
from mynet import MyNet, TorchModelType
import warnings


def run_pipeline(
    # 1. Dataset Parameters
    dataset_name: str,
    config_path: str = 'dataset/dataset_config.json',
    
    # 2. Clustering Parameters
    cluster_method: str = 'hdbscan_ldblock',
    cluster_params: Optional[Dict[str, Any]] = None,
    
    # 3. Feature Selection Parameters
    enable_feature_selection: bool = True,
    selection_params: Optional[Dict[str, Any]] = None,
    
    # 4. Model Parameters
    model_type: str = 'CAM',
    model_params: Optional[Dict[str, Any]] = None,
    training_params: Optional[Dict[str, Any]] = None,
    fit_method: str = 'torch_cv_fit',
    
    # 5. General Parameters
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    全链路模型评估函数
    
    Args:
        dataset_name: 数据集名称 (config key)
        config_path: 配置文件路径
        cluster_method: 聚类方法 ('hdbscan_ldblock', 'kmeans', etc.)
        cluster_params: 聚类参数字典 (e.g. {'ld_threshold': 0.2, 'min_cluster_size': 5})
        enable_feature_selection: 是否启用特征选择
        selection_params: 特征选择参数字典 (e.g. {'max_snp': 10000, 'p_threshold': 0.05})
        model_type: 模型类型 ('CAM', 'MLP', 'ResGS', etc.)
        model_params: 模型超参数字典 (e.g. {'hidden_dim': 64, 'dropout': 0.2})
        training_params: 训练参数字典 (e.g. {'k1_fold': 5, 'num_restarts': 1})
        fit_method: 训练方法 ('torch_cv_fit', 'blup_fit', 'tradition_fit')
        random_seed: 随机种子
        
    Returns:
        Dict containing:
        - dataset: Loaded MyDataset object (potentially modified)
        - cluster_results: (feature_class, class_counts, label_to_id)
        - selection_results: (selected_features, selected_feature_class) or None
        - predictions: DataFrame of predictions
        - metrics: DataFrame of metrics
    """
    warnings.filterwarnings("ignore")
    
    # Set defaults
    if cluster_params is None:
        cluster_params = {}
    if selection_params is None:
        selection_params = {}
    if model_params is None:
        model_params = {}
    if training_params is None:
        training_params = {}
        
    # Set random seed
    np.random.seed(random_seed)
    try:
        import torch
        torch.manual_seed(random_seed)
    except ImportError:
        pass
        
    # 1. Load Dataset
    print(f"\n[Pipeline] Step 1: Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, config_path=config_path)
    print(f"[Pipeline] Dataset loaded. Shape: features={dataset.features.shape}, labels={dataset.labels.shape}")
    
    # 2. Clustering
    models_needing_clustering = ['CAM', 'woGlobal', 'woLocal', 'woCluster']
    
    if model_type in models_needing_clustering:
        print(f"\n[Pipeline] Step 2: Running clustering with method '{cluster_method}'...")
        # Ensure dataset.features is DataFrame for clustering
        if not isinstance(dataset.features, pd.DataFrame):
            raise TypeError("Clustering requires dataset.features to be a pandas DataFrame")
            
        feature_class, class_counts, label_to_id = cluster(
            dataset.features, 
            method=cluster_method, 
            **cluster_params
        )
        print(f"[Pipeline] Clustering complete. Found {len(class_counts)} clusters.")
    else:
        print(f"\n[Pipeline] Step 2: Skipping clustering for model '{model_type}' (feature_class fixed to 1).")
        num_features = dataset.features.shape[1]
        feature_class = [1] * num_features
        class_counts = {1: num_features}
        label_to_id = {1: 0}
    
    # 3. Feature Selection
    selection_results = None
    if enable_feature_selection:
        print(f"\n[Pipeline] Step 3: Running feature selection...")
        selected_features, selected_feature_class = feature_selector(
            dataset, 
            feature_class=feature_class, 
            **selection_params
        )
        
        # Update dataset with selected features
        dataset.features = selected_features
        # Update feature_class for the model
        feature_class = selected_feature_class
        
        selection_results = {
            'selected_features': selected_features,
            'selected_feature_class': selected_feature_class
        }
        print(f"[Pipeline] Feature selection complete. New features shape: {dataset.features.shape}")
    else:
        print("\n[Pipeline] Step 3: Feature selection skipped.")
        
    # 4. Model Training
    print(f"\n[Pipeline] Step 4: Model Training with method '{fit_method}'...")
    
    # Extract training specific params
    k1_fold = training_params.get('k1_fold', 5)
    
    if fit_method == 'torch_cv_fit':
        print(f"[Pipeline] Training Torch model: '{model_type}'...")
        
        # Map string to TorchModelType
        torch_model_type = None
        try:
            # Try direct attribute access (e.g. TorchModelType.CAM)
            torch_model_type = getattr(TorchModelType, model_type)
        except AttributeError:
            pass
            
        if torch_model_type is None:
            # Try matching by name or value
            for t in TorchModelType:
                if t.name == model_type or t.value[0] == model_type:
                    torch_model_type = t
                    break
        
        if torch_model_type is None:
            raise ValueError(f"Unknown model type: {model_type}")
            
        num_restarts = training_params.get('num_restarts', 1)
        early_stopping_patience = training_params.get('early_stopping_patience', 5)
        
        # Initialize MyNet
        net = MyNet(
            dataset=dataset,
            torch_selection=torch_model_type,
            feature_class=feature_class,
            k1_fold=k1_fold,
            model_params=model_params
        )
        
        df_pred, df_metrics = net.torch_cv_fit(
            num_restarts=num_restarts,
            early_stopping_patience=early_stopping_patience
        )
        
    elif fit_method == 'blup_fit':
        print("[Pipeline] Training BLUP model...")
        # Initialize MyNet (torch_selection defaults to MLP, ignored for BLUP)
        net = MyNet(
            dataset=dataset,
            feature_class=feature_class,
            k1_fold=k1_fold,
            model_params=model_params
        )
        df_pred, df_metrics = net.blup_fit()
        
    elif fit_method == 'tradition_fit':
        print("[Pipeline] Training Traditional ML models...")
        # Initialize MyNet (torch_selection defaults to MLP, ignored for tradition_fit)
        net = MyNet(
            dataset=dataset,
            feature_class=feature_class,
            k1_fold=k1_fold,
            model_params=model_params
        )
        n_jobs = training_params.get('n_jobs', -1)
        df_pred, df_metrics = net.tradition_fit(n_jobs=n_jobs)
        
    else:
        raise ValueError(f"Unknown fit_method: {fit_method}. Options: 'torch_cv_fit', 'blup_fit', 'tradition_fit'")
    
    print("\n[Pipeline] Pipeline complete.")
    
    return {
        'dataset': dataset,
        'cluster_results': {
            'feature_class': feature_class,
            'class_counts': class_counts,
            'label_to_id': label_to_id
        },
        'selection_results': selection_results,
        'predictions': df_pred,
        'metrics': df_metrics
    }

if __name__ == "__main__":
    # Example usage
    result = run_pipeline(
        dataset_name='CLMA',
        config_path='dataset/dataset_config.json',
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
    
    print("Predictions:")
    print(result['predictions'].head())
    
    print("\nMetrics:")
    print(result['metrics'])