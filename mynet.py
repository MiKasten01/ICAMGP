"""
mynet_clean.py — GitHub 发布特供版
=====================================
解耦版本：仅保留 MLP + CAM 系列模型（CAM / woGlobal / woLocal / woCluster），
移除了对 DeepGS、ResGS、CropFormer、DNNGP、Deep5mC、Mask_MLP、In_Dropout 等外部模型的依赖。

依赖项:
    - torch, sklearn, numpy, pandas, scipy（标准库）
    - cam（ClusterSparseAttentionModel — 项目内置）
    - rrblup（rrBLUP 混合线性模型 — 项目内置）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

from rrblup import fit_rrblup
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import time
import cam
from enum import Enum


# ============================= Torch Model Enum =============================

class TorchModelType(Enum):
    """支持的 PyTorch 模型类型（GitHub 发布版 — 仅 CAM 系列）"""
    MLP       = ("MLP",       "mse")
    CAM       = ("CAM",       "mse")
    woGlobal  = ("woGlobal",  "mse")
    woLocal   = ("woLocal",   "mse")
    woCluster = ("woCluster", "mse")

    def __init__(self, model_name: str, loss_type: str):
        self._model_name = model_name
        self._loss_type = loss_type

    @property
    def model_name(self):
        return self._model_name

    @property
    def loss_type(self):
        return self._loss_type

    def __str__(self):
        return self._model_name


# ============================= Metric Functions =============================

def pearson_corr(y_true, y_pred):
    """计算皮尔逊相关系数（返回绝对值，越大越好）"""
    return abs(pearsonr(y_true.flatten(), y_pred.flatten())[0])


def r2_score_multi(y_true, y_pred):
    """计算多维数据的 R² 分数（按列计算 R²，然后取均值）"""
    scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return np.mean(scores)


def pearson_corr_multi(y_true, y_pred):
    """计算多维数据的皮尔逊相关系数（按列计算 Pearson，然后取均值）"""
    scores = [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
    return np.mean(np.abs(scores))


def mse_per_dim(y_true, y_pred):
    if y_true.ndim == 1:
        return mean_squared_error(y_true, y_pred)
    else:
        D = y_true.shape[1]
        return [mean_squared_error(y_true[:, i], y_pred[:, i]) for i in range(D)]


def mae_per_dim(y_true, y_pred):
    if y_true.ndim == 1:
        return mean_absolute_error(y_true, y_pred)
    else:
        D = y_true.shape[1]
        return [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(D)]


def r2_per_dim(y_true, y_pred):
    if y_true.ndim == 1:
        return r2_score(y_true, y_pred)
    else:
        D = y_true.shape[1]
        return [r2_score(y_true[:, i], y_pred[:, i]) for i in range(D)]


def pearson_per_dim(y_true, y_pred):
    if y_true.ndim == 1:
        return pearsonr(y_true, y_pred)[0]
    else:
        D = y_true.shape[1]
        return [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(D)]


# ============================= MyNet Class =============================

class MyNet:
    """
    训练控制器：支持 PyTorch 深度学习、传统机器学习 (sklearn)、rrBLUP 三种模式。

    本版本仅包含 CAM 系列模型（CAM / woGlobal / woLocal / woCluster）及
    基线 MLP，移除了与其他模型（DeepGS 等）的耦合。
    """

    def __init__(self, dataset, ml_models=None, torch_selection=TorchModelType.MLP,
                 feature_class=None, feature_mask=None, k1_fold=5, model_params=None):
        """
        Args:
            dataset: MyDataset 实例
            ml_models: sklearn 模型列表（用于 tradition_fit）
            torch_selection: PyTorch 模型类型（TorchModelType 枚举）
            feature_class: 每个特征的聚类标签列表
            feature_mask: 特征初始权重（保留接口，本版未使用）
            k1_fold: K-Fold 折数
            model_params: 模型超参数字典，如 {'hidden_dim': 32, 'num_heads': 8, 'dropout': 0.1}
        """
        self.dataset = dataset
        self.ml_models = ml_models if ml_models else [
            KernelRidge(), SVR(), Ridge(),
            RandomForestRegressor(n_jobs=-1), GradientBoostingRegressor()
        ]

        self.torch_selection = torch_selection
        self.loss_type = torch_selection.loss_type

        self.feature_class = feature_class
        self.k1_fold = k1_fold
        self.model_params = model_params if model_params is not None else {}

        # ---- 模型构建函数路由 ----
        _builder_map = {
            TorchModelType.MLP:       self._build_mlp,
            TorchModelType.CAM:       self._build_CAM,
            TorchModelType.woGlobal:  self._build_woGlobal,
            TorchModelType.woLocal:   self._build_woLocal,
            TorchModelType.woCluster: self._build_woCluster,
        }
        if torch_selection not in _builder_map:
            raise ValueError(f"不支持的模型类型: {torch_selection}")
        self._build_model = _builder_map[torch_selection]

        self.best_ml_models = []
        self.residual_models = []
        self.torch_models = []

    # ========================= 模型构建 =========================

    def _build_mlp(self):
        """基线 MLP"""
        input_dim = self.dataset.features.shape[1]
        output_dim = self.dataset.labels.shape[1]
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def _build_CAM(self):
        """ClusterSparseAttentionModel（完整版：局部 + 全局注意力）"""
        output_dim = self.dataset.labels.shape[1]
        return cam.ClusterSparseAttentionModel(
            feature_class=self.feature_class,
            hidden_dim=self.model_params.get('hidden_dim', 32),
            num_heads=self.model_params.get('num_heads', 8),
            dropout=self.model_params.get('dropout', 0.1),
            output_dim=output_dim,
            max_cluster=self.model_params.get('max_cluster', 100),
        )

    def _build_woLocal(self):
        """消融实验：移除簇内注意力 (woLocal)"""
        output_dim = self.dataset.labels.shape[1]
        return cam.ClusterSparseAttentionModel(
            feature_class=self.feature_class,
            hidden_dim=self.model_params.get('hidden_dim', 32),
            num_heads=self.model_params.get('num_heads', 8),
            dropout=self.model_params.get('dropout', 0.1),
            output_dim=output_dim,
            ablation_type='woLocal',
            max_cluster=self.model_params.get('max_cluster', 100),
        )

    def _build_woGlobal(self):
        """消融实验：移除簇间注意力 (woGlobal)"""
        output_dim = self.dataset.labels.shape[1]
        return cam.ClusterSparseAttentionModel(
            feature_class=self.feature_class,
            hidden_dim=self.model_params.get('hidden_dim', 32),
            num_heads=self.model_params.get('num_heads', 8),
            dropout=self.model_params.get('dropout', 0.1),
            output_dim=output_dim,
            ablation_type='woGlobal',
            max_cluster=self.model_params.get('max_cluster', 100),
        )

    def _build_woCluster(self):
        """消融实验：所有特征视为单一簇 (woCluster)"""
        output_dim = self.dataset.labels.shape[1]
        return cam.ClusterSparseAttentionModel(
            feature_class=self.feature_class,
            hidden_dim=self.model_params.get('hidden_dim', 32),
            num_heads=self.model_params.get('num_heads', 4),
            dropout=self.model_params.get('dropout', 0.1),
            output_dim=output_dim,
            ablation_type='woCluster',
            max_cluster=10000,
        )

    # ========================= 传统机器学习训练 =========================

    def tradition_fit(self, param_grids=None, metric_funcs=None, n_jobs=-1):
        """GridSearchCV + K-Fold 交叉验证（传统 ML）"""
        X, y = self.dataset.to_numpy()

        print(f"\n使用 GridSearchCV 进行 {self.k1_fold}-折交叉验证 (固定参数每折一致)")

        if metric_funcs is None:
            metric_funcs = {
                "MSE": mse_per_dim,
                "MAE": mae_per_dim,
                "R2": r2_per_dim,
                "Pearson": pearson_per_dim
            }

        if param_grids is None:
            param_grids = {
                'Lasso': {
                    'model': Lasso(),
                    'param_grid': {'alpha': [1e-03, 1e-02, 0.1, 1, 10, 100]}
                },
                'ElasticNet': {
                    'model': ElasticNet(),
                    'param_grid': {'alpha': [1e-03, 1e-02, 0.1, 1, 10, 100],
                                   'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
                },
                'KRR_rbf': {
                    'model': KernelRidge(kernel='rbf'),
                    'param_grid': {'alpha': [1e-03, 1e-02, 0.1, 1, 10, 100]}
                },
                'KRR_cos': {
                    'model': KernelRidge(kernel='cosine'),
                    'param_grid': {'alpha': [1e-03, 1e-02, 0.1, 1, 10, 100]}
                },
                'KRR_sig': {
                    'model': KernelRidge(kernel='sigmoid'),
                    'param_grid': {'alpha': [1e-03, 1e-02, 0.1, 1, 10, 100]}
                },
                'SVR_rbf': {
                    'model': SVR(kernel='rbf'),
                    'param_grid': {'C': [0.1, 1, 10, 100]}
                },
                'AdaBoostRegressor': {
                    'model': AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), random_state=42),
                    'param_grid': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
                },
                'RandomForest': {
                    'model': RandomForestRegressor(),
                    'param_grid': {'n_estimators': [50, 100, 200],
                                   'max_depth': [10, 20]}
                }
            }

        kf = KFold(n_splits=self.k1_fold, shuffle=True, random_state=42)
        all_pred, all_metrics_records = [], []
        all_metrics = {name: [] for name in metric_funcs.keys()}

        for model_name, cfg in param_grids.items():
            print(f"\n==============================\n模型: {model_name}")
            model = cfg["model"]
            grid = cfg["param_grid"]
            scorer = make_scorer(pearson_corr, greater_is_better=True)

            t0 = time.time()

            grid_search = GridSearchCV(model, grid, cv=3, scoring=scorer,
                                       n_jobs=n_jobs, refit=True, verbose=0)
            grid_search.fit(X, y.ravel())

            best_params = grid_search.best_params_
            print(f"🔍 全局最优参数: {best_params}")

            best_model_class = model.__class__
            try:
                best_model = best_model_class(**best_params)
            except TypeError:
                cleaned = {k.replace('model__', ''): v for k, v in best_params.items()}
                best_model = best_model_class(**cleaned)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                m = best_model.__class__(**best_model.get_params())
                m.fit(X_train, y_train.ravel())
                y_pred_final = m.predict(X_valid)
                if y_pred_final.ndim == 1:
                    y_pred_final = y_pred_final.reshape(-1, 1)
                y_valid = y_valid.reshape(-1, 1)

                for i in range(len(y_valid)):
                    for j in range(y.shape[1] if y.ndim > 1 else 1):
                        all_pred.append({
                            "Fold": fold_idx + 1,
                            f"True_{model_name}": float(y_valid[i, j] if y.ndim > 1 else y_valid[i]),
                            f"Pred_{model_name}": float(y_pred_final[i, j] if y_pred_final.ndim > 1 else y_pred_final[i])
                        })

                for metric_name, metric_func in metric_funcs.items():
                    score = metric_func(y_valid, y_pred_final)
                    if isinstance(score, (list, np.ndarray)):
                        score = float(np.mean(score))
                    all_metrics[metric_name].append(score)
                    all_metrics_records.append({
                        "metric_name": metric_name,
                        "fold": fold_idx + 1,
                        f"value_{model_name}": score
                    })

                print(f" 第 {fold_idx + 1}/{self.k1_fold} 折 评价指标：")
                for metric_name, scores in all_metrics.items():
                    print(f"    {metric_name}: {scores[-1]}")

            elapsed_time = round(time.time() - t0, 2)

            print("\n交叉验证平均评价指标：")
            for metric_name, scores_per_fold in all_metrics.items():
                scores_array = np.array(scores_per_fold)
                mean_score = round(float(np.mean(scores_array)), 4)
                std_score = round(float(np.std(scores_array, ddof=1)), 4)
                print(f"  {metric_name}: {mean_score} ± {std_score}")

                all_metrics_records.append({
                    "metric_name": metric_name,
                    "fold": "all",
                    f"mean_{model_name}": mean_score,
                    f"std_{model_name}": std_score
                })

            print(f"  ⏱ Time: {elapsed_time} s")
            all_metrics_records.append({
                "metric_name": "Time",
                "fold": "all",
                f"mean_{model_name}": elapsed_time,
                f"std_{model_name}": 0.0
            })

        print("\n✅ 全部传统模型训练与交叉验证完成")

        df_pred = pd.DataFrame(all_pred)
        df_metrics = pd.DataFrame(all_metrics_records)

        df_pred = (
            df_pred.groupby("Fold")
            .apply(lambda g: pd.concat([g.drop(columns=["Fold"]).reset_index(drop=True)], axis=1))
            .reset_index(drop=True)
        )
        df_pred.insert(0, "Fold", sorted(list(range(1, df_pred.shape[0] + 1))))
        df_pred = df_pred.sort_values(by=["Fold"]).reset_index(drop=True)

        df_metrics = (
            df_metrics.pivot_table(
                index=["metric_name", "fold"],
                aggfunc="first"
            )
            .reset_index()
            .sort_values(by=["metric_name", "fold"])
            .reset_index(drop=True)
        )

        print(f"\n📄 预测结果表: {df_pred.shape}")
        print(f"📄 指标结果表: {df_metrics.shape}")

        return df_pred, df_metrics

    # ========================= PyTorch 训练 =========================

    def torch_fit(self, train_data, valid_data, model, batch_size=32, lr=0.001, epochs=1000,
                  early_stopping_patience=5, verbose=1):
        """
        PyTorch 训练（MSE 损失 + Early Stopping）。

        Args:
            train_data: (X_train, y_train) numpy 数组元组
            valid_data: (X_valid, y_valid) numpy 数组元组
            model: torch.nn.Module
            batch_size: 批大小
            lr: 学习率
            epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            verbose: 1=打印每轮信息, 0=静默
        Returns:
            (best_model, best_valid_loss)
        """
        print("\n 开始 PyTorch 训练...")
        X_train, y_train_res = train_data
        X_valid, y_valid_res = valid_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train_res, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(
            TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                          torch.tensor(y_valid_res, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False)

        # 本版仅使用 MSE 损失
        def compute_loss(outputs, targets):
            return F.mse_loss(outputs, targets)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_valid_loss = float("inf")
        best_model = None
        patience_counter = 0

        for epoch in range(epochs):
            # ---- train ----
            model.train()
            total_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = compute_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # ---- validation ----
            model.eval()
            total_valid_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in valid_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = compute_loss(outputs, batch_y)
                    total_valid_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_valid_loss = total_valid_loss / len(valid_loader)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f" 早停触发！Epoch {epoch + 1}, Best Valid Loss: {best_valid_loss:.6f}")
                    break

        print(f"\n PyTorch 训练完成！Best Valid Loss: {best_valid_loss:.6f}\n")
        return best_model, best_valid_loss

    # ========================= K-Fold CV =========================

    def torch_cv_fit(self, metric_funcs=None, num_restarts=1, early_stopping_patience=5):
        """
        K-Fold 交叉验证（纯 PyTorch）。每次 fold 内支持多次 restart 取最优。
        """
        print(f"\n开始 {self.k1_fold} 折交叉验证（仅使用 PyTorch 模型）...")
        if metric_funcs is None:
            metric_funcs = {
                "MSE": mse_per_dim,
                "MAE": mae_per_dim,
                "R2": r2_per_dim,
                "Pearson": pearson_per_dim
            }

        X, y = self.dataset.to_numpy()
        kf = KFold(n_splits=self.k1_fold, shuffle=True, random_state=42)
        all_metrics = {metric: [] for metric in metric_funcs}
        all_pred = []
        all_metrics_records = []

        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
            print(f"\n===== 处理 {fold_idx + 1}/{self.k1_fold} 折 =====")

            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            torch.cuda.empty_cache()
            best_res_model = None
            best_valid_loss = float("inf")
            valid_restarts = 0
            max_attempts = num_restarts * 3
            attempt = 0

            while valid_restarts < num_restarts and attempt < max_attempts:
                print(f"\n训练第 {valid_restarts + 1}/{num_restarts} 次模型（尝试 {attempt + 1}/{max_attempts}）")

                res_model = self._build_model()

                trained_res_model, last_valid_loss = self.torch_fit(
                    (X_train, y_train),
                    (X_valid, y_valid),
                    model=res_model,
                    early_stopping_patience=early_stopping_patience
                )

                if not np.isfinite(last_valid_loss):
                    print("⚠️ last_valid_loss 为 NaN，舍弃该模型，重新训练")
                    attempt += 1
                    continue

                if last_valid_loss < best_valid_loss:
                    best_valid_loss = last_valid_loss
                    best_res_model = trained_res_model

                valid_restarts += 1
                attempt += 1

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_res_model.to(device)
            best_res_model.eval()

            # ---- 分批预测 ----
            def predict_in_batches(model, X_data, batch_size=64, device="cuda"):
                model.eval()
                preds = []
                with torch.no_grad():
                    for i in range(0, len(X_data), batch_size):
                        batch = torch.tensor(X_data[i:i + batch_size], dtype=torch.float32, device=device)
                        pred = model(batch).cpu().numpy()
                        preds.append(pred)
                        del batch
                    torch.cuda.empty_cache()
                return np.concatenate(preds, axis=0)

            y_pred_final = predict_in_batches(best_res_model, X_valid, batch_size=32, device=device)

            # 保存预测结果
            for i in range(len(y_valid)):
                for j in range(y.shape[1]):
                    all_pred.append({
                        "Fold": fold_idx + 1,
                        "True": y_valid[i, j],
                        "Pred": y_pred_final[i, j]
                    })

            for metric_name, metric_func in metric_funcs.items():
                score = metric_func(y_valid, y_pred_final)
                all_metrics[metric_name].append(score)
                all_metrics_records.append({
                    "metric_name": metric_name,
                    "fold": fold_idx + 1,
                    "value": score
                })

            print(f" 第 {fold_idx + 1}/{self.k1_fold} 折 评价指标：")
            for metric_name, scores in all_metrics.items():
                print(f"    {metric_name}: {scores[-1]}")

            self.torch_models.append(best_res_model)

        print("\n交叉验证平均评价指标：")
        for metric_name, scores_per_fold in all_metrics.items():
            scores_array = np.array(scores_per_fold)
            mean_score = round(float(np.mean(scores_array)), 4)
            std_score = round(float(np.std(scores_array, ddof=1)), 4)
            print(f"  {metric_name}: {mean_score} ± {std_score}")

            all_metrics_records.append({
                "metric_name": metric_name,
                "fold": "all",
                "mean": mean_score,
                "std": std_score
            })

        print("\n所有 Fold 训练完成！")

        df = pd.DataFrame(all_pred)
        metrics_df = pd.DataFrame(all_metrics_records)
        return df, metrics_df

    # ========================= rrBLUP =========================

    def blup_fit(self, metric_funcs=None):
        """rrBLUP (Ridge-Regression BLUP) K-Fold 交叉验证"""
        if metric_funcs is None:
            metric_funcs = {
                "MSE": mean_squared_error,
                "MAE": mean_absolute_error,
                "R2": r2_score,
                "Pearson": pearson_corr
            }

        X, y = self.dataset.to_numpy()
        kf = KFold(n_splits=self.k1_fold, shuffle=True, random_state=42)

        all_metrics = {metric: [] for metric in metric_funcs}
        all_pred = []
        all_metrics_records = []

        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
            print(f"\n===== 处理 {fold_idx + 1}/{self.k1_fold} 折 =====")

            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            result = fit_rrblup(y_train, X_train, use_grm=False)
            y_pred = X_valid @ result['u'] + result['beta'][0]

            for i in range(len(y_valid)):
                all_pred.append({
                    "Fold": fold_idx + 1,
                    "True": y_valid[i],
                    "Pred": y_pred[i]
                })

            for metric_name, metric_func in metric_funcs.items():
                score = metric_func(y_valid, y_pred)
                all_metrics[metric_name].append(score)
                all_metrics_records.append({
                    "metric_name": metric_name,
                    "fold": fold_idx + 1,
                    "value": score
                })

            print(f" 第 {fold_idx + 1}/{self.k1_fold} 折 评价指标：")
            for metric_name, scores in all_metrics.items():
                print(f"    {metric_name}: {scores[-1]}")

        print("\n交叉验证平均评价指标：")
        for metric_name, scores_per_fold in all_metrics.items():
            scores_array = np.array(scores_per_fold)
            mean_score = round(float(np.mean(scores_array)), 4)
            std_score = round(float(np.std(scores_array, ddof=1)), 4)
            print(f"  {metric_name}: {mean_score} ± {std_score}")

            all_metrics_records.append({
                "metric_name": metric_name,
                "fold": "all",
                "mean": mean_score,
                "std": std_score
            })

        print("\n所有 Fold 训练完成！")
        df = pd.DataFrame(all_pred)
        metrics_df = pd.DataFrame(all_metrics_records)
        return df, metrics_df
