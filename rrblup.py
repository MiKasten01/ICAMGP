# 纯 Python 版本的 rrBLUP（mixed_solve）实现 + 一键式 GRM 计算（A_mat）

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def crossprod(A, B):
    return np.dot(A.T, B)

def tcrossprod(A, B):
    return np.dot(A, B.T)

def A_mat(X, min_MAF=None, max_missing=None, impute_method="mean"):
    """
    计算基于 VanRaden 方法的基因组关系矩阵（GRM）

    X: (n, m) 基因型矩阵，元素为 {-1, 0, 1}
    return: (n, n) GRM 矩阵 A
    """
    n, m = X.shape
    tmp = X + 1  # 编码为 {0,1,2}
    freq = np.nanmean(tmp, axis=0) / 2
    MAF = np.minimum(freq, 1 - freq)

    if min_MAF is None:
        min_MAF = 1 / (2 * n)
    if max_missing is None:
        max_missing = 1 - 1 / (2 * n)

    frac_missing = np.sum(np.isnan(X), axis=0) / n
    keep = np.where((MAF >= min_MAF) & (frac_missing <= max_missing))[0]

    freq = freq[keep]
    X = X[:, keep]
    freq_mat = np.dot(np.ones((n, 1)), freq.reshape(1, -1))
    W = X + 1 - 2 * freq_mat
    W[np.isnan(W)] = 0

    var_A = 2 * np.mean(freq * (1 - freq))
    A = tcrossprod(W, W) / var_A / len(keep)
    return A


def mixed_solve_py(y, Z=None, K=None, X=None, method="REML", bounds=(1e-9, 1e9), return_Hinv=False):
    """
    纯 Python 实现的混合线性模型求解器（BLUP/REML）

    Parameters:
        y: (n, 1) 表型
        Z: (n, m) 随机效应设计矩阵（如基因型）
        K: (m, m) 协方差矩阵（如 G 矩阵），默认单位矩阵
        X: (n, p) 固定效应设计矩阵（默认全1）
        method: "ML" 或 "REML"
        bounds: lambda 搜索范围
        return_Hinv: 是否返回 H 的逆

    Returns:
        字典，含 Vu, Ve, beta, u, LL（可选 Hinv）
    """
    n = len(y)
    y = y.reshape(-1, 1)

    if X is None:
        X = np.ones((n, 1))
    if Z is None:
        Z = np.eye(n)
    if K is None:
        K = np.eye(Z.shape[1])

    XtX = crossprod(X, X)
    XtX_inv = np.linalg.inv(XtX)
    P = np.eye(n) - X @ XtX_inv @ X.T

    ZKZt = Z @ K @ Z.T
    Py = P @ y

    def log_likelihood_REML(log_lambda):
        lam = np.exp(log_lambda)
        H = ZKZt + lam * np.eye(n)
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return np.inf
        Linv_Py = np.linalg.solve(L, Py)
        ll = (n - X.shape[1]) * np.log((Linv_Py**2).sum()) + 2 * np.sum(np.log(np.diag(L)))
        return ll

    log_bounds = (np.log(bounds[0]), np.log(bounds[1]))
    res = minimize_scalar(log_likelihood_REML, bounds=log_bounds, method='bounded')
    lambda_opt = np.exp(res.x)

    H = ZKZt + lambda_opt * np.eye(n)
    Hinv = np.linalg.inv(H)
    beta = np.linalg.solve(X.T @ Hinv @ X, X.T @ Hinv @ y)
    u = K @ Z.T @ Hinv @ (y - X @ beta)
    Vu = float(((y - X @ beta).T @ Hinv @ Z @ u) / Z.shape[1])
    Ve = Vu * lambda_opt
    LL = -0.5 * res.fun

    result = {
        'Vu': Vu,
        'Ve': Ve,
        'beta': beta,
        'u': u,
        'LL': LL
    }
    if return_Hinv:
        result['Hinv'] = Hinv

    return result


def fit_rrblup(y, genotype, use_grm=True):
    """
    rrBLUP 拟合 + 特征效应输出

    Parameters:
        y: (n,) 表型向量
        genotype: (n, m) 编码为 {-1,0,1} 的 SNP 矩阵
        use_grm: 是否先计算 GRM（如设为 False，则直接做 rrBLUP）

    Returns:
        dict: {y_hat, u, beta, Vu, Ve, A, feature_importance}
    """
    n = len(y)
    y = y.reshape(-1, 1)
    X = np.ones((n, 1))

    if use_grm:
        A = A_mat(genotype)
        result = mixed_solve_py(y, Z=np.eye(n), K=A, X=X)
    else:
        result = mixed_solve_py(y, Z=genotype, X=X)
        A = None

    y_hat = X @ result['beta'] + (genotype @ result['u'] if not use_grm else result['u'])
    importance = np.abs(result['u']).flatten()

    return {
        'y_hat': y_hat.flatten(),
        'u': result['u'].flatten(),
        'beta': result['beta'].flatten(),
        'Vu': result['Vu'],
        'Ve': result['Ve'],
        'LL': result['LL'],
        'A': A,
        'feature_importance': importance
    }
