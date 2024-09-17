import numpy as np
import scipy as sp
import sklearn as sk
import pandas as pd
import anndata as ad
import scanpy as sc
import os

def load_adata(path, log_transform = True):
    adata = ad.AnnData(pd.read_csv(os.path.join(path, "ExpressionData.csv"), index_col = 0).T)
    df_pt = pd.read_csv(os.path.join(path, "PseudoTime.csv"), index_col = 0)
    df_pt[np.isnan(df_pt)] = 0
    adata.obs["t_sim"] = np.max(df_pt.to_numpy(), -1)
    if log_transform:
        sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, min_dist = 0.9)
    return adata

def simulate(A, sigma, N, ts, ic_func, dt = 1e-2):
    T = len(ts)
    d = A.shape[1]
    xs = []
    for i in range(T):
        # x = np.random.randn(N, d)*0.05 + x0 if x_init is None else x_init
        x = ic_func(N, d)
        t = 0
        while t < ts[i]:
            x += x @ A * dt + np.random.randn(N, d) @ sigma * dt**0.5
            t += dt
        xs.append(x)
    return np.stack(xs)

def load_boolODE_reference_network(path, adata):
    df = pd.read_csv(path)
    n_genes = adata.shape[1]
    A_ref = pd.DataFrame(np.zeros((n_genes, n_genes), int), index = adata.var.index, columns=adata.var.index)
    for i in range(df.shape[0]):
        _i = df.iloc[i, 1]
        _j = df.iloc[i, 0]
        _v = {"+" : 1, "-" : -1}[df.iloc[i, 2]]
        A_ref.loc[_i, _j] = _v
    return A_ref
