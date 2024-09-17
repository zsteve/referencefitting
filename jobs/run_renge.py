import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy
import sys, os, itertools
sys.path.append(os.path.abspath("../../temporal_perturb/tools/RENGE/src/renge"))
from renge import Renge
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')
sns.set_style('ticks')

import glob
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--T", type=int, default = 5)
parser.add_argument("--centralities", type=str, default = None)
parser.add_argument("--numko", type=int, default = 0)
parser.add_argument("--curated", action = "store_true")

# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Synthetic/dyn-TF/dyn-TF-1000-1', '--centralities', '/home/groups/xiaojie/zys/temporal_perturb/jobs/TF_centralities.csv', '--numko', '2']
# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Curated/HSC/HSC-1000-1', '--centralities', '/home/groups/xiaojie/zys/temporal_perturb/jobs/HSC_centralities.csv', '--numko', '2', '--curated']
args = parser.parse_args()

# Load data
def load_adata(path):
    adata = ad.AnnData(pd.read_csv(os.path.join(path, "ExpressionData.csv"), index_col = 0).T)
    df_pt = pd.read_csv(os.path.join(path, "PseudoTime.csv"), index_col = 0)
    df_pt[np.isnan(df_pt)] = 0
    adata.obs["t_sim"] = np.max(df_pt.to_numpy(), -1)
    sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    return adata
def bin_timepoints(adata):
    adata.obs["t"] = np.digitize(adata.obs.t_sim, t_bins)-1
ko_genes = pd.read_csv(args.centralities, index_col = 0).index[:args.numko] if args.centralities is not None else []

# Get backbone
backbone = os.path.basename(args.path).split('-')[0] if args.curated else os.path.basename(args.path).split('-')[1]
print(f"backbone = {backbone}")

paths = [args.path, ] + [args.path.replace(backbone, f"{backbone}_ko_{g}") for g in ko_genes]
names = [os.path.basename(p).split("-")[0 if args.curated else 1] for p in paths] 
adatas = [load_adata(p) for p in paths]

# Bin timepoints
t_bins = np.linspace(0, 1, args.T+1)[:-1]
for adata in adatas:
    bin_timepoints(adata)
# Get KOs
kos = []
for p in paths:
    try:
        kos.append(os.path.basename(p).split('_ko_')[1].split("-")[0])
    except:
        kos.append(None)

ko_idx = [np.where(np.array(kos) == None)[0][0], ]
for x in ko_genes:
    try:
        ko_idx.append(np.where(np.array(kos) == x)[0][0])
    except:
        pass

# Concat all for RENGE analysis
adata_all = ad.concat([adatas[k] for k in ko_idx], keys = [names[k] for k in ko_idx], index_unique = '_')
adata_all.obs["condition"] = [x[-1] for x in adata_all.obs.index.str.split("_")]

E = pd.DataFrame(adata_all.X, index = adata_all.obs.index, columns=adata_all.var.index)
X = pd.DataFrame(np.zeros_like(adata_all.X), index = adata_all.obs.index, columns = adata_all.var.index)
for i in range(adata_all.shape[0]):
    if adata_all.obs.condition[i] != backbone:
        X.loc[adata_all.obs.index[i], adata_all.obs.condition[i]] = 1
X["t"] = adata_all.obs["t"]

# RENGE predict
reg = Renge()
A = reg.estimate_hyperparams_and_fit(X, E, n_trials=30)
A_pred = pd.DataFrame(A.T, index = adata_all.var.index, columns=adata_all.var.index)
A_pred.to_csv(os.path.join(args.path, f"A_renge_T_{args.T}_ko_{args.numko}.csv"))

# Get filtered
A_qval = reg.calc_qval(n_boot=30)
A_filt = A.copy()
A_filt[A_qval >= 0.01] = 0
A_norm = A_filt / np.abs(A_filt).max(axis=0)
A_norm = A_norm.fillna(0)
A_pred_filter = pd.DataFrame(A_norm.T, index = adata_all.var.index, columns=adata_all.var.index)
A_pred_filter.to_csv(os.path.join(args.path, f"A_renge_filter_T_{args.T}_ko{args.numko}.csv"))
