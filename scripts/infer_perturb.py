import numpy as np
import scipy as sp
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import matplotlib.pyplot as plt

import glob
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--reg", type=float, default = 1e-3)
parser.add_argument("--alpha", type=float, default = 0)
parser.add_argument("--T", type=int, default = 5)
parser.add_argument("--update_couplings_iter", type=int, default = 250)
parser.add_argument("--iter", type=int, default = 1000)
parser.add_argument("--lr", type=float, default = 0.1)
parser.add_argument("--reg_sinkhorn", type=float, default = 0.1)
parser.add_argument("--outfile", type=str, default = "A.csv")
parser.add_argument("--centralities", type=str, default = None)
parser.add_argument("--numko", type=int, default = 0)
parser.add_argument("--curated", action = "store_true")


# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Synthetic/dyn-TF/dyn-TF-1000-1', '--centralities', '/home/groups/xiaojie/zys/temporal_perturb/jobs/TF_centralities.csv', '--numko', '2']
args = parser.parse_args()

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

backbone = os.path.basename(args.path).split('-')[0] if args.curated else os.path.basename(args.path).split('-')[1]
print(backbone)

paths = [args.path, ] + [args.path.replace(backbone, f"{backbone}_ko_{g}") for g in ko_genes]
adatas = [load_adata(p) for p in paths]

t_bins = np.linspace(0, 1, args.T+1)[:-1]
for adata in adatas:
    bin_timepoints(adata)

kos = []
for p in paths:
    try:
        kos.append(os.path.basename(p).split('_ko_')[1].split("-")[0])
    except:
        kos.append(None)

import torch
sys.path.append("/home/groups/xiaojie/zys/temporal_perturb/src")
import importlib
import rf
importlib.reload(rf)

options = {
    "lr" : args.lr, 
    "reg_sinkhorn" : args.reg_sinkhorn,
    "reg_A" : args.reg, 
    "reg_A_elastic" : args.alpha, 
    "iter" : args.iter,
    "ot_coupling" : True,
    "optimizer" : torch.optim.Adam
}

estim = rf.Estimator(adatas, kos, 
                           lr = options["lr"],
                           reg_sinkhorn = options["reg_sinkhorn"], 
                           reg_A = options["reg_A"], 
                           reg_A_elastic = options["reg_A_elastic"], 
                           iter = options["iter"], 
                           ot_coupling = options["ot_coupling"],
                           optimizer = options["optimizer"])

estim.fit(print_iter=100, alg = "alternating", update_couplings_iter=args.update_couplings_iter);
A = pd.DataFrame(estim.A, index = estim.genes, columns = estim.genes)
A.to_csv(os.path.join(args.path, args.outfile))

# plt.imshow(A.values, vmin = -1, vmax = 1, cmap = "bwr")
# plt.show()
