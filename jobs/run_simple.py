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
import seaborn as sns

import glob
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--T", type=int, default = 5)
parser.add_argument("--curated", action = "store_true")

# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Synthetic/dyn-TF/dyn-TF-1000-1',]
# sys.argv = ['infer_perturb.py', '/scratch/users/zys/Curated/HSC/HSC-1000-1', '--curated']
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
ko_genes = []

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

adata = adatas[0]

# GENIE3
sys.path.append("../../temporal_perturb/tools/GENIE3/GENIE3_python/")
from GENIE3 import GENIE3
A_genie3 = GENIE3(adata.X, nthreads = 8)
pd.DataFrame(A_genie3, index = adata.var.index, columns = adata.var.index).to_csv(os.path.join(args.path, "A_genie3.csv"))

# dynGENIE3
sys.path.append("../../temporal_perturb/tools/dynGENIE3/dynGENIE3_python")
X = []
t = []
X.append(np.vstack([adata.X[adata.obs.t == t, :].mean(0) for t in np.sort(adata.obs.t.unique())]))
t.append(np.sort(adata.obs.t.unique()))
from dynGENIE3 import dynGENIE3
A_dyngenie3, _, _, _, _ =  dynGENIE3(X, t)
pd.DataFrame(A_dyngenie3, index = adata.var.index, columns = adata.var.index).to_csv(os.path.join(args.path, f"A_dyngenie3_T_{args.T}.csv"))

# GLASSO
import sklearn as sk
from sklearn import covariance, preprocessing
gl = sk.covariance.GraphicalLassoCV().fit(sk.preprocessing.StandardScaler().fit_transform(adata.X))
A_glasso = -gl.precision_
np.fill_diagonal(A_glasso, 0)
pd.DataFrame(A_glasso, index = adata.var.index, columns = adata.var.index).to_csv(os.path.join(args.path, "A_glasso.csv"))

# SINCERITIES
df = pd.DataFrame(adata.X, columns = adata.var.index, index = adata.obs.index)
df["t"] = adata.obs["t"]
df =df.sort_values(by = "t")
df.to_excel("X.xlsx", index = False)
cmd = "matlab -nodesktop -nosplash -r \"addpath('../../temporal_perturb/tools/SINCERITIES/matlab'); run run_sincerities.m; exit\" | tail"
print(f"Ran SINCERITIES, return code = {os.system(cmd)}")
A_sincerities = np.zeros((adata.shape[1], adata.shape[1]))
for row in pd.read_csv("A.txt").itertuples():
    i = int(row.SourceGENES.split(" ")[-2])
    j = int(row.TargetGENES.split(" ")[-2])
    v = row.Interaction
    A_sincerities[i-1, j-1] = v
os.system("rm X.xlsx A.txt")
pd.DataFrame(A_sincerities, index = adata.var.index, columns = adata.var.index).to_csv(os.path.join(args.path, f"A_sincerities_T_{args.T}.csv"))

