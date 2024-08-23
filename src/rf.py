import numpy as np
import scipy as sp
import sklearn as sk
import pandas as pd
import anndata as ad
import scanpy as sc
import ot
import torch
from torch import optim

def outer(x, y):
    return x[:, None] * y[None, :]

class Estimator:
    def __init__(self, adatas, kos, weights = None, t_key = "t", pca = "separate", norm = True, scale = None, A0 = None, b0 = None, drift = False, lr = 1e-2, iter = 1000, ot_coupling = True, n_pca_components = 10, 
                 reg_sinkhorn = 0.1, reg_A = 0.001, reg_A_elastic = 0, device = "cpu", optimizer = torch.optim.SGD):
        self.n_pca_components = n_pca_components
        self.device = device
        self.adatas = adatas
        self.weights = np.ones(len(adatas)) if weights is None else weights
        self.ot_coupling = ot_coupling
        self.kos = kos
        self.t_key = t_key
        assert all([all(adatas[0].var.index == adatas[i].var.index) for i in range(1, len(adatas))]), "adatas do not share the same var.index"
        self.genes = adatas[0].var.index
        self.n_genes = len(self.genes)
        self.Ms = [torch.tensor(self.mask_matrix(k)).to(self.device) for k in self.kos]
        self.pca = pca
        self.norm = norm
        self.T = len(self.adatas[0].obs[t_key].unique())
        self.drift = drift
        if self.norm:
            self.scaler = sk.preprocessing.StandardScaler(with_mean = True, with_std = True).fit(np.vstack([adata.X for adata in self.adatas]))
            self.std = torch.tensor(np.sqrt(self.scaler.var_))
            # self.Xs = [[self.scaler.transform(adata.X[adata.obs.t == i, :]) for i in range(self.T)] for adata in adatas]
            self.Xs = [[adata.X[adata.obs.t == i, :] - self.scaler.mean_ for i in range(self.T)] for adata in adatas]
        else:
            self.scaler = None
            self.Xs = [[adata.X[adata.obs.t == i, :] for i in range(self.T)] for adata in adatas]
            self.std = torch.ones(adatas[0].X.shape[1])
        if self.pca == "common":
            # calculate a common, global PCA basis (across all conditions)
            self.pca = sk.decomposition.PCA().fit(np.vstack([np.vstack(x) for x in self.Xs]))
            self.M_pca = [torch.tensor(self.pca.components_.T, device = self.device) for _ in range(len(self.kos))]
        elif self.pca == "separate":
            self.pca = [sk.decomposition.PCA().fit(np.vstack(x)) for x in self.Xs]
            self.M_pca = [torch.tensor(_pca.components_.T, device = self.device) for _pca in self.pca]
        self.Xs = [[torch.tensor(x).to(self.device) for x in y] for y in self.Xs]
        # set up scale
        self.scale = scale
        if self.scale is None:
            self.scale = [[ot.utils.euclidean_distances(self.Xs[j][i] @ self.M_pca[j][:, :self.n_pca_components], self.Xs[j][i+1] @ self.M_pca[j][:, :self.n_pca_components], squared = True).mean() for i in range(self.T-1)] for j in range(len(self.kos))]
        self.A = torch.zeros((self.n_genes, self.n_genes), ) if A0 is None else A0
        self.b = torch.zeros((self.n_genes, )) if b0 is None else b0
        # set up OT solutions
        self.reg_sinkhorn = reg_sinkhorn
        self.iter = iter
        self.reg_A = reg_A
        self.reg_A_elastic = reg_A_elastic
        self.lr = lr
        self.optimizer = optimizer
        # set up potential cache
        self.us = [[torch.zeros(self.Xs[i][j].shape[0], dtype = torch.float64).to(self.device) for j in range(self.T-1)] for i in range(len(self.kos))]
        self.vs = [[torch.zeros(self.Xs[i][j+1].shape[0], dtype = torch.float64).to(self.device) for j in range(self.T-1)] for i in range(len(self.kos))]
    def mask_matrix(self, ko):
        M = pd.DataFrame(np.ones((self.n_genes, self.n_genes), dtype = int), index = self.genes, columns = self.genes)
        if ko is not None:
            M.loc[:, ko] = 0
            # M.loc[ko, ko] = 1
        return M.to_numpy()
    def fit(self, print_iter = 100, alg = "joint", update_couplings_iter = 100):
        self.trace = []
        A = torch.tensor(self.A, requires_grad = True, dtype = torch.float64)
        b = torch.tensor(self.b, requires_grad = True, dtype = torch.float64)
        # setup transport plans before first iteration
        t = 1/self.T
        print("Updating transport plans")
        def update_plans():
            Ts = []
            for i in range(len(self.kos)):
                _Ts = []
                for j in range(self.T-1):
                    with torch.no_grad():
                        P = torch.linalg.matrix_exp(t*A * self.Ms[i])
                    # C = ot.utils.euclidean_distances((self.Xs[i][j] @ P) @ self.M_pca[i][:, :self.n_pca_components], self.Xs[i][j+1] @ self.M_pca[i][:, :self.n_pca_components], squared=True)
                    C = ot.utils.euclidean_distances((((self.Xs[i][j] / self.std) @ P + t * (self.b * self.Ms[i][0, :])) * self.std) @ self.M_pca[i][:, :self.n_pca_components],
                                                     self.Xs[i][j+1] @ self.M_pca[i][:, :self.n_pca_components], squared=True)
                    eps = self.reg_sinkhorn * self.scale[i][j]
                    if self.ot_coupling:
                        _Ts.append(ot.sinkhorn(torch.tensor(ot.utils.unif(self.Xs[i][j].shape[0])),
                                        torch.tensor(ot.utils.unif(self.Xs[i][j+1].shape[0])), C, eps, numItermax = 5_000))
                    else:
                        _Ts.append(outer(torch.tensor(ot.utils.unif(self.Xs[i][j].shape[0])),
                                        torch.tensor(ot.utils.unif(self.Xs[i][j+1].shape[0])))) # independent coupling 
                Ts.append(_Ts)
            return Ts
        Ts = update_plans()
        def L_joint(A, b, Ts, Xs, us, vs, M_pca, scale):
            P = torch.linalg.matrix_exp(t*A)
            Ls = []
            for i in range(self.T-1):
                # C = ot.utils.euclidean_distances(Xs[i] @ P @ M_pca[:, :self.n_pca_components], Xs[i+1] @ M_pca[:, :self.n_pca_components], squared=True)
                C = ot.utils.euclidean_distances((((Xs[i] / self.std) @ P + t * b) * self.std) @ M_pca[:, :self.n_pca_components], Xs[i+1] @ M_pca[:, :self.n_pca_components], squared=True)
                eps = self.reg_sinkhorn * scale[i]
                p, q = ot.utils.unif(C.shape[0], type_as = C), ot.utils.unif(C.shape[1], type_as = C)
                if self.ot_coupling:
                    _, log = ot.sinkhorn2(p, q, C, eps,
                                            numItermax = 5_000, log = True,  
                                            warmstart = (us[i].detach(), vs[i].detach()))
                    T = log['u'][:, None] * torch.exp(-C/eps) * log['v'][None, :]
                    with torch.no_grad():
                        us[i] = torch.log(log['u'])
                        vs[i] = torch.log(log['v'])
                    Ls.append(eps * ((torch.log(log['u']) * p).sum() + (torch.log(log['v']) * q).sum()))
                else:
                    T = outer(p, q)
                    Ls.append((C * T).sum() + eps * (T * torch.log(T)).sum())
            return sum(Ls) / len(Ls)
        def L_fixed(A, b, Ts, Xs, mask, scale):
            P = torch.linalg.matrix_exp(t*A)
            Ls = []
            for i in range(self.T-1):
                # C = ot.utils.euclidean_distances(Xs[i] @ P @ self.M_pca, Xs[i+1] @ self.M_pca, squared=True)
                C = ot.utils.euclidean_distances((((Xs[i] / self.std) @ P + t * b) * self.std) * mask[None, :], Xs[i+1] * mask[None, :], squared=True)
                eps = self.reg_sinkhorn * scale[i]
                Ls.append(torch.sum(C * Ts[i]) + eps * (Ts[i] * torch.log(Ts[i])).sum())
            return sum(Ls) / len(Ls)
        def R(A, b, t = 0.0, include_diag = False):
            M = torch.ones_like(A)
            if include_diag == False:
                M.fill_diagonal_( 0)
            return t*(torch.sum((A * M)**2) + torch.sum(b)**2) + (1-t)*(torch.sum(torch.abs(A * M)) + torch.sum(torch.abs(b)))
        optimizer = self.optimizer([A, b] if self.drift else [A, ], lr=self.lr)
        for it in range(self.iter):
            optimizer.zero_grad()
            if alg == "joint":
                loss1 = sum([self.weights[i]*L_joint(A * self.Ms[i], b * self.Ms[i][0, :], Ts[i], self.Xs[i], self.us[i], self.vs[i], self.M_pca[i], self.scale[i]) for i in range(len(self.kos))]) / self.weights.sum()
            elif (alg == "fixed") or (alg == "alternating"):
                loss1 = sum([self.weights[i]*L_fixed(A * self.Ms[i], b * self.Ms[i][0, :], Ts[i], self.Xs[i], self.Ms[i][0, :], self.scale[i]) for i in range(len(self.kos))]) / self.weights.sum()
            loss2 = R(A, b, self.reg_A_elastic)
            loss = loss1 + self.reg_A * loss2
            if it % print_iter == 0:
                print(f"iteration {it}, loss = {loss.item()}, L = {loss1.item()}, R = {loss2.item()}")
            self.trace.append(loss.item())
            loss.backward()
            optimizer.step()
            self.A = A.detach()
            self.b = b.detach()
            if (alg == "alternating") & ((it+1) % update_couplings_iter == 0):
                Ts = update_plans()
        self.A_orig = self.A.clone()
        self.A.fill_diagonal_(0)
        self.Ts = Ts
        return self.A

        
        


