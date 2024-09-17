# Figure out which genes to KO 
import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.append("../scripts")
import utils 
import importlib
importlib.reload(utils)
import networkx as nx

for backbone in ["dyn-BFC", "dyn-BF", "dyn-BFStrange", "dyn-BN8", "dyn-CN5", "dyn-CY", "dyn-FN4", "dyn-FN8", "dyn-LI", "dyn-LL", "dyn-SW", "dyn-TF"]:
    A_ref = utils.get_ref_network(f"/scratch/users/zys/Synthetic/{backbone}")
    np.fill_diagonal(A_ref.values, 0)
    g = nx.DiGraph(A_ref)
    centralities = nx.centrality.eigenvector_centrality(g.reverse())
    nx.set_node_attributes(g, centralities, name = "centrality")
    centralities_sorted = pd.Series(centralities).sort_values()[::-1]
    ko_genes = set(pd.Series(glob.glob(f"/scratch/users/zys/Synthetic/{backbone}_ko_*")).str.split("_ko_").str[1])
    pd.DataFrame(centralities_sorted[centralities_sorted.index.isin(ko_genes)]).to_csv(f"{backbone.split('-')[1]}_centralities.csv")

for backbone in ["mCAD", "GSD", "HSC"]:
    A_ref = utils.get_ref_network(f"/scratch/users/zys/Curated/{backbone}")
    np.fill_diagonal(A_ref.values, 0)
    g = nx.DiGraph(A_ref)
    centralities = nx.centrality.eigenvector_centrality(g.reverse())
    nx.set_node_attributes(g, centralities, name = "centrality")
    centralities_sorted = pd.Series(centralities).sort_values()[::-1]
    ko_genes = set(pd.Series(glob.glob(f"/scratch/users/zys/Curated/{backbone}_ko_*")).str.split("_ko_").str[1])
    pd.DataFrame(centralities_sorted[centralities_sorted.index.isin(ko_genes)]).to_csv(f"{backbone}_centralities.csv")

# import matplotlib.pyplot as plt
# edge_colors = ['blue' if g[u][v]['weight'] > 0 else 'red' for u, v in g.edges()]
# nx.draw(g, with_labels = True, node_color = [centralities[x] for x in g.nodes], edge_color = edge_colors, node_size = 2.5e2, pos=nx.shell_layout(g))
# plt.tight_layout(); plt.show()
