import numpy  as np
import pandas as pd
import os

def get_ref_network(path):
    df = pd.read_csv(os.path.join(path, "refNetwork.csv"))
    genes = pd.read_csv(os.path.join(path, "ExpressionData.csv"), index_col = 0).index
    A_ref = pd.DataFrame(np.zeros((len(genes), len(genes)), int), index = genes, columns = genes)
    for i in range(df.shape[0]):
        _i = df.iloc[i, 1]
        _j = df.iloc[i, 0]
        _v = {"+" : 1, "-" : -1}[df.iloc[i, 2]]
        A_ref.loc[_i, _j] = _v
    return A_ref
