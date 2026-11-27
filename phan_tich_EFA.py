from pathlib import Path
import numpy as np
from openpyxl import load_workbook
from scipy.stats import rankdata

def analyze(path):
    data = load_workbook(path, read_only=True, data_only=True)['Du_lieu_500'].values
    header = next(data)
    names = [f'Q{i:02d}' for i in range(7,37)]
    ix = [header.index(q) for q in names]
    x = np.array([[row[j] for j in ix] for row in data], dtype=float)
    assert x.shape == (500,30) and np.isin(x,[0,1,2,3,4]).all()
    for i, q in enumerate(range(7,37)):
        if 13 <= q <= 24 or q >= 31:
            x[:,i] = 4 - x[:,i]
    ranks = rankdata(x,axis=0)
    corr = np.corrcoef(ranks.T)
    observed = np.linalg.eigvalsh(corr)[::-1]
    rng = np.random.default_rng(42)
    null = []
    for _ in range(1000):
        shuffled = np.column_stack([rng.permutation(column) for column in ranks.T])
        null.append(np.linalg.eigvalsh(np.corrcoef(shuffled.T))[::-1])
    cut = np.quantile(null,.95,axis=0)
    print('Observed / null:',observed[:5],cut[:5])
    count = next((i for i, value in enumerate(observed) if value <= cut[i]), len(cut))
    print("Factors retained:", count)
    if count != 1:
        return  # The one-factor solution below applies only when one factor is retained.
    inv = np.linalg.inv(corr)
    partial = -inv / np.sqrt(np.outer(inv.diagonal(), inv.diagonal()))
    off_diag = ~np.eye(len(names), dtype=bool)
    r2 = np.sum(corr[off_diag]**2)
    print("KMO:", r2 / (r2 + np.sum(partial[off_diag]**2)))
    h = 1-1/inv.diagonal()
    for _ in range(1000):
        reduced = corr.copy()
        np.fill_diagonal(reduced,h)
        values, vectors = np.linalg.eigh(reduced)
        loading = vectors[:,-1]*np.sqrt(values[-1])
        if np.max(abs(loading**2-h)) < 1e-8:
            break
        h = loading**2
    else:
        raise RuntimeError('PAF did not converge')
    loading *= np.sign(loading.mean())
    print('Loadings:',dict(zip(names,loading.round(3))))
    print('Variance:',np.mean(loading**2))
if __name__=='__main__':
    analyze(Path(__file__).resolve().parent/'Du_lieu_khao_sat_500_sinh_vien.xlsx')
