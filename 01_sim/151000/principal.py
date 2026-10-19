"""ピーク時刻の主応力（最大応力点・プッシュロッド穴外縁・パイプ）を確認"""
import numpy as np, sys, io, json, pathlib
import fem, pypardiso
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = pathlib.Path(__file__).parent
X, T, M, BF, holes, fixed = fem.setup()
K = fem.assemble(X, T); nn = len(X)
surf = np.unique(BF[:, :3]); S = fem.stress_operator(X, T, surf)
free = np.setdiff1d(np.arange(3 * nn), (3 * fixed[:, None] + np.arange(3)).ravel())
Kf = K[free][:, free].tocsr()
out = {}
for case in ('fork_first', 'drive_first'):
    r = np.load(HERE / f'loads_{case}.npz'); FAb, FP = r['FAb'], r['FP']
    ks = [0, int(np.argmax(-r['Tpr']))]
    Fm = np.zeros((3 * nn, 2))
    for j, i in enumerate(ks):
        for h in holes['A']: Fm[:, j] += h.load(FAb[i] / 2, nn)
        for h in holes['P']: Fm[:, j] += h.load(FP[i] / 2, nn)
    U = np.zeros_like(Fm); U[free] = pypardiso.spsolve(Kf, Fm[free])
    for j, lab in enumerate(('static', 'peak')):
        sig = (S @ U[:, j]).reshape(-1, 6); vm = fem.von_mises(sig)
        Xs = X[surf]
        dP = np.hypot(Xs[:, 1] + 41.4, Xs[:, 2] - 179.0)
        regions = {'P穴内面(最大点)': dP < 10.3, 'P穴外側の縁(自由縁)': (dP > 12) & (dP < 20) & (M[0] == M[0]),
                   'パイプ': np.isin(surf, np.unique(T[M == 1]))}
        for name, m in regions.items():
            i = np.where(m)[0][np.argmax(vm[m])]
            s = sig[i]; Tn = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
            p = np.linalg.eigvalsh(Tn)
            print(f'{case:12s} {lab:6s} {name:16s} VM {vm[i]:6.1f}  σ1 {p[2]:7.1f}  σ3 {p[0]:7.1f}  at {Xs[i].round(1)}')
            out[f'{case}|{lab}|{name}'] = dict(vm=float(vm[i]), s1=float(p[2]), s3=float(p[0]), xyz=Xs[i].tolist())
json.dump(out, open(HERE / 'principal.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
