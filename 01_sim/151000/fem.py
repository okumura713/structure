"""151000-01 ホイールフォークペア 線形静解析（2次四面体 tet10）

境界条件:
  - パイプ内径 φ24（フレームへの揺動ピン 100005-03）: 内面節点を完全固定（ピン支持）
  - ローラー軸穴 φ20.1（下端, 両側板）: タンデムローラー反力 F_A を 1/2 ずつベアリング荷重（cos分布）
  - プッシュロッド穴 φ20.1（上部, 両側板）: プッシュロッド軸力 T·e を 1/2 ずつベアリング荷重
  F_A は sim2d.py の動力学結果、T は 151000-01 単体のパイプ中心まわりモーメント釣り合いから決まる
  （パイプ反力 F_O = -(F_A + T e) は固定部の反力として自動的に生じる）
材料: 側板 SS400, パイプ STKM13A, E=206 GPa, ν=0.3（溶接部は母材と同一剛性で一体化）
単位: mm, N, MPa
"""
import numpy as np, pathlib, sys, io, time, json
import scipy.sparse as sp
import pypardiso

HERE = pathlib.Path(__file__).parent
E, NU = 206000.0, 0.3
GEO = dict(A=(0.0, 32.5), O=(-55.5, 156.2), P=(-41.4, 179.0))
R_HOLE, R_BORE = 10.05, 12.0

# ---------- tet10 形状関数 ----------
EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]   # gmsh tet10 の辺節点 4..9


def dN_dL(L):
    """バリセントリック L(4) における dN/dL (10x4)"""
    d = np.zeros((10, 4))
    for i in range(4): d[i, i] = 4 * L[i] - 1
    for k, (i, j) in enumerate(EDGES):
        d[4 + k, i] = 4 * L[j]; d[4 + k, j] = 4 * L[i]
    return d


def dN_dxi(xi):
    L = np.array([1 - sum(xi), *xi])
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    return dN_dL(L) @ dL                    # 10x3


GP = np.array([[0.1381966011250105] * 3] * 4); GP[1, 0] = GP[2, 1] = GP[3, 2] = 0.5854101966249685
GW = np.full(4, 1 / 24)
NODE_XI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5], [0, .5, .5], [.5, 0, .5]])


def Dmat():
    l = E * NU / ((1 + NU) * (1 - 2 * NU)); m = E / (2 * (1 + NU))
    D = np.full((3, 3), l) + 2 * m * np.eye(3)
    return np.block([[D, np.zeros((3, 3))], [np.zeros((3, 3)), m * np.eye(3)]])


def Bmats(X, T, xi):
    """各要素の B (ne,6,30) と detJ"""
    dNx = dN_dxi(xi)                                  # 10x3
    J = np.einsum('eki,kj->eij', X[T], dNx)           # ne,3,3  (J_ij = dx_i/dxi_j)
    det = np.linalg.det(J)
    G = np.einsum('kj,eji->eki', dNx, np.linalg.inv(J))   # dN/dx : ne,10,3
    ne = len(T); B = np.zeros((ne, 6, 30))
    for a in range(3): B[:, a, a::3] = G[:, :, a]
    B[:, 3, 0::3] = G[:, :, 1]; B[:, 3, 1::3] = G[:, :, 0]
    B[:, 4, 1::3] = G[:, :, 2]; B[:, 4, 2::3] = G[:, :, 1]
    B[:, 5, 0::3] = G[:, :, 2]; B[:, 5, 2::3] = G[:, :, 0]
    return B, det


def assemble(X, T):
    D = Dmat(); n = len(X)
    dof = (3 * T[:, :, None] + np.arange(3)).reshape(len(T), 30)
    rows, cols, vals = [], [], []
    for s in range(0, len(T), 20000):
        Ts = T[s:s + 20000]; Ke = 0
        for xi, w in zip(GP, GW):
            B, det = Bmats(X, Ts, xi)
            assert (det > 0).all(), 'negative Jacobian'
            Ke = Ke + w * det[:, None, None] * np.einsum('eia,ij,ejb->eab', B, D, B)
        d = dof[s:s + 20000]
        rows.append(np.repeat(d, 30, 1).ravel()); cols.append(np.tile(d, 30).ravel()); vals.append(Ke.ravel())
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n))
    return K


def stress_operator(X, T, nodes):
    """指定節点（表面節点）の節点応力（要素節点値の平均）= S @ u,  S: (6*len(nodes) x 3n)"""
    D = Dmat(); n = len(X)
    loc = np.full(n, -1); loc[nodes] = np.arange(len(nodes))
    dof = (3 * T[:, :, None] + np.arange(3)).reshape(len(T), 30)
    rows, cols, vals = [], [], []
    cnt = np.bincount(T.ravel(), minlength=n).astype(float)
    for k, xi in enumerate(NODE_XI):
        e = np.where(loc[T[:, k]] >= 0)[0]
        B, _ = Bmats(X, T[e], xi)
        DB = np.einsum('ij,ejb->eib', D, B) / cnt[T[e, k]][:, None, None]     # ne,6,30
        r = 6 * loc[T[e, k]][:, None, None] + np.arange(6)[None, :, None]
        rows.append(np.broadcast_to(r, DB.shape).ravel()); cols.append(np.broadcast_to(dof[e, None, :], DB.shape).ravel())
        vals.append(DB.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(6 * len(nodes), 3 * n))


def von_mises(s):
    s = s.reshape(-1, 6) if s.ndim == 1 else s
    sx, sy, sz, txy, tyz, tzx = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))


FACES = [((0, 1, 2), (4, 5, 6)), ((0, 1, 3), (4, 9, 7)), ((0, 2, 3), (6, 8, 7)), ((1, 2, 3), (5, 8, 9))]


def boundary_faces(T, mat):
    F = np.vstack([T[:, list(c) + list(m)] for c, m in FACES]); FM = np.tile(mat, 4)
    key = np.sort(F[:, :3], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    return F[b], FM[b]


class Hole:
    """円筒穴内面のベアリング荷重（cos分布）"""
    TRI_GP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]); TRI_W = np.full(3, 1 / 6)

    def __init__(self, X, faces, center):
        self.c = np.array(center); self.faces = faces
        N, pts, wts, nrm = [], [], [], []
        for (s, t), w in zip(self.TRI_GP, self.TRI_W):
            L = np.array([1 - s - t, s, t])
            Nq = np.r_[L * (2 * L - 1), 4 * L[0] * L[1], 4 * L[1] * L[2], 4 * L[0] * L[2]]
            P = X[faces]                                  # nf,6,3
            # 面積要素: 頂点三角形で近似（穴は細かく分割されるので十分）
            a = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
            x = np.einsum('k,fkj->fj', Nq, P)
            r = x[:, 1:] - self.c; r /= np.linalg.norm(r, axis=1)[:, None]
            N.append(Nq); wts.append(w * a); nrm.append(r)
        self.N, self.w, self.n = np.array(N), np.array(wts), np.array(nrm)   # (3,6) (3,nf) (3,nf,2)

    def load(self, F, nn):
        """穴に F(Y,Z) [N] を与える節点荷重ベクトル（3nn）"""
        f = np.zeros(3 * nn); Fm = np.hypot(*F)
        if Fm < 1e-9: return f
        u = F / Fm
        p = np.clip(self.n @ u, 0, None)                    # (3,nf) cos分布
        tr = p[..., None] * self.n                          # 単位圧力あたり traction
        tot = np.einsum('qf,qfj->j', self.w, tr)
        # 法線圧(cos分布) a·tr と、合力誤差補正用の一様せん断 b (cos分布重み) で合力を F に一致させる
        cw = np.einsum('qf,qf->', self.w, p)
        A2 = np.c_[tot, [cw, 0], [0, cw]]                   # 未知 [p0, by, bz]
        p0, by, bz = np.linalg.lstsq(np.vstack([A2, [0, 1e-3 * cw, 0], [0, 0, 1e-3 * cw]]),
                                     np.r_[F, 0, 0], rcond=None)[0]
        for q in range(3):
            trq = p0 * tr[q] + p[q][:, None] * np.array([by, bz])
            fq = self.w[q][:, None, None] * self.N[q][None, :, None] * trq[:, None, :]   # nf,6,2
            for j in range(2):
                np.add.at(f, 3 * self.faces.ravel() + 1 + j, fq[:, :, j].ravel())
        return f


def fix_curved(X, T):
    """ヤコビアンが負になる曲がった要素の中間節点を辺の中点へ戻す"""
    X = X.copy()
    for it in range(10):
        bad = np.zeros(len(T), bool)
        for xi in np.vstack([GP, NODE_XI * 0.98 + 0.005]):
            _, det = Bmats(X, T, xi); bad |= det <= 0
        if not bad.any(): break
        for k, (i, j) in enumerate(EDGES):
            X[T[bad, 4 + k]] = (X[T[bad, i]] + X[T[bad, j]]) / 2
        print(f'  fix_curved iter{it}: {bad.sum()} elements straightened')
    return X


def setup():
    d = np.load(HERE / 'mesh.npz'); X, T, M = d['xyz'], d['tet'], d['mat']
    X = fix_curved(X, T)
    nn = len(X)
    BF, BM = boundary_faces(T, M)
    def on_cyl(c, r, tol, m):
        rr = np.hypot(X[:, 1] - c[0], X[:, 2] - c[1]); ok = np.abs(rr - r) < tol
        fm = ok[BF].all(1) & (BM == m)
        return BF[fm]
    holes = {}
    for key, c in (('A', GEO['A']), ('P', GEO['P'])):
        f = on_cyl(c, R_HOLE, 0.08, 0)
        left = X[f[:, 0], 0] < 183.5
        holes[key] = [Hole(X, f[left], c), Hole(X, f[~left], c)]
        print(f'hole {key}: faces L={left.sum()} R={(~left).sum()}')
    bore = on_cyl(GEO['O'], R_BORE, 0.08, 1)
    fixed = np.unique(bore)
    print('bore fixed nodes', len(fixed), 'X range', X[fixed, 0].min(), X[fixed, 0].max())
    return X, T, M, BF, holes, fixed


def solve_case(case, X, T, M, BF, holes, fixed, K, S, surf, dt_frame=0.01):
    r = np.load(HERE / f'loads_{case}.npz')
    t = r['t']; FAb, FP = r['FAb'], r['FP']
    # フレーム抽出: 等間隔 + 各荷重ピーク時刻
    sel = set(np.round(np.arange(0, t[-1], dt_frame) / (t[1] - t[0])).astype(int).tolist())
    for a in (np.hypot(*FAb.T), np.abs(r['Tpr']), np.hypot(*r['FO'].T)):
        sel.add(int(np.argmax(a)))
    sel = np.array(sorted(i for i in sel if i < len(t)))
    nn = len(X)
    Fmat = np.zeros((3 * nn, len(sel)))
    for k, i in enumerate(sel):
        f = np.zeros(3 * nn)
        for h in holes['A']: f += h.load(FAb[i] / 2, nn)
        for h in holes['P']: f += h.load(FP[i] / 2, nn)
        Fmat[:, k] = f
    free = np.setdiff1d(np.arange(3 * nn), (3 * fixed[:, None] + np.arange(3)).ravel())
    Kf = K[free][:, free].tocsr()
    t0 = time.time()
    U = np.zeros_like(Fmat)
    U[free] = pypardiso.spsolve(Kf, Fmat[free])
    print(f'  solve {len(sel)} frames: {time.time() - t0:.1f}s')
    reac = K @ U - Fmat
    VM = np.zeros((len(surf), len(sel)), np.float32)
    for k in range(len(sel)):
        VM[:, k] = von_mises(S @ U[:, k])
    return dict(t=t[sel], idx=sel, U=U.reshape(nn, 3, -1)[surf].astype(np.float32), VM=VM,
                FA=FAb[sel], FP=FP[sel], FO=r['FO'][sel], T=r['Tpr'][sel],
                reac=np.array([reac[1::3].sum(0), reac[2::3].sum(0)]).T)


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    X, T, M, BF, holes, fixed = setup()
    t0 = time.time(); K = assemble(X, T); print(f'K assembled {K.shape} nnz={K.nnz} {time.time() - t0:.1f}s')
    surf = np.unique(BF[:, :3])                      # 表面の頂点節点（表示・評価用）
    S = stress_operator(X, T, surf); print('stress operator ok', len(surf))
    summ = {}
    for case in ('fork_first', 'drive_first'):
        print(case)
        R = solve_case(case, X, T, M, BF, holes, fixed, K, S, surf)
        err = np.abs(R['reac'] - R['FO']).max()
        k = int(np.argmax(R['VM'].max(0))); n = int(np.argmax(R['VM'][:, k]))
        nmat = np.zeros(len(X), int); nmat[T[M == 1].ravel()] = 1
        print(f'  反力チェック |pipe reaction - F_O|max = {err:.2f} N')
        print(f'  最大ミーゼス {R["VM"][n, k]:.1f} MPa  t={R["t"][k]:.3f}s  位置 {X[surf[n]].round(1)}  材料 {"パイプ" if nmat[surf[n]] else "側板"}')
        print(f'  静的(t=0) 最大 {R["VM"][:, 0].max():.1f} MPa')
        tri = np.searchsorted(surf, BF[:, :3])
        np.savez_compressed(HERE / f'fem_{case}.npz', X=X[surf], tri=tri, nmat=nmat[surf], **R)
        summ[case] = dict(vm_max=float(R['VM'][n, k]), t=float(R['t'][k]), xyz=X[surf[n]].tolist(),
                          vm_static=float(R['VM'][:, 0].max()))
    json.dump(summ, open(HERE / 'fem_summary.json', 'w'), indent=1)
