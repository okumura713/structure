"""ABM15M 段差乗り越え 2D 動力学シミュレーション（側面 Y-Z 平面, 左右対称）

モデル:
  - 車体(フレーム+フォーク+駆動ユニット+荷物)を1つの剛体とする（油圧はロック、リンクは固定）
  - フォーク先端（151000-01 + タンデムローラー）は車体に対し上下方向に フォーク曲げ剛性 k_fork で弾性支持
    （C形鋼 約150x45 t4 の片持ち/パレット支持の中間を想定。感度は k_fork を振って確認）
  - タンデムローラー(160000)はホイールフォーク(151000-01)の軸穴まわりに揺動する剛体
  - 接地: 駆動輪 R75 (1輪, 中央), 従動ローラー R32.5 x 2 (ピッチ90) x 左右2セット
  - 接触力: Hunt-Crossley  F = K δ^1.5 (1 + α δ')  （ウレタン車輪相当）
  - 駆動輪は速度制御（55 m/min を保つよう牽引、ただし |Ft| ≤ μN）
  - 段差: 高さ h_step [mm] の直角段差（角部は鋭角）。2026-10-20 に 6 mm → 3 mm へ変更（6 mm は実使用より厳しすぎた）
出力: 151000-01（片側）にかかる 3ピン荷重の時刻歴 → loads_<case>.npz
座標: Y = 車体前後(+Y がフォーク先端側), Z = 鉛直上向き, 原点はSTEPモデル原点。単位 SI
"""
import numpy as np, pathlib, json
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

HERE = pathlib.Path(__file__).parent
mm = 1e-3
g = 9.81

# ---------- STEPから抽出した幾何 [mm] ----------
GEO = dict(
    A=(0.0, 32.5),          # ローラー軸穴（151000-01 下端穴, タンデムローラー揺動軸）
    O=(-55.5, 156.2),       # パイプ φ24 中心（フレームへの揺動ピン 100005-03）
    P=(-41.4, 179.0),       # プッシュロッド ピン穴（100007-04）
    R=(-870.8, 170.9),      # プッシュロッド後端ピン（ベルクランク 131003-01）
    D=(-1223.3, 75.0),      # 駆動輪中心
    r_roll=32.5, pitch=45.0, r_drive=75.0,
    fork_top=201.0, backrest=-835.0,
)
# ---------- 条件・仮定 ----------
PAR = dict(
    m_load=1500.0,            # 荷物 [kg]
    load_center=550.0,        # バックレストから荷重中心 [mm]（パレット1100の中央）
    load_cg_h=500.0,          # フォーク上面から荷物重心高さ [mm]（仮定: 荷姿高さ1000）
    load_L=1100.0, load_H=1000.0,
    m_truck=300.0,            # 車両自重 [kg]（仮定; STEP体積からの推定）
    truck_cg=(-950.0, 400.0), truck_rg=350.0,
    I_bogie=2 * 4.2e-3,       # タンデムローラー揺動慣性 左右2組 [kg m^2]
    K_roll=2.68e8,            # ローラー接触 K [N/m^1.5]（φ65ウレタン: 3kN で δ0.5mm）
    K_drive=1.9e8,            # 駆動輪接触 K [N/m^1.5]（φ150ウレタン: 6kN で δ1.0mm）
    alpha=1.3,                # Hunt-Crossley 減衰 [s/m]（ウレタン反発係数 e≈0.65 相当）
    k_fork=2.0e6,             # フォーク先端 上下剛性 片側 [N/m]（= 2000 N/mm, None で剛体）
    m_tip=10.0,               # フォーク先端 等価質量 片側 [kg]（フォーク先端部+151000+ローラー+ロッド）
    zeta_fork=0.05,           # フォーク/パレット/荷物 系の減衰比（車体等価質量基準）
    mu=0.6,                   # 駆動輪-床 摩擦係数
    v0=55.0 / 60.0,           # 走行速度 [m/s]
    h_step=3.0,               # 段差 [mm]（初版は 6.0）
    tau_v=0.1,                # 速度制御 時定数 [s]
)


def build(par=PAR):
    G = {k: (np.array(v) * mm if isinstance(v, tuple) else v) for k, v in GEO.items()}
    yL = (GEO['backrest'] + par['load_center']) * mm
    zL = (GEO['fork_top'] + par['load_cg_h']) * mm
    mL, mT = par['m_load'], par['m_truck']
    cT = np.array(par['truck_cg']) * mm
    M = mL + mT
    Gc = (mL * np.array([yL, zL]) + mT * cT) / M
    I = (mL * ((par['load_L'] * mm) ** 2 + (par['load_H'] * mm) ** 2) / 12 + mL * np.sum((np.array([yL, zL]) - Gc) ** 2)
         + mT * (par['truck_rg'] * mm) ** 2 + mT * np.sum((cT - Gc) ** 2))
    return dict(G=G, M=M, I=I, Gc=Gc, cL=np.array([yL, zL]), par=par)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def ground_contacts(c, r, ys, h, d):
    """円(中心c,半径r)と段差地形の接触。d=+1: +Y方向へ上る段差, d=-1: -Y方向へ上る段差
    戻り値: [(δ, n, p)]  δ:食い込み, n:地面→中心の単位法線, p:接触点"""
    out = []
    s = d * (c[0] - ys)          # 段差位置からの進行方向距離
    if s < 0:
        out.append((r - c[1], np.array([0.0, 1.0]), np.array([c[0], 0.0])))      # 下段床
        p = np.array([ys, np.clip(c[1], 0.0, h)])                                # 段差面/角
        v = c - p; L = np.hypot(*v)
        if L > 1e-12: out.append((r - L, v / L, p))
    else:
        out.append((r - (c[1] - h), np.array([0.0, 1.0]), np.array([c[0], h])))  # 上段床
    return [o for o in out if o[0] > 0]


def contact_force(dlt, dlt_dot, K, alpha):
    return max(0.0, K * dlt ** 1.5 * (1 + alpha * dlt_dot))


def make_rhs(S, d, ys, Kv):
    G, M, I, Gc, par = S['G'], S['M'], S['I'], S['Gc'], S['par']
    h = par['h_step'] * mm
    rA, rD = G['A'] - Gc, G['D'] - Gc
    rr, pt = G['r_roll'] * mm, G['pitch'] * mm
    mt = 2 * par['m_tip']
    rigid = par['k_fork'] is None
    kf = 0.0 if rigid else 2 * par['k_fork']
    m_eff = 1 / (1 / M + (G['A'][0] - Gc[0]) ** 2 / I)          # A点での車体等価質量
    cf = 0.0 if rigid else 2 * par['zeta_fork'] * np.sqrt(kf * m_eff)

    def forces(t, x):
        yg, zg, phi, beta, u, vy, vz, w, wb, ud = x
        Rm = rot(phi); cg = Gc + np.array([yg, zg])
        vel = lambda r: np.array([vy, vz]) + w * np.array([-r[1], r[0]])
        Ftot = np.array([0.0, -M * g]); Mtot = 0.0
        # 駆動輪
        dD = Rm @ rD; cD = cg + dD; vD = vel(dD)
        FD = np.zeros(2); ND = 0.0
        for dl, n, p in ground_contacts(cD, G['r_drive'] * mm, ys, h, d):
            Fn = contact_force(dl, -vD @ n, par['K_drive'], par['alpha'])
            t_ = np.array([n[1], -n[0]]) * d
            Ft = np.clip(Kv * (par['v0'] - d * vy), -par['mu'] * Fn, par['mu'] * Fn)
            FD += Fn * n + Ft * t_; ND += Fn
        Ftot += FD; Mtot += dD[0] * FD[1] - dD[1] * FD[0]
        # タンデムローラー（左右2組 → 力2倍）
        dA = Rm @ rA; cA = cg + dA + np.array([0.0, u]); vA = vel(dA) + np.array([0.0, ud])
        FB = np.zeros(2); MB = 0.0; rolls = []
        for sgn in (-1, 1):
            e = rot(beta) @ np.array([sgn * pt, 0.0])
            c = cA + e; vc = vA + wb * np.array([-e[1], e[0]])
            Fr = np.zeros(2)
            for dl, n, p in ground_contacts(c, rr, ys, h, d):
                Fr += 2 * contact_force(dl, -vc @ n, par['K_roll'], par['alpha']) * n
            FB += Fr; MB += e[0] * Fr[1] - e[1] * Fr[0]; rolls.append(Fr / 2)
        Fs = kf * u + cf * ud                       # フォークばね力（車体を上へ, 先端を下へ）
        Fv = FB if rigid else np.array([FB[0], Fs])  # 車体が先端から受ける力（水平は剛結）
        Ftot += Fv; Mtot += dA[0] * Fv[1] - dA[1] * Fv[0]
        Ftip = FB[1] - Fs - mt * g
        return Ftot, Mtot, MB, FD, FB, rolls, ND, Ftip

    def rhs(t, x):
        Ftot, Mtot, MB, *_, Ftip = forces(t, x)
        az = Ftot[1] / M
        # 先端相対加速度 = 先端絶対加速度 - 車体A点加速度(鉛直, 回転分は微小なので並進+角加速度近似)
        aA = az + (Mtot / I) * (S['G']['A'][0] - S['Gc'][0])
        uacc = 0.0 if rigid else Ftip / mt - aA
        return [x[5], x[6], x[7], x[8], x[9], Ftot[0] / M, az, Mtot / I, MB / S['par']['I_bogie'], uacc]
    return rhs, forces


def static_eq(S, y0, d, ys):
    rhs, forces = make_rhs(S, d, ys, 0.0)
    rigid = S['par']['k_fork'] is None
    def f(q):
        x = [y0, q[0], q[1], q[2], 0.0 if rigid else q[3], 0, 0, 0, 0, 0]
        F, Mo, MB, *_, Ftip = forces(0, x)
        return [F[1] / 1e4, Mo / 1e3, MB / 10] + ([] if rigid else [Ftip / 1e4])
    u0 = 0.0 if rigid else 3000.0 / S['par']['k_fork']
    zA, zD = -0.0005 - u0, -0.001
    ph = (zA - zD) / (S['G']['A'][0] - S['G']['D'][0])
    q0 = [zD + ph * (S['Gc'][0] - S['G']['D'][0]), ph, 0.0] + ([] if rigid else [u0])
    q, info, ier, msg = fsolve(f, q0, xtol=1e-12, full_output=True)
    assert ier == 1 and max(abs(np.array(f(q)))) < 1e-6, msg
    return [y0, q[0], q[1], q[2], 0.0 if rigid else q[3], d * S['par']['v0'], 0, 0, 0, 0]



def run(case, par=PAR, save=True):
    S = build(par); G = S['G']
    d = 1 if case == 'fork_first' else -1
    v0 = par['v0']
    Kv = S['M'] / par['tau_v']
    # 段差位置: 先頭の車輪が 0.15 m 手前から走り出す
    if d == 1:
        lead = G['A'][0] + G['pitch'] * mm + G['r_roll'] * mm
        ys = lead + 0.15; y_end = G['D'][0] + G['r_drive'] * mm
        dist = ys - y_end + 0.10
    else:
        lead = G['D'][0] - G['r_drive'] * mm
        ys = lead - 0.15
        dist = (G['A'][0] + G['pitch'] * mm + G['r_roll'] * mm + 0.10) - ys
    T = dist / v0
    x0 = static_eq(S, 0.0, d, ys)
    rhs, forces = make_rhs(S, d, ys, Kv)
    te = np.arange(0, T, 2e-4)
    sol = solve_ivp(rhs, (0, T), x0, method='DOP853', t_eval=te, rtol=1e-7, atol=1e-10, max_step=2e-4)
    X = sol.y.T
    rec = dict(t=sol.t, x=X)
    FA, FDs, R1, R2, ND = [], [], [], [], []
    for t, x in zip(sol.t, X):
        Ftot, Mtot, MB, FD, FB, rolls, nd, _ = forces(t, x)
        FA.append(FB / 2); FDs.append(FD); R1.append(rolls[0]); R2.append(rolls[1]); ND.append(nd)
    rec.update(FA=np.array(FA), FD=np.array(FDs), Rrear=np.array(R1), Rfront=np.array(R2), ND=np.array(ND))
    # ---- 151000-01 の3ピン荷重（車体座標系, 片側1組） ----
    phi = X[:, 2]
    FAb = np.einsum('kij,kj->ki', np.array([rot(-p) for p in phi]), rec['FA'])   # 車体座標へ
    A, O, P, R = G['A'], G['O'], G['P'], G['R']
    e = (R - P) / np.linalg.norm(R - P)
    cr = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    Tpr = -cr(A - O, FAb) / cr(P - O, e)           # プッシュロッド軸力（+引張）
    FP = Tpr[:, None] * e[None, :]
    FO = -(FAb + FP)
    rec.update(FAb=FAb, FP=FP, FO=FO, Tpr=Tpr, ys=ys, d=d, h_step=par['h_step'], Gc=S['Gc'], M=S['M'], I=S['I'], cL=S['cL'],
               k_fork=-1.0 if par['k_fork'] is None else par['k_fork'])
    if save: np.savez(HERE / f'loads_{case}.npz', **{k: np.asarray(v) for k, v in rec.items()})
    st = FAb[0]
    print(f'[{case}] M={S["M"]:.0f}kg I={S["I"]:.1f}kgm2  CoG={S["Gc"]/mm}mm')
    print(f'  静的: 軸穴荷重 {np.hypot(*st):.0f} N  プッシュロッド {Tpr[0]:.0f} N  パイプ {np.hypot(*FO[0]):.0f} N  駆動輪 {ND[0]:.0f} N')
    k = np.argmax(np.hypot(*FAb.T))
    rec['summary'] = dict(static=float(np.hypot(*st)), peak=float(np.hypot(*FAb[k])), t_peak=float(sol.t[k]),
                          T_static=float(Tpr[0]), T_peak=float(Tpr.min()), FO_peak=float(np.hypot(*FO.T).max()))
    print(f'  最大: t={sol.t[k]:.3f}s 軸穴荷重 {np.hypot(*FAb[k]):.0f} N (Fy={FAb[k,0]:.0f}, Fz={FAb[k,1]:.0f})'
          f'  ロッド {Tpr[k]:.0f} N  パイプ {np.hypot(*FO[k]):.0f} N   速度最小 {np.min(d*X[:,5]):.3f} m/s')
    return rec


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    for c in ('fork_first', 'drive_first'):
        run(c)
    if '--sens' in sys.argv:          # フォーク剛性の感度
        res = {}
        for kf in (1.0e6, 2.0e6, 5.0e6, None):
            for c in ('fork_first', 'drive_first'):
                r = run(c, dict(PAR, k_fork=kf), save=False)
                res[f'{c}|{kf}'] = r['summary']
        json.dump(res, open(HERE / 'sensitivity.json', 'w'), indent=1)
