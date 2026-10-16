"""タンデムローラー(ABM15M-160000-01) 段差乗り越えシミュレーション（側面図アニメーション）

前提:
  - 段差 H=25mm, 30度スロープ（水平43.3mm）で上段へ、スロープ上下の角部は R5 フィレット
  - ローラー Ø65 (r=32.5), ローラー間ピッチ 90mm, 揺動軸はローラー中心を結ぶ線の中点
  - 揺動軸は水平に等速移動し、上下は自由（荷重で両ローラーとも常に接地）
  - 滑りなし転がり。側板は 高さ45(±22.5) の帯と、Y±37 中心 R28 の端部円弧の交差形状（STEPから取得）
    下端の連結板用切欠き(Y±16〜19.5)は連結板161003(下端-22.2)で埋まるため無視
  - 段差は側板(X±34〜46)の下まで全幅にあるものとして干渉を判定
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
import sys, pathlib

R = 32.5          # ローラー半径
PITCH = 90.0      # ローラー間ピッチ
H = 25.0          # 段差
SLOPE_DEG = 30.0  # スロープ角度
FILLET_R = 5.0    # 角部フィレット
SLOPE_W = H / np.tan(np.radians(SLOPE_DEG))
PLATE_HALF = 37.0  # 側板長円 中心間 1/2
PLATE_R = 28.0     # 側板 端部R
PLATE_H = 22.5     # 側板 高さ1/2（全高45）
BOSS_R = 15.0

# 地面プロファイル（折れ線 + 角部フィレット）
S0 = 150.0  # スロープ開始位置（フィレット前の仮想交点）

def fillet(poly, r, n=24):
    out = [poly[0]]
    for a, p, b in zip(poly[:-2], poly[1:-1], poly[2:]):
        d1 = (p - a) / np.linalg.norm(p - a); d2 = (b - p) / np.linalg.norm(b - p)
        th = np.arccos(np.clip(d1 @ d2, -1, 1))
        t = r * np.tan(th / 2)
        turn = np.sign(d1[0] * d2[1] - d1[1] * d2[0])
        nrm = turn * np.array([-d1[1], d1[0]])
        c = p - t * d1 + r * nrm
        a0 = np.arctan2(*(p - t * d1 - c)[::-1]); a1 = np.arctan2(*(p + t * d2 - c)[::-1])
        ang = np.linspace(a0, a0 + turn * th, n)
        out += list(c + r * np.c_[np.cos(ang), np.sin(ang)])
    out.append(poly[-1])
    return np.array(out)

corners = np.array([[-300, 0], [S0, 0], [S0 + SLOPE_W, H], [700, H]], float)
ground = fillet(corners, FILLET_R)

def ground_z(s):
    return np.interp(s, ground[:, 0], ground[:, 1])

# 地面を細かくサンプリング（ローラー中心高さ・干渉計算用）
gs = np.linspace(ground[0, 0], ground[-1, 0], 200001)
gz = ground_z(gs)

def center_h(s):
    """ローラー中心が s にあるときの中心高さ（地面に接する最低位置）"""
    m = np.abs(gs - s) <= R
    return np.max(gz[m] + np.sqrt(R**2 - (gs[m] - s)**2))

# 前ローラー位置でスイープ → 後ローラーをピッチ拘束で求める
s1_list = np.arange(-60, 440, 0.25)
rows = []
for s1 in s1_list:
    h1 = center_h(s1)
    lo, hi = s1 - PITCH, s1
    for _ in range(50):
        mid = (lo + hi) / 2
        f = (s1 - mid)**2 + (h1 - center_h(mid))**2 - PITCH**2
        lo, hi = (mid, hi) if f > 0 else (lo, mid)
    s2 = (lo + hi) / 2
    rows.append((s1, h1, s2, center_h(s2)))
rows = np.array(rows)
piv_s = (rows[:, 0] + rows[:, 2]) / 2
piv_z = (rows[:, 1] + rows[:, 3]) / 2
tilt = np.degrees(np.arctan2(rows[:, 1] - rows[:, 3], rows[:, 0] - rows[:, 2]))

# 揺動軸の水平位置で等間隔に再サンプリング（等速移動）
N = 240
ps = np.linspace(piv_s[0], piv_s[-1], N)
def rs(a): return np.interp(ps, piv_s, a)
F = dict(s1=rs(rows[:, 0]), h1=rs(rows[:, 1]), s2=rs(rows[:, 2]), h2=rs(rows[:, 3]),
         pz=rs(piv_z), tilt=rs(tilt))

# ローラー回転角 = 中心軌跡長 / R
def roll_angle(s, h):
    d = np.r_[0, np.cumsum(np.hypot(np.diff(s), np.diff(h)))]
    return -d / R
F['a1'] = roll_angle(F['s1'], F['h1'])
F['a2'] = roll_angle(F['s2'], F['h2'])

# 側板（長円）と地面のクリアランス：地面点から長円までの符号付き距離の最小値
def plate_clearance(i):
    c, sn = np.cos(np.radians(F['tilt'][i])), np.sin(np.radians(F['tilt'][i]))
    m = np.abs(gs - ps[i]) < 120
    dx, dz = gs[m] - ps[i], gz[m] - F['pz'][i]
    u = dx * c + dz * sn          # 側板の長手方向
    v = -dx * sn + dz * c
    uc = np.clip(u, -PLATE_HALF, PLATE_HALF)
    d = np.maximum(np.hypot(u - uc, v) - PLATE_R, np.abs(v) - PLATE_H)
    k = np.argmin(d)
    return d[k], gs[m][k], gz[m][k]
clr = np.array([plate_clearance(i) for i in range(N)])
F['clr'] = clr[:, 0]

print(f'揺動軸 上昇量: {F["pz"].max() - F["pz"].min():.2f} mm')
print(f'最大傾斜角: {F["tilt"].max():.2f} deg')
print(f'側板-地面 最小クリアランス: {F["clr"].min():.2f} mm  (平地 {F["clr"][0]:.2f} mm)')
bad = F['clr'] < 0
if bad.any():
    print(f'干渉あり: 揺動軸位置 s={ps[bad][0]-S0:.1f}〜{ps[bad][-1]-S0:.1f} mm（スロープ開始基準）, 最大食い込み {-F["clr"].min():.2f} mm')

# ---------------- 描画 ----------------
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
fig = plt.figure(figsize=(10, 5.6), dpi=100)
ax = fig.add_axes([0.04, 0.38, 0.92, 0.56])
axg = fig.add_axes([0.07, 0.09, 0.86, 0.25])
ax.set_aspect('equal'); ax.set_xlim(40, 540); ax.set_ylim(-20, 105); ax.axis('off')
ax.fill(np.r_[ground[:, 0], ground[-1, 0], ground[0, 0]], np.r_[ground[:, 1], -40, -40],
        color='#c9c4b8', ec='#555', lw=1.2, zorder=1)
ax.set_title(f'ABM15M-160000-01  段差{H:.0f}mm（{SLOPE_DEG:.0f}°スロープ・角R{FILLET_R:.0f}）乗り越え　側面図', fontsize=12)

def plate_poly(i):
    ta = np.arcsin(PLATE_H / PLATE_R)
    t = np.linspace(-ta, ta, 40)
    u = np.r_[PLATE_HALF + PLATE_R * np.cos(t), -PLATE_HALF - PLATE_R * np.cos(t)]
    v = np.r_[PLATE_R * np.sin(t), -PLATE_R * np.sin(t)]
    c, sn = np.cos(np.radians(F['tilt'][i])), np.sin(np.radians(F['tilt'][i]))
    return np.c_[ps[i] + u * c - v * sn, F['pz'][i] + u * sn + v * c]

plate = Polygon(plate_poly(0), closed=True, fc='#6a8caf', ec='#24425f', alpha=0.55, lw=1.2, zorder=4)
ax.add_patch(plate)
boss = Circle((0, 0), BOSS_R, fc='#d9d9d9', ec='#333', zorder=5); ax.add_patch(boss)
pivot = Circle((0, 0), 3, fc='k', zorder=6); ax.add_patch(pivot)
rollers = []
for _ in range(2):
    outer = Circle((0, 0), R, fc='#8a8f96', ec='#222', lw=1.2, zorder=2)
    brg = Circle((0, 0), 21, fc='#b7bcc2', ec='#333', lw=0.8, zorder=3)
    shaft = Circle((0, 0), 10, fc='#e8e8e8', ec='#333', lw=0.8, zorder=6)
    for p in (outer, brg, shaft): ax.add_patch(p)
    spokes, = ax.plot([], [], color='#d33', lw=2, zorder=3)
    rollers.append((outer, brg, shaft, spokes))
trail, = ax.plot([], [], color='k', lw=0.8, ls='--', zorder=7)
hit, = ax.plot([], [], 'o', ms=10, mfc='none', mec='red', mew=2.5, zorder=8)
txt = ax.text(45, 100, '', fontsize=10, va='top', family=plt.rcParams['font.family'])

x_mm = ps - S0
axg.plot(x_mm, F['pz'] - F['pz'][0], color='#24425f', label='揺動軸 上昇量 [mm]')
axg.plot(x_mm, F['tilt'], color='#c07020', label='傾斜角 [deg]')
axg.plot(x_mm, F['clr'], color='#2a8a3a', label='側板-地面 クリアランス [mm]')
axg.axhline(0, color='red', lw=0.8)
axg.set_xlabel('揺動軸の水平位置（スロープ開始基準）[mm]'); axg.grid(alpha=0.3)
axg.legend(loc='upper left', fontsize=8, ncol=3)
cursor = axg.axvline(x_mm[0], color='k', lw=1)

def draw(i):
    for (outer, brg, shaft, spokes), s, h, a in ((rollers[0], F['s1'][i], F['h1'][i], F['a1'][i]),
                                                 (rollers[1], F['s2'][i], F['h2'][i], F['a2'][i])):
        for p in (outer, brg, shaft): p.center = (s, h)
        t = a + np.array([0, np.pi, np.nan, np.pi / 2, 3 * np.pi / 2])
        spokes.set_data(s + 29 * np.cos(t), h + 29 * np.sin(t))
    plate.set_xy(plate_poly(i))
    boss.center = pivot.center = (ps[i], F['pz'][i])
    trail.set_data(ps[:i + 1], F['pz'][:i + 1])
    c = F['clr'][i]
    if c < 0:
        hit.set_data([clr[i, 1]], [clr[i, 2]]); plate.set_facecolor('#d9534f')
    else:
        hit.set_data([], []); plate.set_facecolor('#6a8caf')
    txt.set_text(f'揺動軸 上昇 {F["pz"][i]-F["pz"][0]:5.1f} mm   傾斜 {F["tilt"][i]:5.1f}°   '
                 f'側板クリアランス {c:5.1f} mm' + ('  ← 干渉' if c < 0 else ''))
    cursor.set_xdata([x_mm[i]])
    return []

out = pathlib.Path(__file__).with_name('step_climb.gif')
anim = animation.FuncAnimation(fig, draw, frames=N, interval=40)
anim.save(out, writer=animation.PillowWriter(fps=25))
draw(int(np.argmin(F['clr']))); fig.savefig(out.with_name('step_climb_worst.png'))
print('saved', out)
