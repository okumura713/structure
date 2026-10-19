"""151000-01 段差乗り越え 応力アニメーション
左: 151000-01 のミーゼス応力コンター（2視点, 変形拡大表示）
右: 車両の側面図（全体 / 段差まわり拡大）と応力・荷重の時刻歴
使い方: python animate.py [fork_first|drive_first] [--test]
"""
import numpy as np, pathlib, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Polygon, Rectangle
import imageio_ffmpeg

HERE = pathlib.Path(__file__).parent
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
SY = 245.0          # SS400 降伏点 [MPa]
DEF_SCALE = 100     # 変形表示倍率
CASE_JP = {'fork_first': 'フォーク先行（従動ローラーが先に段差へ進入）',
           'drive_first': '駆動輪先行（従動ローラーは最後に段差を越える）'}

case = sys.argv[1] if len(sys.argv) > 1 else 'drive_first'
F = np.load(HERE / f'fem_{case}.npz')
L = np.load(HERE / f'loads_{case}.npz')
X, tri, VM, U = F['X'], F['tri'], F['VM'], F['U']
nf = VM.shape[1]
tF = F['t']
vmax_case = float(VM.max())
hot = int(np.argmax(VM.max(1)))                      # 最大応力節点
k_peak = int(np.argmax(VM[hot]))

# ---------------- 3D 投影レンダラ ----------------
def view_mat(az, el):
    a, e = np.radians(az), np.radians(el)
    f = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # 視線（物体→カメラ）
    r = np.cross([0, 0, 1], f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.array([r, u, f])

class View:
    def __init__(self, ax, az, el, title, zoom=None):
        self.R = view_mat(az, el)
        P = X @ self.R.T
        depth = P[tri, 2].mean(1)
        self.order = np.argsort(depth)
        self.tri = tri[self.order]
        n = np.cross(X[self.tri[:, 1]] - X[self.tri[:, 0]], X[self.tri[:, 2]] - X[self.tri[:, 0]])
        n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
        light = self.R.T @ np.array([-0.3, 0.5, 1.0]); light /= np.linalg.norm(light)
        self.shade = 0.45 + 0.55 * np.abs(n @ light)
        self.ax = ax
        self.pc = PolyCollection(P[self.tri][:, :, :2], edgecolors='none', antialiaseds=False)
        ax.add_collection(self.pc)
        lo, hi = P[:, :2].min(0), P[:, :2].max(0); c = (lo + hi) / 2; s = (hi - lo).max() / 2 * 1.08
        if zoom: c = P[hot, :2]; s = zoom
        ax.set_xlim(c[0] - s, c[0] + s); ax.set_ylim(c[1] - s, c[1] + s); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=10)
        self.mk, = ax.plot([], [], 'o', ms=13, mfc='none', mec='k', mew=2)
        self.mt = ax.text(0, 0, '', fontsize=9, fontweight='bold', ha='left', va='bottom',
                          bbox=dict(fc='white', ec='k', alpha=0.8, pad=2))

    def draw(self, k, norm, cmap):
        Xd = X + DEF_SCALE * U[:, :, k]
        P = Xd @ self.R.T
        self.pc.set_verts(P[self.tri][:, :, :2])
        c = cmap(norm(VM[self.tri, k].mean(1)))
        c[:, :3] *= self.shade[:, None]
        self.pc.set_facecolor(c)
        h = P[hot]
        self.mk.set_data([h[0]], [h[1]])
        self.mt.set_position((h[0] + 8, h[1] + 8)); self.mt.set_text(f'{VM[hot, k]:.0f} MPa')

# ---------------- 車両 側面図 ----------------
mm = 1e3
Gc = L['Gc'] * mm; xs = L['x']; tL = L['t']; ys = float(L['ys']) * mm; dirn = int(L['d'])
H_STEP = 6.0
def state(t):
    return np.array([np.interp(t, tL, xs[:, j]) for j in range(xs.shape[1])])
def to_world(p, s):
    yg, zg, phi = s[0] * mm, s[1] * mm, s[2]
    c, sn = np.cos(phi), np.sin(phi)
    q = np.atleast_2d(p) - Gc
    return np.c_[Gc[0] + yg + c * q[:, 0] - sn * q[:, 1], Gc[1] + zg + sn * q[:, 0] + c * q[:, 1]]

A0, O0, P0, R0, D0 = map(np.array, [(0, 32.5), (-55.5, 156.2), (-41.4, 179.0), (-870.8, 170.9), (-1223.3, 75.0)])
SHAPES = dict(
    unit=np.array([(-1430, 25), (-1020, 25), (-1020, 180), (-1430, 180)]),
    mast=np.array([(-1104, 184), (-835, 184), (-835, 802), (-1104, 802)]),
    pallet=np.array([(-835, 201), (265, 201), (265, 345), (-835, 345)]),
    goods=np.array([(-815, 345), (245, 345), (245, 1201), (-815, 1201)]),
)
FORK = np.array([(-835, 145.0), (101, 150.0), (235, 185.0), (235, 195.0), (101, 201.0), (-835, 201.0)])

def ground_xy(y0, y1):
    if dirn == 1: return np.array([(y0, -60), (y0, 0), (ys, 0), (ys, H_STEP), (y1, H_STEP), (y1, -60)])
    return np.array([(y0, -60), (y0, H_STEP), (ys, H_STEP), (ys, 0), (y1, 0), (y1, -60)])

class Side:
    def __init__(self, ax, xlim, ylim, zoom=False, follow=False):
        self.ax = ax; ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
        self.xlim0 = np.array(xlim); self.follow = follow
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Polygon(ground_xy(xlim[0] - 3000, xlim[1] + 3000), fc='#cfc8b8', ec='#555', lw=1.2, zorder=1))
        if follow:
            ax.annotate('段差 6mm', (ys, H_STEP), (ys, -35), ha='center', fontsize=8, color='#a00',
                        arrowprops=dict(arrowstyle='->', color='#a00'), zorder=8)
        self.p = {}
        for k, fc in (('unit', '#9aa5b1'), ('mast', '#b8c2cc'), ('pallet', '#c8a26b'), ('goods', '#e8d9b0')):
            self.p[k] = ax.add_patch(Polygon(SHAPES[k], fc=fc, ec='#444', lw=0.8, alpha=0.9, zorder=2))
        self.fork = ax.add_patch(Polygon(FORK, fc='#7d8a96', ec='#333', lw=0.8, zorder=3))
        self.rod, = ax.plot([], [], color='#556', lw=3 if zoom else 1.5, zorder=4)
        self.w = ax.add_patch(Polygon(np.zeros((3, 2)), fc='red', ec='k', lw=1.0, zorder=6))
        self.wheels = []
        for r in (75.0, 32.5, 32.5):
            c = ax.add_patch(Circle((0, 0), r, fc='#3a3f45', ec='k', lw=0.8, zorder=5))
            hub = ax.add_patch(Circle((0, 0), r * 0.55, fc='#9ca3ab', ec='k', lw=0.5, zorder=5))
            sp, = ax.plot([], [], color='#e33', lw=1.5 if zoom else 0.8, zorder=5)
            self.wheels.append((c, hub, sp, r))
        self.bog, = ax.plot([], [], color='#333', lw=4 if zoom else 2, zorder=5)
        self.dist = np.zeros(3)

    def draw(self, t, s, wcol):
        if self.follow: self.ax.set_xlim(*(self.xlim0 + s[0] * mm))
        for k in SHAPES: self.p[k].set_xy(to_world(SHAPES[k], s))
        u = s[4] * mm
        f = FORK.copy(); f[:, 1] += u * np.clip((f[:, 0] + 835) / (A0[0] + 835), 0, 1.2)
        self.fork.set_xy(to_world(f, s))
        Aw = to_world(A0, s)[0] + [0, u]; Ow = to_world(O0, s)[0] + [0, u]; Pw = to_world(P0, s)[0] + [0, u]
        Rw = to_world(R0, s)[0]
        self.rod.set_data([Pw[0], Rw[0]], [Pw[1], Rw[1]])
        self.w.set_xy(np.array([Ow, Pw, Aw])); self.w.set_facecolor(wcol)
        b = s[3]; e = np.array([np.cos(b), np.sin(b)]) * 45
        cs = [to_world(D0, s)[0], Aw - e, Aw + e]
        self.bog.set_data([cs[1][0], cs[2][0]], [cs[1][1], cs[2][1]])
        for (c, hub, sp, r), cc in zip(self.wheels, cs):
            c.center = hub.center = cc
            a = -(cc[0] - 0) / r * dirn
            ang = a + np.array([0, np.pi, np.nan, np.pi / 2, 3 * np.pi / 2])
            sp.set_data(cc[0] + 0.9 * r * np.cos(ang), cc[1] + 0.9 * r * np.sin(ang))

# ---------------- 図 ----------------
cmap = matplotlib.colormaps['jet']
norm = colors.Normalize(0, vmax_case)
fig = plt.figure(figsize=(16, 9), dpi=100)
fig.suptitle(f'151000-01 ホイールフォークペア　段差6mm乗り越え時の応力（ミーゼス）  ― {CASE_JP[case]}', fontsize=14, y=0.985)
ax1 = fig.add_axes([0.0, 0.33, 0.29, 0.60]); ax2 = fig.add_axes([0.27, 0.33, 0.29, 0.60])
v1 = View(ax1, -20, 25, '外側から（右側板）')
v2 = View(ax2, 200, 25, '最大応力部の拡大（プッシュロッド穴, 内側・後方から）', zoom=38)
ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes, fill=False, ec='#888', lw=1))
cax = fig.add_axes([0.06, 0.30, 0.44, 0.022])
cb = fig.colorbar(cm.ScalarMappable(norm, cmap), cax=cax, orientation='horizontal')
cb.set_label('ミーゼス相当応力 [MPa]（赤 = この走行パターンでの最大応力）', fontsize=9)
cax.axvline(SY, color='k', lw=2); cax.text(SY, 1.6, f'降伏点 SS400 {SY:.0f}', ha='center', fontsize=8, transform=cax.get_xaxis_transform())
axs = fig.add_axes([0.575, 0.60, 0.415, 0.33])
side = Side(axs, (-1520, 420), (-60, 1260), follow=True)
axs.set_title('車両側面（荷物1500kg・55m/min）', fontsize=10)
axz = fig.add_axes([0.575, 0.33, 0.415, 0.25])
zoom = Side(axz, (ys - 340, ys + 340), (-12, 220), zoom=True)
axz.set_title('段差まわり拡大（段差 6mm）', fontsize=10)
axh = fig.add_axes([0.06, 0.06, 0.92, 0.17])
vm_hist = VM.max(0)
axh.plot(tF, vm_hist, color='#c0392b', lw=1.8, label='151000-01 最大ミーゼス [MPa]')
axh.axhline(SY, color='k', ls='--', lw=1); axh.text(tF[0] + 0.01, SY + 8, '降伏点 245 MPa', fontsize=8)
axh.set_ylabel('応力 [MPa]'); axh.set_xlabel('時間 [s]'); axh.grid(alpha=0.3)
axh.set_xlim(tF[0], tF[-1]); axh.set_ylim(0, vmax_case * 1.12)
axh2 = axh.twinx()
axh2.plot(tL, np.hypot(*L['FAb'].T) / 1e3, color='#2471a3', lw=1, label='ローラー軸荷重 [kN]')
axh2.plot(tL, -L['Tpr'] / 1e3, color='#7d3c98', lw=1, label='プッシュロッド圧縮力 [kN]')
axh2.set_ylabel('荷重 [kN]'); axh2.set_ylim(0, max(-L['Tpr'].min(), 1) / 1e3 * 1.12)
h1, l1 = axh.get_legend_handles_labels(); h2, l2 = axh2.get_legend_handles_labels()
axh.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8, ncol=3)
cur = axh.axvline(tF[0], color='k', lw=1)
info = fig.text(0.01, 0.955, '', fontsize=10, va='top', family=plt.rcParams['font.family'])

def draw(k):
    t = tF[k]; s = state(t)
    v1.draw(k, norm, cmap); v2.draw(k, norm, cmap)
    wcol = cmap(norm(VM[:, k].max()))
    side.draw(t, s, wcol); zoom.draw(t, s, wcol)
    cur.set_xdata([t])
    i = int(np.argmin(np.abs(tL - t)))
    info.set_text(f't = {t:5.3f} s    最大応力 {VM[:, k].max():5.0f} MPa（安全率 {SY / VM[:, k].max():4.2f}）    '
                  f'ローラー軸荷重 {np.hypot(*L["FAb"][i]) / 1e3:5.1f} kN    プッシュロッド {-L["Tpr"][i] / 1e3:5.1f} kN(圧縮)    '
                  f'変形表示 x{DEF_SCALE}')
    return []

if '--test' in sys.argv:
    draw(k_peak); fig.savefig(HERE / f'test_{case}.png'); sys.exit()
# ピーク時刻の静止画
draw(k_peak); fig.savefig(HERE / f'stress_peak_{case}.png')
draw(0); fig.savefig(HERE / f'stress_static_{case}.png')
anim = animation.FuncAnimation(fig, draw, frames=nf, interval=60)
anim.save(HERE / f'stress_{case}.mp4', writer=animation.FFMpegWriter(fps=16, bitrate=4000))
print('saved', case, 'peak', vmax_case, 'at t', tF[k_peak], 'node', X[hot])
