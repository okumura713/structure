"""151000-01(ビードあり) をSTEPから抽出し、2次四面体メッシュを作成して mesh.npz に保存
使い方: python mesh_part.py <全体サイズ mm> <溶接すき間許容差 mm> <穴まわり局所サイズ mm>"""
import gmsh, numpy as np, sys, pathlib
HERE = pathlib.Path(__file__).parent
STEP = HERE.parents[1] / '00_sketch' / 'ABM_Assy.stp'
H = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(str(STEP)); gmsh.model.occ.synchronize()
keep = []
for d, t in gmsh.model.getEntities(3):
    n = gmsh.model.getEntityName(d, t)
    if '151000-01ビードあり' in n: keep.append((d, t)); print(t, n.split('/')[-1], gmsh.model.occ.getMass(3, t))
gmsh.model.occ.remove([e for e in gmsh.model.getEntities(3) if e not in keep], recursive=True)
gmsh.model.occ.synchronize()
# 溶接（ビード含む）で一体化：CAD上のビード-パイプ間すき間(約0.34mm)を許容差で吸収して接触面を共有化
gmsh.option.setNumber("Geometry.ToleranceBoolean", float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
out, _ = gmsh.model.occ.fragment(keep[:1], keep[1:]); gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3); print('volumes after fragment', vols)
mat = {}
for d, t in vols:
    bb = gmsh.model.getBoundingBox(d, t)
    mat[t] = 1 if (bb[3] - bb[0]) > 90 else 0      # 1: パイプ(STKM13A), 0: 側板(SS400)
HF = float(sys.argv[3]) if len(sys.argv) > 3 else H / 3      # 穴・溶接部まわりの局所サイズ
gmsh.option.setNumber("Mesh.MeshSizeMax", H); gmsh.option.setNumber("Mesh.MeshSizeMin", HF)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
# 局所細分化: ピン穴(φ20.1 x2, パイプ)の円筒面と、パイプ外周(溶接部)からの距離で細かくする
cyl = []
for d, t in gmsh.model.getEntities(2):
    if gmsh.model.getType(d, t) != 'Cylinder': continue
    bb = np.array(gmsh.model.getBoundingBox(d, t)); c = (bb[1:3] + bb[4:6]) / 2; r = (bb[4:6] - bb[1:3]).max() / 2
    for cc in ((-41.4, 179.0), (0.0, 32.5), (-55.5, 156.2)):
        if np.hypot(*(c - cc)) < 3 and 9 < r < 17: cyl.append(t)
f1 = gmsh.model.mesh.field.add("Distance"); gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", cyl)
f2 = gmsh.model.mesh.field.add("Threshold"); gmsh.model.mesh.field.setNumber(f2, "InField", f1)
for k, v in (("SizeMin", HF), ("SizeMax", H), ("DistMin", 1.5), ("DistMax", 12.0)):
    gmsh.model.mesh.field.setNumber(f2, k, v)
gmsh.model.mesh.field.setAsBackgroundMesh(f2)
gmsh.option.setNumber("Mesh.ElementOrder", 2); gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.model.mesh.generate(3)
nt, xyz, _ = gmsh.model.mesh.getNodes(); xyz = xyz.reshape(-1, 3)
idx = np.zeros(nt.max() + 1, int); idx[nt] = np.arange(len(nt))
T, M = [], []
for d, t in vols:
    et, _, nd = gmsh.model.mesh.getElements(3, t)
    for ty, n in zip(et, nd):
        if ty == 11: e = idx[np.array(n).reshape(-1, 10)]; T.append(e); M.append(np.full(len(e), mat[t]))
T = np.vstack(T); M = np.concatenate(M)
print('nodes', len(xyz), 'tet10', len(T))
np.savez(HERE / 'mesh.npz', xyz=xyz, tet=T, mat=M)
