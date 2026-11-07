"""List every collision prim in Isaac Lab's go2.usd: owning rigid body, shape type, size, pose in the body frame.

Run from the repo root: PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python experiments/w3_unity/inspect_go2_collision.py
"""

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

path = UNITREE_GO2_CFG.spawn.usd_path
print("[usd]", path, flush=True)
stage = Usd.Stage.Open(path)
cache = UsdGeom.XformCache()


def owning_body(prim):
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


rows = []
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    body = owning_body(prim)
    bname = body.GetName() if body else "?"
    rel = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
    t = rel.ExtractTranslation()
    scale = Gf.Vec3d(*(rel.GetRow3(i).GetLength() for i in range(3)))
    tname = prim.GetTypeName()
    rot = rel.RemoveScaleShear().ExtractRotation()
    ax, ang = rot.GetAxis(), rot.GetAngle()
    rot_s = f"rot={ang:.2f}deg about ({ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f})"
    info = ""
    if tname == "Sphere":
        info = f"r={UsdGeom.Sphere(prim).GetRadiusAttr().Get():.4f}"
    elif tname in ("Cylinder", "Capsule"):
        g = UsdGeom.Cylinder(prim) if tname == "Cylinder" else UsdGeom.Capsule(prim)
        info = f"r={g.GetRadiusAttr().Get():.4f} h={g.GetHeightAttr().Get():.4f} axis={g.GetAxisAttr().Get()}"
    elif tname == "Cube":
        info = f"size={UsdGeom.Cube(prim).GetSizeAttr().Get():.4f}"
    elif tname == "Mesh":
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        bb = Gf.Range3d()
        for q in pts:
            bb.UnionWith(Gf.Vec3d(q))
        info = f"npts={len(pts)} bbox_min={tuple(round(v, 4) for v in bb.GetMin())} bbox_max={tuple(round(v, 4) for v in bb.GetMax())}"
        approx = prim.GetAttribute("physics:approximation")
        if approx and approx.Get():
            info += f" approx={approx.Get()}"
    print(
        f"[col] body={bname:12s} type={tname:8s} {info} {rot_s} "
        f"pos_in_body=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f}) scale=({scale[0]:.3f},{scale[1]:.3f},{scale[2]:.3f}) path={prim.GetPath()}",
        flush=True,
    )

app.close()
