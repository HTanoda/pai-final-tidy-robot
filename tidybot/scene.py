"""MuJoCo シーン構築：Franka Panda + テーブル + 生活物体 + 片付け先の箱。

MjSpec でシーンをプログラム的に組み立てる。物体の種類・配置・色・照明は
乱数で変えられるので、視覚モデル用の合成データ生成にも同じコードを使う。
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

# ---------------------------------------------------------------- 物体カタログ
# class_id 0 は背景（テーブル・ロボット・箱・未知物体）
@dataclasses.dataclass(frozen=True)
class ObjSpec:
    name: str        # 英語ID（LLM/コード用）
    ja: str          # 日本語名（動画/レポート用）
    shape: str       # sphere / box / cylinder / capsule
    size: tuple      # MuJoCo geom size
    rgba: tuple
    mass: float


CATALOG: list[ObjSpec] = [
    ObjSpec("apple", "りんご", "sphere", (0.032,), (0.80, 0.08, 0.10, 1), 0.08),
    ObjSpec("orange", "みかん", "sphere", (0.034,), (1.00, 0.50, 0.05, 1), 0.08),
    ObjSpec("banana", "バナナ", "capsule", (0.018, 0.045), (0.95, 0.85, 0.15, 1), 0.06),
    ObjSpec("can", "空き缶", "cylinder", (0.030, 0.050), (0.72, 0.74, 0.78, 1), 0.03),
    ObjSpec("bottle", "ペットボトル", "cylinder", (0.026, 0.070), (0.35, 0.75, 0.85, 1), 0.04),
    ObjSpec("red_block", "赤い積み木", "box", (0.024, 0.024, 0.024), (0.85, 0.12, 0.12, 1), 0.05),
    ObjSpec("blue_block", "青い積み木", "box", (0.024, 0.024, 0.024), (0.15, 0.30, 0.85, 1), 0.05),
    ObjSpec("yellow_block", "黄色い積み木", "box", (0.024, 0.024, 0.024), (0.95, 0.80, 0.10, 1), 0.05),
]
CLASS_NAMES = ["background"] + [o.name for o in CATALOG]
CLASS_JA = ["背景"] + [o.ja for o in CATALOG]
NUM_CLASSES = len(CLASS_NAMES)
BY_NAME = {o.name: o for o in CATALOG}

# ---------------------------------------------------------------- 片付け先
@dataclasses.dataclass(frozen=True)
class BinSpec:
    name: str
    ja: str
    pos: tuple       # 中心 (x, y)
    rgba: tuple


BINS: list[BinSpec] = [
    BinSpec("toy_box", "おもちゃ箱", (0.28, 0.45), (0.30, 0.55, 0.90, 1)),
    BinSpec("fruit_basket", "果物かご", (0.56, 0.45), (0.70, 0.50, 0.28, 1)),
    BinSpec("recycle_bin", "資源ごみ箱", (0.42, -0.47), (0.25, 0.60, 0.35, 1)),
]
BIN_BY_NAME = {b.name: b for b in BINS}
BIN_HALF = 0.085      # 箱の内寸の半分
BIN_WALL_H = 0.075

# 物体を置く領域（テーブル上、ロボット正面）
SPAWN_X = (0.38, 0.68)
SPAWN_Y = (-0.24, 0.24)

OVERHEAD_CAM = "overhead"
FRONT_CAM = "front"


def _lookat_xyaxes(pos, target, up=(0, 0, 1)):
    pos, target, up = map(np.asarray, (pos, target, up))
    f = target - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return list(x) + list(y)


def _quat_z(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


@dataclasses.dataclass
class Placement:
    obj: ObjSpec
    x: float
    y: float
    yaw: float


def sample_placements(rng: np.random.Generator, names: list[str] | None = None,
                      n: int | None = None, min_dist: float = 0.095) -> list[Placement]:
    """物体の配置をランダムにサンプルする（互いに min_dist 以上離す）。"""
    if names is None:
        n = n or int(rng.integers(3, 7))
        names = list(rng.choice([o.name for o in CATALOG], size=n, replace=False))
    out: list[Placement] = []
    for name in names:
        for _ in range(500):
            x, y = rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)
            if all(math.hypot(x - p.x, y - p.y) > min_dist for p in out):
                break
        out.append(Placement(BY_NAME[name], x, y, float(rng.uniform(-math.pi, math.pi))))
    return out


def _geom_type(shape):
    return {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}[shape]


def _rest_height(o: ObjSpec):
    if o.shape == "sphere":
        return o.size[0]
    if o.shape == "box":
        return o.size[2]
    if o.shape == "cylinder":
        return o.size[1]
    return o.size[0]  # capsule（横倒し）


def build_spec(placements: list[Placement], distractors: int = 0,
               rng: np.random.Generator | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.quality.shadowsize = 4096
    wb = spec.worldbody
    # 指先中心（Tool Center Point）
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0])

    # 床・テーブル
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.55, 0.55, 0.58], rgb2=[0.48, 0.48, 0.52], width=300, height=300)
    spec.add_material(name="floor_mat", textures=["", "floor_tex"], texrepeat=[8, 8],
                      texuniform=True, reflectance=0.05)
    spec.add_material(name="table_mat", rgba=[0.72, 0.60, 0.47, 1], reflectance=0.02,
                      specular=0.1, shininess=0.1)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                pos=[0, 0, -0.75], material="floor_mat")
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, 0.75, 0.375],
                pos=[0.35, 0, -0.375], material="table_mat", friction=[1.0, 0.005, 0.0001])

    wb.add_light(name="key", pos=[0.6, -0.6, 2.2], dir=[-0.15, 0.3, -1], type=mujoco.mjtLightType.mjLIGHT_SPOT,
                 diffuse=[0.45, 0.45, 0.45], specular=[0.05, 0.05, 0.05], castshadow=True,
                 attenuation=[1, 0, 0], cutoff=70, exponent=1)
    wb.add_light(name="fill", pos=[-0.5, 0.5, 2.0], dir=[0.3, -0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.25, 0.25, 0.25], specular=[0, 0, 0], castshadow=False)

    # カメラ
    wb.add_camera(name=OVERHEAD_CAM, pos=[0.45, 0.0, 1.30], xyaxes=[0, -1, 0, 1, 0, 0], fovy=48)
    wb.add_camera(name=FRONT_CAM, pos=[1.25, -0.80, 0.80],
                  xyaxes=_lookat_xyaxes([1.25, -0.80, 0.80], [0.40, 0.0, 0.05]), fovy=45)
    wb.add_camera(name="side", pos=[0.45, 1.35, 0.75],
                  xyaxes=_lookat_xyaxes([0.45, 1.35, 0.75], [0.45, 0.0, 0.05]), fovy=50)

    # 片付け先の箱（上面が開いた箱を5枚の板で作る）
    t = 0.006
    for b in BINS:
        body = wb.add_body(name=f"bin_{b.name}", pos=[b.pos[0], b.pos[1], 0])
        body.add_geom(name=f"bin_{b.name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BIN_HALF + t, BIN_HALF + t, t / 2], pos=[0, 0, t / 2], rgba=b.rgba)
        for i, (dx, dy, sx, sy) in enumerate([(1, 0, t, BIN_HALF + t), (-1, 0, t, BIN_HALF + t),
                                              (0, 1, BIN_HALF + t, t), (0, -1, BIN_HALF + t, t)]):
            body.add_geom(name=f"bin_{b.name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[sx, sy, BIN_WALL_H / 2],
                          pos=[dx * (BIN_HALF + t), dy * (BIN_HALF + t), BIN_WALL_H / 2],
                          rgba=b.rgba)

    # 物体
    for i, p in enumerate(placements):
        o = p.obj
        z = _rest_height(o) + 0.002
        quat = _quat_z(p.yaw)
        if o.shape == "capsule":  # 横倒し（カプセル軸を x 方向へ）
            quat = _quat_mul(quat, [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
        body = wb.add_body(name=f"obj{i}", pos=[p.x, p.y, z], quat=quat)
        body.add_freejoint(name=f"obj{i}_joint")
        size = list(o.size) + [0] * (3 - len(o.size))
        body.add_geom(name=f"obj{i}_{o.name}", type=_geom_type(o.shape), size=size,
                      rgba=list(o.rgba), mass=o.mass, condim=6,
                      friction=[1.5, 0.01, 0.002], solref=[0.01, 1], priority=1)

    # 未知物体（学習用：背景として扱う紛らわしい物）
    if distractors and rng is not None:
        for j in range(distractors):
            x, y = rng.uniform(0.2, 0.75), rng.uniform(-0.35, 0.35)
            s = rng.uniform(0.015, 0.035, size=3)
            body = wb.add_body(name=f"dis{j}", pos=[x, y, s[2] + 0.002],
                               quat=_quat_z(rng.uniform(-math.pi, math.pi)))
            body.add_freejoint()
            kind = rng.choice(["box", "cylinder", "ellipsoid"])
            gt = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                  "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[kind]
            body.add_geom(name=f"dis{j}_geom", type=gt, size=list(s),
                          rgba=list(rng.uniform(0.1, 0.9, 3)) + [1], mass=0.05)
    return spec


def geom_class_lut(model: mujoco.MjModel) -> np.ndarray:
    """geom id -> class id の対応表（セグメンテーションの教師ラベル用）。"""
    lut = np.zeros(model.ngeom + 1, dtype=np.int64)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("obj"):
            cls = name.split("_", 1)[1]
            lut[g] = CLASS_NAMES.index(cls)
    return lut


HOME_Q = np.array([0.0, -0.5, 0.0, -2.3, 0.0, 1.85, 0.785])
# 俯瞰カメラの視界からアームを外す観測姿勢
OBSERVE_Q = np.array([0.0, -1.76, 0.0, -2.4, 0.0, 0.8, 0.785])


def reset(model: mujoco.MjModel, data: mujoco.MjData, q=OBSERVE_Q, gripper=255.0):
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = q
    data.qpos[7:9] = 0.04
    data.ctrl[:7] = q
    data.ctrl[7] = gripper
    mujoco.mj_forward(model, data)
