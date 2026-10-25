"""視覚モデル用の合成データ生成（ドメインランダム化つき）。

MuJoCo のセグメンテーション描画から画素ごとの正解ラベルを自動で作るので、
人手のアノテーションは不要。

    python -m tidybot.perception.dataset --out data/seg --n 4000 --workers 4
"""
from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

from .. import scene as S
from .camera import IMG


def random_scene(rng: np.random.Generator):
    n = int(rng.integers(2, 9))
    names = list(rng.choice([o.name for o in S.CATALOG], size=n, replace=False))
    placements = S.sample_placements(rng, names, min_dist=0.075)
    # 一部は箱の中や作業領域の外側にも置く
    for p in placements:
        r = rng.random()
        if r < 0.15:
            b = S.BINS[rng.integers(len(S.BINS))]
            p.x = b.pos[0] + rng.uniform(-0.05, 0.05)
            p.y = b.pos[1] + rng.uniform(-0.05, 0.05)
        elif r < 0.25:
            p.x, p.y = rng.uniform(0.25, 0.8), rng.uniform(-0.35, 0.35)
    spec = S.build_spec(placements, distractors=int(rng.integers(0, 4)), rng=rng)
    model = spec.compile()
    return model, placements


def randomize_appearance(m: mujoco.MjModel, rng: np.random.Generator):
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(("obj", "bin_")):
            m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] + rng.normal(0, 0.06, 3), 0, 1)
    tm = m.material("table_mat").id
    m.mat_rgba[tm, :3] = np.clip(np.array([0.72, 0.60, 0.47]) + rng.normal(0, 0.08, 3), 0, 1)
    m.vis.headlight.ambient[:] = rng.uniform(0.1, 0.45)
    m.vis.headlight.diffuse[:] = rng.uniform(0.2, 0.5)
    for li in range(m.nlight):
        m.light_diffuse[li] = rng.uniform(0.15, 0.6)
        m.light_pos[li, :2] += rng.normal(0, 0.3, 2)


def make_sample(seed: int):
    rng = np.random.default_rng(seed)
    m, placements = random_scene(rng)
    randomize_appearance(m, rng)
    d = mujoco.MjData(m)
    q = S.OBSERVE_Q + rng.normal(0, 0.15, 7) * (rng.random() < 0.5)
    S.reset(m, d, q)
    # 一部の物体はランダムな姿勢で落として転がす（倒れたペットボトルなど）
    for i, p in enumerate(placements):
        if rng.random() < 0.2:
            j = m.joint(f"obj{i}_joint").qposadr[0]
            d.qpos[j + 2] += 0.05
            qr = rng.normal(size=4)
            d.qpos[j + 3:j + 7] = qr / np.linalg.norm(qr)
    for _ in range(int(rng.integers(150, 400))):
        mujoco.mj_step(m, d)
    from .camera import OverheadCamera
    cam = OverheadCamera(m)
    rgb, _ = cam.capture(d)
    seg_r = mujoco.Renderer(m, IMG, IMG)
    seg_r.enable_segmentation_rendering()
    seg_r.update_scene(d, S.OVERHEAD_CAM)
    seg = seg_r.render()
    lut = S.geom_class_lut(m)
    ids = seg[..., 0]
    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    label = np.where(is_geom & (ids >= 0), lut[np.clip(ids, 0, m.ngeom)], 0).astype(np.uint8)
    return rgb, label


def _worker(args):
    seed, out = args
    rgb, label = make_sample(seed)
    np.savez_compressed(Path(out) / f"{seed:06d}.npz", rgb=rgb, label=label)
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/seg")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.out) for s in range(a.start, a.start + a.n)]
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_worker, jobs, chunksize=8)):
            if k % 200 == 0:
                print(f"{k}/{a.n}", flush=True)


if __name__ == "__main__":
    main()
