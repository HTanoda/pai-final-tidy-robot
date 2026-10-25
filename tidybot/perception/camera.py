"""俯瞰 RGB-D カメラ：画像の取得と、画素→ワールド座標の逆投影。"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .. import scene as S

IMG = 256


class OverheadCamera:
    def __init__(self, model: mujoco.MjModel, size: int = IMG):
        self.m = model
        self.size = size
        self.rgb_r = mujoco.Renderer(model, size, size)
        self.depth_r = mujoco.Renderer(model, size, size)
        self.depth_r.enable_depth_rendering()
        self.cam = model.camera(S.OVERHEAD_CAM).id
        fovy = model.cam_fovy[self.cam]
        self.f = (size / 2) / math.tan(math.radians(fovy) / 2)

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self.rgb_r.update_scene(data, S.OVERHEAD_CAM)
        self.depth_r.update_scene(data, S.OVERHEAD_CAM)
        return self.rgb_r.render().copy(), self.depth_r.render().copy()

    def pixel_to_world(self, data: mujoco.MjData, u: np.ndarray, v: np.ndarray,
                       depth: np.ndarray) -> np.ndarray:
        """画素 (u=列, v=行) と深度 [m] からワールド座標 (N,3) を求める。"""
        c = self.size / 2 - 0.5
        x_c = (u - c) / self.f * depth
        y_c = -(v - c) / self.f * depth
        z_c = -depth
        pts_c = np.stack([x_c, y_c, z_c], axis=-1)
        R = data.cam_xmat[self.cam].reshape(3, 3)
        return pts_c @ R.T + data.cam_xpos[self.cam]
