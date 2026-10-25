"""Franka Panda の運動制御：減衰最小二乗法による逆運動学と、把持・運搬の動作プリミティブ。"""
from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

from . import scene as S

GRIP_OPEN = 255.0
GRIP_CLOSE = 0.0
SAFE_Z = 0.24          # 移動時の TCP 高さ
# 箱の中で物体を置く位置（中心からのずれ）。順に埋めて積み重なりを防ぐ
BIN_SLOTS = [(-0.04, -0.04), (0.04, 0.04), (-0.04, 0.04), (0.04, -0.04), (0.0, 0.0)]


def _down_rotation(yaw: float) -> np.ndarray:
    """指が下向き、開閉方向（hand の y 軸）が水平方向 yaw を向く回転行列。"""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def _rot_error(R_cur: np.ndarray, R_goal: np.ndarray) -> np.ndarray:
    dR = R_goal @ R_cur.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, dR.flatten())
    if q[0] < 0:
        q = -q
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q, 1.0)
    return vel


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 on_step: Callable[[], None] | None = None):
        self.m, self.d = model, data
        self.ik_data = mujoco.MjData(model)
        self.site = model.site("tcp").id
        self.hand = model.body("hand").id
        self.lo = model.jnt_range[:7, 0].copy()
        self.hi = model.jnt_range[:7, 1].copy()
        self.on_step = on_step or (lambda: None)
        self.held: str | None = None

    # ------------------------------------------------------------ 逆運動学
    def ik(self, pos, yaw: float, q_seed=None, iters: int = 300) -> tuple[np.ndarray, float]:
        m, d = self.m, self.ik_data
        d.qpos[:] = self.d.qpos
        if q_seed is not None:
            d.qpos[:7] = q_seed
        R_goal = _down_rotation(yaw)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err_norm = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            err = np.concatenate([np.asarray(pos) - p, 0.5 * _rot_error(R, R_goal)])
            err_norm = float(np.linalg.norm(err))
            if err_norm < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp, jacr])[:, :7]
            lam = 1e-2
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            # 関節中央へ寄せる零空間項（姿勢を自然に保つ）
            null = (np.eye(7) - np.linalg.pinv(J) @ J) @ (0.05 * (S.HOME_Q - d.qpos[:7]))
            d.qpos[:7] = np.clip(d.qpos[:7] + dq + null, self.lo, self.hi)
        return d.qpos[:7].copy(), err_norm

    def best_yaw(self, yaw: float, symmetric: int = 2) -> float:
        """対称性（2回 or 4回）を使い、手首の回転量が最小になる把持方向を選ぶ。"""
        cur = self.tcp_yaw()
        cands = [yaw + k * 2 * math.pi / symmetric for k in range(symmetric)]
        return min(cands, key=lambda a: abs(math.remainder(a - cur, 2 * math.pi)))

    def tcp_yaw(self) -> float:
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return math.atan2(R[1, 1], R[0, 1])

    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.site].copy()

    # ------------------------------------------------------------ 実行
    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.on_step()

    def move_joints(self, q_goal, duration: float):
        q0 = self.d.ctrl[:7].copy()
        n = max(1, int(duration / self.m.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5        # 最小躍度補間
            self.d.ctrl[:7] = q0 + (q_goal - q0) * s
            self.step()

    def move_to(self, pos, yaw: float, duration: float = 1.5, segments: int = 1):
        """TCP を pos へ。segments>1 のときはデカルト空間で直線補間（上下動用）。"""
        start = self.tcp_pos()
        q = self.d.ctrl[:7].copy()
        for k in range(1, segments + 1):
            target = start + (np.asarray(pos) - start) * k / segments
            q, _ = self.ik(target, yaw, q_seed=q)
            self.move_joints(q, duration / segments)

    def gripper(self, value: float, wait: float = 0.5):
        self.d.ctrl[7] = value
        self.step(int(wait / self.m.opt.timestep))

    def go_observe(self, duration: float = 1.5):
        self.move_joints(S.OBSERVE_Q, duration)

    # ------------------------------------------------------------ 動作プリミティブ
    def pick(self, xyz, yaw: float, symmetric: int = 2) -> None:
        x, y, z = xyz
        yaw = self.best_yaw(yaw, symmetric)
        self.gripper(GRIP_OPEN, 0.05)
        self.move_to([x, y, SAFE_Z], yaw, 1.6)
        self.move_to([x, y, z + 0.06], yaw, 0.6)
        self.move_to([x, y, z], yaw, 0.6, segments=6)
        self.gripper(GRIP_CLOSE, 0.6)
        self.move_to([x, y, SAFE_Z], yaw, 0.8, segments=6)

    def holding(self) -> bool:
        """指の開き幅から、何かをつかんでいるかを判定する。"""
        width = self.d.qpos[7] + self.d.qpos[8]
        return width > 0.004

    def place(self, bin_name: str, grasp_z: float, slot: int = 0) -> None:
        """箱の中の slot 番目の位置へ、物体の底が箱の床すれすれになる高さで置く。"""
        b = S.BIN_BY_NAME[bin_name]
        off = BIN_SLOTS[slot % len(BIN_SLOTS)]
        x, y = b.pos[0] + off[0], b.pos[1] + off[1]
        yaw = self.best_yaw(self.tcp_yaw(), 2)
        z = grasp_z + 0.03
        self.move_to([x, y, max(SAFE_Z, z + 0.08)], yaw, 1.6)
        self.move_to([x, y, z], yaw, 0.6, segments=5)
        self.gripper(GRIP_OPEN, 0.5)
        self.move_to([x, y, max(SAFE_Z, z + 0.08)], yaw, 0.6, segments=5)
