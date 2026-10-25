"""学習済みセグメンテーションモデルで物体を検出し、把持に必要な 3D 情報を推定する。"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import scene as S
from .model import SegNet, preprocess

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models/segnet.pt"


@dataclasses.dataclass
class Detection:
    id: str                 # 例: "apple#0"
    cls: str
    ja: str
    xyz: np.ndarray         # 把持点（ワールド座標）
    top_z: float
    yaw: float              # 指の開閉方向
    symmetric: int          # 把持方向の対称性
    area_px: int
    score: float            # 画素ごとの確率の平均
    location: str           # "table" / 箱の名前
    mask: np.ndarray = dataclasses.field(repr=False)

    def describe(self) -> dict:
        return {"id": self.id, "name": self.cls, "ja": self.ja,
                "position_m": [round(float(v), 3) for v in self.xyz[:2]],
                "height_m": round(self.top_z, 3), "location": self.location,
                "confidence": round(self.score, 2)}


class Detector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu"):
        ck = torch.load(weights, map_location=device)
        self.classes = ck["classes"]
        self.net = SegNet(len(self.classes))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = preprocess(torch.from_numpy(rgb[None])).to(self.device)
        prob = torch.softmax(self.net(x), 1)[0].cpu().numpy()
        return prob.argmax(0), prob

    def detect(self, rgb, depth, cam, data, min_area: int = 12) -> tuple[list[Detection], np.ndarray]:
        label, prob = self.segment(rgb)
        dets: list[Detection] = []
        counts: dict[str, int] = {}
        for c in range(1, len(self.classes)):
            mask = (label == c).astype(np.uint8)
            n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for k in range(1, n):
                area = int(stats[k, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                m = comp == k
                v, u = np.nonzero(m)
                pts = cam.pixel_to_world(data, u.astype(float), v.astype(float), depth[v, u])
                det = self._geometry(c, pts, m, float(prob[c][m].mean()), area)
                name = self.classes[c]
                det.id = f"{name}#{counts.get(name, 0)}"
                counts[name] = counts.get(name, 0) + 1
                dets.append(det)
        return dets, label

    def _geometry(self, c, pts, mask, score, area) -> Detection:
        name = self.classes[c]
        xy = pts[:, :2]
        center = xy.mean(0)
        top_z = float(np.percentile(pts[:, 2], 90))
        cov = np.cov((xy - center).T) if len(xy) > 2 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        elong = math.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12))
        if elong > 1.6:
            # 細長い物体（バナナ、倒れたボトル）は長軸に直交する向きでつかむ
            major = evecs[:, 1]
            yaw, sym = math.atan2(major[1], major[0]) + math.pi / 2, 2
        elif name.endswith("_block"):
            rect = cv2.minAreaRect((xy * 1000).astype(np.float32))
            yaw, sym = math.radians(rect[2]), 4
        else:
            yaw, sym = 0.0, 4
        grasp_z = float(np.clip(top_z - 0.025, 0.015, 0.2))
        location = "table"
        for b in S.BINS:
            if abs(center[0] - b.pos[0]) < S.BIN_HALF + 0.01 and abs(center[1] - b.pos[1]) < S.BIN_HALF + 0.01:
                location = b.name
        return Detection("", name, S.CLASS_JA[c], np.array([center[0], center[1], grasp_z]),
                         top_z, yaw, sym, area, score, location, mask)
