"""セグメンテーションモデルの学習。

    python -m tidybot.perception.train --data data/seg --epochs 20 --out models/segnet.pt

学習曲線とクラス別 IoU を models/train_log.json に保存する。GPU があれば自動で使う。
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import scene as S
from .model import SegNet, preprocess


class SegDataset(Dataset):
    def __init__(self, files, augment: bool):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        rgb, label = z["rgb"], z["label"].astype(np.int64)
        if self.augment:
            # 明るさ・コントラスト・ノイズ（物体の色クラスが変わらない程度）
            img = rgb.astype(np.float32)
            img = img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
            img += np.random.normal(0, 4, img.shape)
            rgb = np.clip(img, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.5:   # 左右反転（y 方向）
                rgb, label = rgb[:, ::-1], label[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(rgb)), torch.from_numpy(np.ascontiguousarray(label))


def iou_per_class(pred, gt, n):
    out = []
    for c in range(n):
        inter = ((pred == c) & (gt == c)).sum().item()
        union = ((pred == c) | (gt == c)).sum().item()
        out.append(inter / union if union else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/seg")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="models/segnet.pt")
    a = ap.parse_args()

    torch.manual_seed(0)
    files = sorted(glob.glob(f"{a.data}/*.npz"))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    n_val = max(1, len(files) // 10)
    val_files, train_files = files[:n_val], files[n_val:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train={len(train_files)} val={len(val_files)} device={dev}", flush=True)

    tl = DataLoader(SegDataset(train_files, True), batch_size=a.bs, shuffle=True, num_workers=2,
                    drop_last=True)
    vl = DataLoader(SegDataset(val_files, False), batch_size=32, num_workers=2)
    model = SegNet(S.NUM_CLASSES).to(dev)
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    # 背景が画素の大半を占めるので重みを下げる
    weight = torch.ones(S.NUM_CLASSES, device=dev)
    weight[0] = 0.1

    log = {"config": vars(a), "classes": S.CLASS_NAMES, "epochs": []}
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for rgb, label in tl:
            x, y = preprocess(rgb).to(dev), label.to(dev)
            loss = F.cross_entropy(model(x), y, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for rgb, label in vl:
                preds.append(model(preprocess(rgb).to(dev)).argmax(1).cpu())
                gts.append(label)
        ious = iou_per_class(torch.cat(preds), torch.cat(gts), S.NUM_CLASSES)
        miou = float(np.nanmean(ious[1:]))
        rec = {"epoch": ep + 1, "train_loss": tot / len(tl), "val_mIoU": miou,
               "val_IoU": dict(zip(S.CLASS_NAMES, ious)), "sec": time.time() - t0}
        log["epochs"].append(rec)
        print(f"ep{ep + 1} loss={rec['train_loss']:.4f} mIoU={miou:.3f} ({rec['sec']:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "classes": S.CLASS_NAMES}, a.out)
        Path(a.out).with_name("train_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
