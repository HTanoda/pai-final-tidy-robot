"""models/train_log.json から学習曲線とクラス別 IoU の図を作る（レポート用）。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "models/train_log.json").read_text())
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures")
out.mkdir(parents=True, exist_ok=True)
ep = [e["epoch"] for e in log["epochs"]]

fig, ax1 = plt.subplots(figsize=(6, 3.5))
ax1.plot(ep, [e["train_loss"] for e in log["epochs"]], "o-", color="#3b6fb6", label="train loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("train loss")
ax2 = ax1.twinx()
ax2.plot(ep, [e["val_mIoU"] for e in log["epochs"]], "s-", color="#d9822b", label="val mIoU")
ax2.set_ylabel("val mIoU (objects)")
ax2.set_ylim(0, 1)
fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5))
fig.tight_layout()
fig.savefig(out / "training_curve.png", dpi=150)

last = log["epochs"][-1]["val_IoU"]
names = [n for n in log["classes"] if n != "background"]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(names, [last[n] for n in names], color="#3b6fb6")
ax.set_xlim(0, 1)
ax.set_xlabel("val IoU")
for i, n in enumerate(names):
    ax.text(last[n] + 0.01, i, f"{last[n]:.2f}", va="center")
fig.tight_layout()
fig.savefig(out / "class_iou.png", dpi=150)
print("saved to", out)
