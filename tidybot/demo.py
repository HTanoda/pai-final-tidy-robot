"""デモ実行と動画生成：指示 → 認識 → LLM 計画 → ピック&プレース（認識による確認と再試行つき）。

    python -m tidybot.demo --scenario scenarios/demo.json --out videos/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import scene as S
from .perception.camera import OverheadCamera
from .perception.detector import Detection, Detector
from .planner import make_plan
from .robot import Robot

ROOT = Path(__file__).resolve().parents[1]
FONT_PATHS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "C:/Windows/Fonts/meiryo.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"]

W, H = 1280, 720
MAIN_W, MAIN_H = 896, 504
SIDE_X = MAIN_W + 12
PANEL_IMG = 360

PALETTE = np.array([[0, 0, 0], [230, 40, 40], [255, 140, 0], [250, 220, 30], [170, 170, 190],
                    [60, 190, 230], [200, 30, 90], [40, 80, 230], [200, 200, 0]], np.uint8)


def _font(size: int, bold: bool = False):
    paths = FONT_PATHS[1:2] + FONT_PATHS if bold else FONT_PATHS
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Video:
    """シミュレーションの進行に合わせて 1 フレームずつ合成して書き出す。"""

    def __init__(self, model, data, out: Path, fps: int = 25, scale: float = 1.0):
        self.m, self.d, self.fps = model, data, fps
        self.scale = scale
        self.mw, self.mh = int(MAIN_W * scale) // 2 * 2, int(MAIN_H * scale) // 2 * 2
        self.r = mujoco.Renderer(model, self.mh, self.mw)
        self.writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                         macro_block_size=1)
        self.every = max(1, round(1 / (fps * model.opt.timestep)))
        self.k = 0
        self.f_title, self.f_big = _font(30, True), _font(26, True)
        self.f_mid, self.f_small = _font(20), _font(16)
        self.instruction = ""
        self.reply = ""
        self.status = ""
        self.plan_lines: list[tuple[str, str]] = []   # (状態, 文)
        self.vision: Image.Image | None = None
        self.caption = ""
        self.frames = 0

    def on_step(self):
        self.k += 1
        if self.k % self.every == 0:
            self.emit()

    def render_main(self) -> Image.Image:
        self.r.update_scene(self.d, S.FRONT_CAM)
        img = Image.fromarray(self.r.render())
        return img if self.scale == 1.0 else img.resize((MAIN_W, MAIN_H))

    def compose(self, main: Image.Image) -> Image.Image:
        c = Image.new("RGB", (W, H), (24, 26, 32))
        c.paste(main, (0, 0))
        dr = ImageDraw.Draw(c)
        # 右パネル：AI の視界
        dr.text((SIDE_X, 8), "ロボットの視界（AI認識）", font=self.f_mid, fill=(200, 220, 255))
        if self.vision is not None:
            c.paste(self.vision, (SIDE_X, 38))
        y = 38 + PANEL_IMG + 10
        dr.text((SIDE_X, y), "計画（LLM）", font=self.f_mid, fill=(200, 220, 255))
        y += 30
        for state, line in self.plan_lines[:8]:
            mark, col = {"done": ("✓", (120, 220, 120)), "doing": ("▶", (255, 210, 90)),
                         "fail": ("✗", (255, 110, 110))}.get(state, ("・", (210, 210, 210)))
            dr.text((SIDE_X, y), f"{mark} {line}", font=self.f_small, fill=col)
            y += 24
        # 下部：指示と返事
        dr.rectangle([0, MAIN_H, MAIN_W, H], fill=(16, 18, 22))
        if self.instruction:
            dr.text((16, MAIN_H + 12), "指示", font=self.f_small, fill=(150, 160, 180))
            dr.text((64, MAIN_H + 6), f"「{self.instruction}」", font=self.f_big, fill=(255, 255, 255))
        if self.reply:
            dr.text((16, MAIN_H + 62), "返事", font=self.f_small, fill=(150, 160, 180))
            dr.text((64, MAIN_H + 58), self.reply, font=self.f_mid, fill=(170, 230, 255))
        if self.status:
            dr.text((16, MAIN_H + 150), self.status, font=self.f_mid, fill=(255, 210, 90))
        if self.caption:
            dr.rectangle([0, 0, MAIN_W, 44], fill=(0, 0, 0))
            dr.text((14, 6), self.caption, font=self.f_title, fill=(255, 255, 255))
        dr.text((MAIN_W - 150, MAIN_H - 28), f"t = {self.d.time:5.1f} s", font=self.f_small,
                fill=(40, 40, 40))
        return c

    def emit(self, n: int = 1):
        frame = np.asarray(self.compose(self.render_main()))
        for _ in range(n):
            self.writer.append_data(frame)
            self.frames += 1

    def hold(self, sec: float):
        """シミュレーションを止めて（考えている間など）同じ画面を表示する。"""
        self.emit(max(1, int(sec * self.fps)))

    def card(self, title: str, lines: list[str], sec: float):
        c = Image.new("RGB", (W, H), (18, 22, 30))
        dr = ImageDraw.Draw(c)
        dr.text((80, 220), title, font=_font(48, True), fill=(255, 255, 255))
        for i, line in enumerate(lines):
            dr.text((84, 310 + i * 44), line, font=_font(28), fill=(190, 210, 235))
        frame = np.asarray(c)
        for _ in range(int(sec * self.fps)):
            self.writer.append_data(frame)
            self.frames += 1

    def close(self):
        self.writer.close()


def vision_panel(rgb, label, dets: list[Detection], highlight: str | None = None) -> Image.Image:
    over = rgb.astype(np.float32)
    m = label > 0
    over[m] = 0.45 * over[m] + 0.55 * PALETTE[label[m]]
    img = Image.fromarray(over.astype(np.uint8)).resize((PANEL_IMG, PANEL_IMG), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    f = _font(14, True)
    s = PANEL_IMG / rgb.shape[0]
    for det in dets:
        v, u = np.nonzero(det.mask)
        x0, y0, x1, y1 = u.min() * s - 3, v.min() * s - 3, u.max() * s + 3, v.max() * s + 3
        col = (255, 255, 0) if det.id == highlight else (255, 255, 255)
        dr.rectangle([x0, y0, x1, y1], outline=col, width=3 if det.id == highlight else 1)
        dr.text((x0, y0 - 17), f"{det.ja} {det.score:.2f}", font=f, fill=col,
                stroke_width=2, stroke_fill=(0, 0, 0))
    return img


class TidyBot:
    def __init__(self, placements, weights=None, out: Path | None = None, scale=1.0,
                 plan_dir: Path | None = None, offline=False):
        self.m = S.build_spec(placements).compile()
        self.d = mujoco.MjData(self.m)
        S.reset(self.m, self.d)
        self.video = Video(self.m, self.d, out, scale=scale) if out else None
        self.robot = Robot(self.m, self.d, self.video.on_step if self.video else None)
        self.cam = OverheadCamera(self.m)
        self.det = Detector(weights) if weights else Detector()
        self.plan_dir, self.offline = plan_dir, offline
        self.log: list[dict] = []
        self.robot.step(250)   # 物体を落ち着かせる

    def observe(self, highlight=None) -> list[Detection]:
        rgb, depth = self.cam.capture(self.d)
        dets, label = self.det.detect(rgb, depth, self.cam, self.d)
        self._last = (rgb, label, dets)
        if self.video:
            self.video.vision = vision_panel(rgb, label, dets, highlight)
        return dets

    def _find(self, dets: list[Detection], name: str, near=None) -> Detection | None:
        cands = [d for d in dets if d.cls == name]
        if not cands:
            return None
        if near is None:
            return cands[0]
        return min(cands, key=lambda d: np.linalg.norm(d.xyz[:2] - near))

    def run_instruction(self, instruction: str, tag: str):
        v = self.video
        if v:
            v.instruction, v.reply, v.plan_lines = instruction, "", []
            v.status = "カメラで机の上を確認しています…"
        dets = self.observe()
        if v:
            v.hold(1.5)
            v.status = "AIが指示を解釈して計画を立てています…"
            v.hold(1.0)
        objects = [d.describe() for d in dets]
        cache = self.plan_dir / f"{tag}.json" if self.plan_dir else None
        t0 = time.time()
        plan, source = make_plan(instruction, objects, cache, self.offline)
        print(f"[{tag}] plan by {source} ({time.time() - t0:.1f}s): {plan.message}")
        by_id = {d.id: d for d in dets}
        if v:
            v.reply = plan.message
            v.plan_lines = [("todo", f"{by_id[s.object_id].ja if s.object_id in by_id else s.object_id}"
                                     f" → {S.BIN_BY_NAME[s.bin].ja}") for s in plan.steps]
            v.status = ""
            v.hold(2.5)
        slots = {b.name: sum(1 for d in dets if d.location == b.name) for b in S.BINS}
        rec = {"instruction": instruction, "planner": source, "message": plan.message,
               "steps": []}
        for i, step in enumerate(plan.steps):
            name = step.object_id.split("#")[0]
            prev = by_id.get(step.object_id)
            ok = self.pick_and_place(i, name, step.bin, prev.xyz[:2] if prev else None, slots)
            rec["steps"].append({"object": step.object_id, "bin": step.bin, "success": ok})
        self.log.append(rec)
        if v:
            n_ok = sum(s["success"] for s in rec["steps"])
            v.status = f"完了：{n_ok}/{len(plan.steps)} 個を片付けました"
            self.robot.go_observe(1.0)
            self.observe()
            v.hold(2.0)
            v.status = ""

    def pick_and_place(self, i, name, bin_name, near, slots, retries: int = 2) -> bool:
        v = self.video
        ja = S.BY_NAME[name].ja
        for attempt in range(retries + 1):
            dets = self.observe()
            det = self._find([d for d in dets if d.location == "table"], name, near)
            if det is None:
                if v:
                    v.plan_lines[i] = ("fail", v.plan_lines[i][1] + "（見つからない）")
                return False
            self.observe(highlight=det.id)
            if v:
                v.plan_lines[i] = ("doing", v.plan_lines[i][1])
                v.status = f"{ja}をつかみます" + (f"（再試行 {attempt}）" if attempt else "")
            self.robot.pick(det.xyz, det.yaw, det.symmetric)
            if not self.robot.holding():
                if v:
                    v.status = f"{ja}をつかめませんでした。もう一度見直します"
                self.robot.gripper(255, 0.2)
                self.robot.go_observe(1.2)
                near = det.xyz[:2]
                continue
            if v:
                v.status = f"{ja}を{S.BIN_BY_NAME[bin_name].ja}へ運びます"
            self.robot.place(bin_name, det.xyz[2], slots[bin_name])
            slots[bin_name] += 1
            self.robot.go_observe(1.2)
            # 視覚で結果を確認
            dets = self.observe()
            ok = any(d.cls == name and d.location == bin_name for d in dets)
            if v:
                v.plan_lines[i] = ("done" if ok else "fail", v.plan_lines[i][1])
                v.status = f"確認：{ja}は{S.BIN_BY_NAME[bin_name].ja}に入りました" if ok else \
                    f"確認：{ja}が箱に入っていません"
            if ok:
                return True
            near = None
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(ROOT / "scenarios/demo.json"))
    ap.add_argument("--out", default=str(ROOT / "videos/demo.mp4"))
    ap.add_argument("--plans", default=str(ROOT / "plans"))
    ap.add_argument("--scale", type=float, default=1.0, help="描画解像度の倍率（下げると速い）")
    ap.add_argument("--offline", action="store_true", help="LLM を使わず簡易パーサで計画する")
    a = ap.parse_args()

    sc = json.loads(Path(a.scenario).read_text())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sc.get("seed", 0))
    placements = S.sample_placements(rng, sc["objects"])
    for p, fixed in zip(placements, sc.get("positions", [])):
        p.x, p.y, p.yaw = fixed
    bot = TidyBot(placements, out=Path(a.out), scale=a.scale, plan_dir=Path(a.plans) / sc["name"],
                  offline=a.offline)
    v = bot.video
    v.card(sc["title"], sc.get("subtitle", []), 4.0)
    bot.observe()
    v.caption = "シーン：散らかった机"
    v.hold(2.0)
    for k, inst in enumerate(sc["instructions"]):
        v.caption = f"指示 {k + 1} / {len(sc['instructions'])}"
        bot.run_instruction(inst, f"step{k + 1}")
    v.caption = ""
    v.card("おわり", sc.get("ending", []), 3.0)
    v.close()
    Path(a.out).with_suffix(".json").write_text(json.dumps(bot.log, ensure_ascii=False, indent=1))
    print(f"video: {a.out} ({v.frames / v.fps:.1f}s)")


if __name__ == "__main__":
    main()
