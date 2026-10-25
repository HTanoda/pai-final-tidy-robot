"""自然言語の指示を「どの物体をどの箱へ」という計画に変換する（Claude API）。

視覚モデルが見つけた物体の一覧と箱の一覧を LLM に渡し、構造化出力（JSON）で
ピック&プレースの手順を受け取る。API キーがない環境では、キーワードによる
簡易パーサにフォールバックする（動作確認用）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from . import scene as S

MODEL = "claude-opus-5-5"

SYSTEM = """あなたは家庭用片付けロボットのタスクプランナーです。
ユーザーの指示と、カメラで検出された物体の一覧、片付け先の箱の一覧が与えられます。
指示を満たすために「どの物体を、どの箱に入れるか」を順番に決めてください。

ルール:
- 物体は object_id（検出一覧の id）で指定し、一覧にない物体は使わない。
- 片付け先は bin（箱の name）で指定する。
- すでに正しい箱に入っている物体は動かさない。
- 指示が曖昧なときは、物体の種類（食べ物・おもちゃ・ごみ）から常識的に判断する。
- 指示に当てはまる物体がない、または実行できない指示なら steps を空にし、理由を message で伝える。
- 手前（x が小さい）にある物から運ぶと、腕が他の物体にぶつかりにくい。
- message はロボットの返事として、日本語で1〜2文、です・ます調で書く。
"""


class Step(BaseModel):
    object_id: str = Field(description="検出一覧の id（例: apple#0）")
    bin: str = Field(description="片付け先の箱の name")
    reason: str = Field(description="この手順を選んだ理由（日本語・短く）")


class Plan(BaseModel):
    steps: list[Step]
    message: str = Field(description="ユーザーへの返事（日本語）")


def _scene_prompt(instruction: str, objects: list[dict]) -> str:
    bins = [{"name": b.name, "ja": b.ja} for b in S.BINS]
    return (f"指示: {instruction}\n\n"
            f"検出された物体:\n{json.dumps(objects, ensure_ascii=False, indent=1)}\n\n"
            f"片付け先の箱:\n{json.dumps(bins, ensure_ascii=False, indent=1)}")


def plan_with_claude(instruction: str, objects: list[dict]) -> Plan:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": _scene_prompt(instruction, objects)}],
        output_format=Plan,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"planner returned no plan (stop_reason={resp.stop_reason})")
    return resp.parsed_output


# ------------------------------------------------------------ フォールバック
_CATEGORY = {"apple": "fruit_basket", "orange": "fruit_basket", "banana": "fruit_basket",
             "can": "recycle_bin", "bottle": "recycle_bin",
             "red_block": "toy_box", "blue_block": "toy_box", "yellow_block": "toy_box"}
_KEYWORDS = {"果物": ["apple", "orange", "banana"], "フルーツ": ["apple", "orange", "banana"],
             "おもちゃ": ["red_block", "blue_block", "yellow_block"],
             "積み木": ["red_block", "blue_block", "yellow_block"],
             "ごみ": ["can", "bottle"], "ゴミ": ["can", "bottle"]}


def plan_with_rules(instruction: str, objects: list[dict]) -> Plan:
    wanted = set()
    for kw, names in _KEYWORDS.items():
        if kw in instruction:
            wanted |= set(names)
    for name, spec in S.BY_NAME.items():
        if spec.ja in instruction:
            wanted.add(name)
    if not wanted or "全部" in instruction or "片付けて" == instruction.strip():
        wanted = set(_CATEGORY)
    steps = [Step(object_id=o["id"], bin=_CATEGORY[o["name"]], reason="カテゴリ規則")
             for o in sorted(objects, key=lambda o: o["position_m"][0])
             if o["name"] in wanted and o["location"] != _CATEGORY[o["name"]]]
    return Plan(steps=steps, message=f"{len(steps)}個の物を片付けます。（簡易パーサ）")


def make_plan(instruction: str, objects: list[dict], cache: Path | None = None,
              offline: bool = False) -> tuple[Plan, str]:
    """計画を作る。cache があればそれを使い、なければ作って保存する。"""
    if cache is not None and cache.exists():
        d = json.loads(cache.read_text())
        return Plan.model_validate(d["plan"]), d["source"]
    if not offline and os.environ.get("ANTHROPIC_API_KEY"):
        plan, source = plan_with_claude(instruction, objects), MODEL
    else:
        plan, source = plan_with_rules(instruction, objects), "rules"
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"instruction": instruction, "objects": objects,
                                     "source": source, "plan": plan.model_dump()},
                                    ensure_ascii=False, indent=1))
    return plan, source
