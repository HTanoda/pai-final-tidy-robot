# 言葉で頼める片付けロボット（PAI 最終課題）

「お腹がすいたから、食べられるものを集めておいて」のような**あいまいな日本語の指示**を受けて、
ロボットアーム（Franka Panda）が机の上の物を見分け、適切な箱へ片付けるシステムです。
シミュレータは MuJoCo を使っています。

```
 指示（日本語） ──▶ LLM プランナー（Claude） ──▶ 「どの物を・どの箱へ」の手順
                         ▲ 検出結果                         │
 俯瞰 RGB-D カメラ ─▶ 自前学習の U-Net ─▶ 物体の種類・3D位置・向き     ▼
                                        ▲          逆運動学でピック&プレース
                                        └── 動作のたびに再認識して成否を確認・再試行
```

## AI の使い方

| 役割 | 手法 | 学習 |
|---|---|---|
| 視覚（物体の認識） | 小型 U-Net による 9 クラスのセマンティックセグメンテーション（背景＋物体8種） | **あり**：シミュレータで自動生成した合成画像（ドメインランダム化）で学習 |
| 言語（指示の解釈・計画） | Claude（Anthropic API）の構造化出力で、検出物体と箱の一覧から手順を生成 | なし（推論のみ） |

- 視覚モデルの教師データは MuJoCo のセグメンテーション描画から自動で作るため、人手のアノテーションは不要です。
  色・照明・テーブルの色・アームの姿勢・倒れた物体・未知の物体（背景扱い）をランダム化しています。
- 赤い積み木とりんご、黄色い積み木とバナナのように**色が似た物を形で見分ける**必要があります。
- 認識結果（マスク）と深度画像から、把持点の 3D 座標・高さ・つかむ向き（細長い物は長軸に直交）を求めます。
- 計画は物体の一覧（JSON）を見て LLM が決めるので、「食べられるもの」「飲み終わったもの」のような
  カテゴリ名が出てこない指示や、存在しない物の指示（「ぬいぐるみ」）にも対応します。
- 各動作の後にもう一度カメラで確認し、つかみ損ねたら再認識してやり直します。

## 構成

```
tidybot/
  scene.py               シーン生成（MjSpec）：机・物体カタログ・箱・カメラ
  robot.py               減衰最小二乗法の逆運動学、ピック/プレース動作
  planner.py             Claude による計画（構造化出力）と簡易パーサ
  demo.py                デモ実行と動画合成
  perception/
    dataset.py           合成データ生成（ドメインランダム化）
    model.py             U-Net
    train.py             学習（クラス別 IoU を記録）
    detector.py          推論 → 物体ごとの 3D 位置・把持方向
    camera.py            俯瞰 RGB-D カメラと逆投影
scenarios/demo.json      デモの物体と指示
plans/                   LLM が生成した計画（再現用キャッシュ）
models/                  学習済み重みと学習ログ
```

## 動かし方

```bash
pip install -r requirements.txt
./scripts/fetch_assets.sh                      # Franka Panda モデル（MuJoCo Menagerie）を取得

# 1. 合成データ生成と学習（学習済み重みは models/ に同梱）
python -m tidybot.perception.dataset --out data/seg --n 3000
python -m tidybot.perception.train --data data/seg --epochs 20

# 2. デモ（ANTHROPIC_API_KEY があれば Claude で計画、なければ --offline で簡易パーサ）
export ANTHROPIC_API_KEY=...                   # 計画は plans/ に保存され、2回目以降は再利用
python -m tidybot.demo --scenario scenarios/demo.json --out videos/demo.mp4
```

ヘッドレス環境では `MUJOCO_GL=osmesa`（または `egl`）で描画します。

## ライセンス・出典

- Franka Emika Panda モデル：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0）
- シミュレータ：[MuJoCo](https://mujoco.org/)
