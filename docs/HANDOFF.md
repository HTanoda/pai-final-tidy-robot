# 引き継ぎ資料（2026-10-25 時点）

PAI 応用編1 最終課題「言葉で頼める片付けロボット」の実装状況と次の作業。
締切：**2026-11-02(月) AM10:00**。課題ルールはプロジェクト共有フォルダ `最終課題/Physical_AI応用1講座_最終課題_10-8版.md`。

## 現状（できていること）

| 項目 | 状態 | 数値・場所 |
|---|---|---|
| シーン（MuJoCo + Franka Panda、物体8種、箱3つ） | 完了 | `tidybot/scene.py` |
| IK 制御・ピック&プレース | 完了 | 正解位置を使った試験で 15/16 成功（失敗1件は別の物体に当たって倒れたボトル） |
| 合成データ生成 | 完了 | 3000枚（256×256、ドメインランダム化）、生成 約70分（4 CPU） |
| 視覚モデル学習 | 完了 | U-Net 約49万パラメータ、CPU 12エポック 66分、**検証 mIoU 0.896**（物体8クラス平均）。重み `models/segnet.pt`、ログ `models/train_log.json`、図 `docs/figures/` |
| 通し試験（学習済み視覚 + 簡易パーサ計画、低解像度） | 完了 | **8/8 成功、再試行なし**、動画 111秒 |
| LLM プランナー | コード完了・**未実行** | `tidybot/planner.py`（`claude-opus-5-5`、`messages.parse` の構造化出力） |
| 動画合成 | 完了 | `tidybot/demo.py`（ロボットの視界・計画リスト・指示/返事の字幕） |

クラス別 IoU（最終エポック）：りんご 0.95 / みかん 0.95 / バナナ 0.92 / ボトル 0.90 / 赤積み木 0.90 / 黄積み木 0.86 / 青積み木 0.85 / 空き缶 0.84。

共有フォルダ `/mnt/project-files/tidy-robot/` に、プレビュー動画（`preview_vision_lowres.mp4`）、その実行ログ（`preview_vision_log.json`）、代表フレーム、学習ログと図のコピーがあります。

## 次の作業

1. **LLM 計画の実行**（API キーが必要）
   - 方法A（推奨）：クラウド環境に `ANTHROPIC_API_KEY` を登録してあれば、クラウド側でそのまま `python -m tidybot.demo` を実行する（`api.anthropic.com` には到達可能）。
   - 方法B：ユーザーの開発機（Windows、RTX 5070 Ti、`ANTHROPIC_API_KEY` 設定済み）で Remote Control を使い、リポジトリを clone して `python -m tidybot.demo` を実行する。Windows なら MuJoCo は GPU で描画できるので速い。
   - 生成された計画は `plans/demo/stepN.json` に保存される。**これをコミットする**（再実行時はキャッシュを使うので API を呼ばない）。
   - 計画を作り直したいときは `plans/demo/` を消してから実行する。
2. **本番動画のレンダリング**：`python -m tidybot.demo --scenario scenarios/demo.json --out videos/demo.mp4`（`--scale 1.0`）。クラウドの CPU（osmesa）では 40〜60 分程度かかる見込み。長さが 1〜3 分に収まることを確認（シナリオは指示4つ・物体8個で約2分の想定）。
3. 本番の実行ログ `videos/demo.json`、動画の長さ、代表フレームを、レポート担当と台本担当のスレッドに渡す。
4. 時間があれば改善：
   - 俯瞰パネルのラベルが重なる（`vision_panel` のラベル位置）。
   - 失敗からの回復を見せる場面（例：倒れたボトルを横向きにつかむ）を入れると「技術力」の加点材料になる。
   - 簡易パーサは「食べられるもの」を解釈できず全部片付けてしまう（LLM 使用時は関係なし）。

## 環境構築（新しいクラウドコンテナ）

```bash
apt-get update && apt-get install -y libosmesa6 fonts-noto-cjk ffmpeg
git clone https://github.com/HTanoda/pai-final-tidy-robot && cd pai-final-tidy-robot
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch は PyPI 版（CPU でも動く）
./scripts/fetch_assets.sh                  # Franka モデルを third_party/ へ
export MUJOCO_GL=osmesa                    # クラウドでは EGL が使えない
```

- クラウドからは `huggingface.co` と `download.pytorch.org` に接続できない（学習済みモデルのダウンロード不可。だから視覚モデルは自前学習にした）。PyPI は使える。
- 合成データ（`data/seg/`、78MB）は git に入れていない。再学習するなら `python -m tidybot.perception.dataset` で作り直す。
- osmesa 描画は 640×480 で約0.3秒/フレーム。動画のレンダリングが一番時間を食う。

## 提出物の要件（再掲）

- 動画：MP4、1〜3分、YouTube タイトル `PAI最終課題＿hirokitanoda`、限定公開以上（公開だと加点）、倍速なら明記。
- レポート：PDF、ファイル名 `説明資料＿PAI最終課題＿hirokitanoda.pdf`。
- コード：このリポジトリ（public）。
- 氏名・所属・メールアドレスなどの個人情報は成果物に入れない（omnicampus アカウント名は可）。
