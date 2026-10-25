# 提出チェックリスト

締切：**2026年11月2日（月）AM10:00**
ルール原本：「Physical AI応用1講座 最終課題（10/8版）」
omnicampus アカウント名：**hirokitanoda**

提出は2か所です。**omnicampus**（PDFレポート＋GitHub URL＋YouTube URL）と **YouTube**（動画本体）。

## 1. 動画（YouTube）

### 作る

- [ ] 計画は Claude で作られている（返事の欄に「（簡易パーサ）」と出ていない）
- [ ] `plans/demo/` に LLM の計画が保存され、リポジトリにコミット済み（再現用）
- [ ] 動画を生成：`python -m tidybot.demo --scenario scenarios/demo.json --out videos/demo.mp4`
- [ ] 尺が **1:00以上 3:00以下**（`ffprobe -v error -show_entries format=duration -of csv=p=0 videos/demo.mp4`）
- [ ] 3分を超えたら2倍速にし、**動画内と概要欄に「2倍速」と明記**（`docs/video_script.md` 末尾参照）
- [ ] 形式が **MP4**
- [ ] 4つの指示がすべて最後まで映っている（ぬいぐるみの指示では動かずに返事だけする）
- [ ] 画面に氏名・所属・メールアドレス・PCのユーザー名やパス（例 `C:\Users\…`）が映っていない
- [ ] ナレーションや BGM を入れる場合、著作権に問題がない

### アップロード（YouTube Studio）

- [ ] タイトル：**PAI最終課題＿hirokitanoda**（「＿」は全角アンダースコア）
- [ ] 概要欄：`docs/video_script.md` のテンプレートを貼る（再生速度と GitHub URL を含む）
- [ ] 視聴者：「**いいえ、子ども向けではありません**」
- [ ] 年齢制限：「**いいえ**」
- [ ] 公開設定：**限定公開以上**（**公開**にすると優秀賞選考で加点）
- [ ] 「インスタント プレミア公開」に**チェックしない**
- [ ] 予約公開にする場合、公開日時を **11/2 10:00 より前**に設定
- [ ] ログアウトした別ブラウザ（シークレットウィンドウ）で URL を開き、再生できることを確認
- [ ] 採点が終わるまで削除・非公開にしない

## 2. コード（GitHub）

- [ ] リポジトリ https://github.com/HTanoda/pai-final-tidy-robot が **public**（Settings → General → Danger Zone → Change visibility）
- [ ] 作業ブランチの PR がすべて `main` にマージ済み
- [ ] README に概要・AIの使い方・動かし方がある
- [ ] 学習済み重み（`models/`）と学習ログがコミット済み、または取得方法が README にある
- [ ] **API キーがコミットされていない**（`git log -p | grep -i "sk-ant"` で何も出ない）
- [ ] 氏名・所属・メールアドレスがコードやコミットに含まれていない（コミットの作者は `…@users.noreply.github.com`）
- [ ] 外部アセット（Franka Panda / MuJoCo Menagerie）の出典とライセンスを README に記載
- [ ] ログアウト状態でリポジトリのページが開けることを確認

## 3. レポート（PDF）

- [ ] ファイル名：**説明資料＿PAI最終課題＿hirokitanoda.pdf**
- [ ] 内容：課題の概要・目的・システム構成・AIの使い方（学習＋推論）・工夫した点・結果・今後の課題
- [ ] 結果に数値がある（視覚モデルのクラス別 IoU、片付けの成功数など）
- [ ] 動画の URL と GitHub の URL を記載
- [ ] 氏名・所属・メールアドレスを書いていない（omnicampus アカウント名は可）
- [ ] PDF のプロパティ（作成者欄）に氏名が入っていない（Word なら「ファイル→情報→ドキュメント検査」で削除）

## 4. omnicampus への提出

- [ ] レポート PDF をアップロード
- [ ] GitHub リポジトリ URL を入力
- [ ] YouTube 動画 URL を指定フォームに入力
- [ ] 提出完了画面を確認（**11/2 AM10:00 締切**。前日までに出すのが安全）

## 剽窃・関係ないファイルについて

- 他人の作品や最終課題サンプルの丸写しはしていない
- 課題と関係ないファイル（`data/`、`videos/` の生成物、個人ファイル）をリポジトリに含めていない
