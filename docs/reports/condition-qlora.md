# 条件指定データによるQLoRA学習

2026-09-25 14:51 JSTに学習プロセスを開始しました。完了状況は下記の`status.json`で確認します。

## 入力と設定

- データ: `datasets/dashcam_reranker_v3_conditions`
- 学習11,072ペア、検証2,241ペア。評価用4,964ペアは学習には使いません。
- ペア抽出上限は変更せず、作成済みの18,277ペアを使用します。
- モデル: `Qwen/Qwen3-VL-Reranker-8B`
- 前回と同じ4bit QLoRA、rank/alpha 32、dropout 0.05、2エポック、batch 2、勾配蓄積8、学習率0.0001、seed 42、正例重み1.0。
- 画像のみを入力し、captionは使いません。視覚エンコーダは凍結します。
- 最適化は1,384ステップ。開始前と各エポック終了時、最終状態で検証を行います。

## 実行記録

実行記録ディレクトリ: `results/qlora-v3-conditions-20260925T055127Z/`

- `launch.json`: 実行コマンド、入力ファイルのSHA256、件数、開始時刻。
- `train.log`: 学習ログ。
- `status.json`: 実行中は`running`。終了時に`completed`または`failed`と終了コードを記録。
- `supervise.py`: ターミナルから独立して学習を継続し、終了状態を保存するラッパー。

モデル出力先: `models/qwen3-vl-reranker-8b-dashcam-v3-conditions/`

学習スクリプトが作成する`runs/<run-id>/`に設定・アダプタ・完了時の学習履歴を保存します。検証nDCG@5が学習前を超えた場合のみ、完了時に`best`リンクを更新します。

進捗確認:

```bash
tail -f results/qlora-v3-conditions-20260925T055127Z/train.log
cat results/qlora-v3-conditions-20260925T055127Z/status.json
```

テストデータでの学習前後比較は学習完了後に行います。
