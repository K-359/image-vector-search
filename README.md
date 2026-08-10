# Text-Based Image Retrieval with Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-8B

`images/` ディレクトリ内の画像から、入力テキストに意味的に近い画像を検索するプログラムです。

Qwen3-VL-Embedding-2B を使って、画像とテキストを同じ埋め込み空間のベクトルに変換し、FAISS で類似検索します。
初段検索の上位候補は、bitsandbytes で8bit量子化した Qwen3-VL-Reranker-8B で再ランキングします。

## ディレクトリ構成

```text
.
├── images/
│   ├── sample001.jpg
│   ├── sample002.jpg
│   └── ...
├── data/
│   ├── images.faiss
│   ├── image_paths.json
│   ├── image_captions.jsonl
│   ├── caption_embeddings.faiss
│   ├── caption_embeddings_meta.json
│   └── image_dates.json
├── results/
│   └── 20260424_153012/
│       ├── query.txt
│       ├── 01_0.8123_sample001.jpg
│       └── ...
├── datasets/
│   └── dashcam_reranker_ft_v1/
│       ├── manifests/sampled_images.jsonl
│       ├── raw_teacher/scene_cards.jsonl
│       ├── derived/queries.jsonl
│       ├── pairs.train.jsonl
│       ├── pairs.val.jsonl
│       ├── pairs.test.jsonl
│       └── reports/
├── models/
│   └── qwen3-vl-reranker-8b-dashcam-v1/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── scripts/
│   ├── build_index.py
│   ├── generate_captions.py
│   ├── generate_image_dates.py
│   ├── search.py
│   ├── reranker_common.py
│   ├── build_reranker_dataset.py
│   ├── train_reranker_qlora.py
│   └── evaluate_reranker.py
├── requirements.txt
└── README.md
```

## セットアップ

仮想環境を作成します。

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows の場合は以下です。

```bash
.venv\Scripts\activate
```

依存ライブラリをインストールします。

```bash
pip install -U pip
pip install -r requirements.txt
```

## 画像を配置する

検索対象の画像を `images/` ディレクトリに入れます。

対応拡張子:

```text
.jpg
.jpeg
.png
```

例:

```text
images/
├── car_001.jpg
├── dog_002.jpg
└── snow_003.jpg
```

## 画像インデックスを作成する

最初に、画像をベクトル化して検索用インデックスを作成します。

```bash
python scripts/build_index.py
```

成功すると、以下のファイルが作成されます。

```text
data/images.faiss
data/image_paths.json
```

- `images.faiss`: 画像ベクトルの検索インデックス
- `image_paths.json`: ベクトルIDと元画像ファイルパスの対応表

## 実験用の日付メタデータを作成する

画像に実際の日付メタデータがない場合は、検索実験用にランダムな日付を割り当てられます。

```bash
python scripts/generate_image_dates.py
```

デフォルトでは `data/image_paths.json` の各画像に対して、`2023-01-01` から `2025-12-31` までの1096日を割り当て、`data/image_dates.json` に保存します。
10万件の場合は、各日に91枚または92枚が対応するように日付ラベルを作ってからシャッフルします。

期間や乱数シードを変更する場合は、以下のように指定します。

```bash
python scripts/generate_image_dates.py --start-date 2023-01-01 --end-date 2025-12-31 --seed 42
```

`image_dates.json` は以下のような形式です。

```json
{
  "schema_version": 1,
  "start_date": "2023-01-01",
  "end_date": "2025-12-31",
  "dates_by_path": {
    "images/sample001.jpg": "2023-02-11"
  }
}
```

`scripts/search.py` は `data/image_dates.json` が存在する場合だけ自動で読み込み、検索結果の表示とLLMへ渡す画像文脈に日付を含めます。

画像を追加・削除・変更した場合は、インデックス作成と日付メタデータ作成を再実行してください。

## 画像キャプションを事前生成する

キャプションに対してベクトル検索や BM25 検索を行う実験用に、画像ごとの車外状況説明を事前生成できます。
Ollama の `qwen3.5:9b` を `think=False` で呼び出します。

```bash
python scripts/generate_captions.py
```

デフォルトでは `data/image_paths.json` に含まれる画像を順に処理し、`data/image_captions.jsonl` に1画像1行で追記します。
既存の出力に含まれる画像パスは自動でスキップするため、途中で止まった場合も同じコマンドで再開できます。
キャプションは1件生成するたびに保存されます。
プロセス停止などで `data/image_captions.jsonl` の末尾に不完全な行が残った場合は、次回起動時にその末尾行だけを削除し、該当画像を未処理として再生成します。

出力例:

```json
{"schema_version":1,"image_path":"images/sample001.jpg","caption":"片側一車線の道路を走行しており、前方に車両が見えます。周囲に歩行者や自転車は見えません。前方車両は自車と同じ方向に進んでいるように見えます。道路上に大きな障害物は見えません。","prompt":"入力画像は車内または車載カメラから撮影された交通シーンです。画像検索に使うため、見える事実にもとづく日本語キャプションを4文程度で出力してください。道路環境、車両・自転車・歩行者などの対象物、自車から見た位置、進行方向、動きを具体的に書いてください。危険、ヒヤリハット、接近、飛び出し、逆走、信号無視、車線はみ出し、急停止などに見える要素があれば、その内容と該当する語を明示してください。見えないことや不確実なことは断定しないでください。改行や段落は必要ありません。","model":"qwen3.5:9b","created_at":"2026-05-22T05:00:00+00:00"}
```

動作確認として最初の数枚だけ処理する場合:

```bash
python scripts/generate_captions.py --limit 10
```

Ollama の接続先やタイムアウトを変える場合:

```bash
python scripts/generate_captions.py --ollama-url http://localhost:11434 --ollama-timeout 600
```

既存出力を破棄して最初から作り直す場合:

```bash
python scripts/generate_captions.py --overwrite
```

画像単位の失敗で処理を止めず、失敗内容を `data/image_caption_errors.jsonl` に記録して続行する場合:

```bash
python scripts/generate_captions.py --continue-on-error
```

`scripts/search.py --mode caption` は `data/image_captions.jsonl` を読み込み、初回実行時にキャプション埋め込みの FAISS インデックスを `data/caption_embeddings.faiss` に作成します。
`data/image_captions.jsonl` の末尾にキャプションが追加された場合は、追加分だけを `data/caption_embeddings.faiss` に追記します。
既存のメタデータと整合しない変更がある場合は自動で作り直します。
初回インデックス作成は Reranker や Ollama の検索要否判定モデルを呼ぶ前に実行されます。
このときだけ `--caption-index-embedding-device auto` により、利用可能なら `--reranker-device` と同じGPUでキャプション埋め込みを生成します。
生成後はEmbeddingモデルを解放してからRerankerをロードし、その後の検索クエリの埋め込みはデフォルトでCPU推論します。

## テキストで画像を検索する

以下のように検索します。

```bash
python scripts/search.py "赤い車が雪道を走っている"
```

デフォルトでは従来の画像ベクトル検索で上位10件を検索します。
検索方式は `--mode` で切り替えられます。

```bash
# 従来の画像ベクトル検索
python scripts/search.py "赤い車が雪道を走っている" --mode image

# 事前生成キャプションを使った、ベクトル検索 + BM25 検索のハイブリッド
python scripts/search.py "赤い車が雪道を走っている" --mode caption
```

どちらのモードも、初段検索の上位50件をデフォルトで Qwen3-VL-Reranker-8B に渡します。
再ランカーは bitsandbytes の8bit量子化を使い、デフォルトでは全体を `cuda:0` にロードします。
VRAMを再ランカーへ優先配分するため、検索クエリ用EmbeddingモデルはデフォルトでCPUへロードします。
`caption` モードのインデックス作成/追記時だけ、デフォルトでは利用可能な `--reranker-device` を使ってGPU推論します。
配置先は `--embedding-device`, `--caption-index-embedding-device`, `--reranker-device` で変更できます。
`image` モードでは画像、`caption` モードでは画像とキャプションを組み合わせて再ランキングします。
短い逆走自転車クエリは、再ランカーが進行方向を判定しやすい英語の視覚条件へ正規化します。
初段検索には元のクエリを使い、正規化後のクエリは結果ディレクトリの
`reranker_query.txt` に保存します。

```bash
# 再ランキングする候補数を変更
python scripts/search.py "赤い車が雪道を走っている" --rerank-candidates 100

# GPUメモリに余裕がある場合はバッチサイズを上げて高速化
python scripts/search.py "赤い車が雪道を走っている" --reranker-batch-size 4

# 再ランキングを無効化
python scripts/search.py "赤い車が雪道を走っている" --rerank-candidates 0
```

`caption` モードのハイブリッドスコアは、キャプションベクトル検索順位と BM25 検索順位を RRF (Reciprocal Rank Fusion) で統合します。
BM25 はスコアが 0 より大きい結果だけを RRF の入力にします。
RRF の順位定数はデフォルトで `60` です。

```bash
python scripts/search.py "赤い車が雪道を走っている" --mode caption --caption-rrf-k 30
```

別のキャプションJSONLを使う場合:

```bash
python scripts/search.py "赤い車が雪道を走っている" --mode caption --captions-jsonl data/image_captions.jsonl
```

初回のキャプション埋め込み生成で GPU メモリが足りない場合は、バッチサイズを下げます。

```bash
python scripts/search.py "赤い車が雪道を走っている" --mode caption --caption-batch-size 8
```

検索結果は標準出力に表示されるだけでなく、`results/` ディレクトリにもコピーされます。

例:

```text
results/
└── 20260424_153012/
    ├── query.txt
    ├── raw_query.txt
    ├── llm_response.txt
    ├── ollama_thinking.txt
    ├── ollama_thinking.json
    ├── 01_0.8123_car_001.jpg
    ├── 02_0.7991_snow_045.jpg
    ├── 03_0.7814_vehicle_120.jpg
    └── ...
```

## 検索結果の保存形式

検索を実行するたびに、`results/実行時刻/` というディレクトリが作成されます。

ディレクトリ名の形式:

```text
YYYYMMDD_HHMMSS
```

例:

```text
20260424_153012
```

各画像ファイル名は以下の形式で保存されます。

```text
順位_スコア_元ファイル名
```

例:

```text
01_0.8123_car_001.jpg
02_0.7991_snow_045.jpg
```

また、検索に使ったクエリは以下に保存されます。

```text
query.txt
```

例:

```text
赤い車が雪道を走っている
```

`query.txt` には実際に検索へ使ったクエリが保存されます。ユーザが入力した元の文は `raw_query.txt` に保存されます。クエリ変換を使わない通常検索では、両方とも同じ内容になります。

LLM の最終回答は `llm_response.txt` に保存されます。`--skip-answer-generation` 指定時は、このファイルは作成されません。
Ollama の thinking は `ollama_thinking.txt` に保存され、画像検索要否判定、クエリ変換、回答生成の各ステップごとに確認できます。
推論中も `ollama_thinking.txt` へ逐次追記され、処理完了後に同じファイルが整形済みの内容で上書きされます。
同じ内容を機械的に扱いやすい形式で確認したい場合は `ollama_thinking.json` を参照してください。
画像検索が不要と判定された場合も `results/実行時刻/` は作成され、`raw_query.txt`、`query.txt`、`llm_response.txt`、`ollama_thinking.txt`、`ollama_thinking.json` が保存されます。`--skip-answer-generation` 指定時は `llm_response.txt` を除くファイルが保存されます。

検索時は、まず Ollama の `qwen3.5:9b` が回答に過去画像データベースの参照が必要かを `Yes` / `No` で判定します。
`Yes` の場合は画像検索し、最上位の検索結果画像を LLM に渡して回答を生成します。
`No` の場合は画像検索を行わず、LLM が通常のテキスト質問として回答します。
検索結果画像を使って回答させるには、`--ollama-model` に画像入力へ対応した Ollama モデルを指定してください。
`--interactive` では、入力待ちを開始する前に Qwen3-VL-Embedding-2B と
8bit量子化した Qwen3-VL-Reranker-8B をロードし、LLM 回答をストリーミング表示します。

画像検索要否判定をスキップし、常に画像検索が必要なものとして処理する場合は、`--skip-image-search-decision` を指定します。

```bash
python scripts/search.py "赤い車が雪道を走っている" --skip-image-search-decision
```

LLM の最終回答生成をスキップし、検索結果だけを取得する場合は、`--skip-answer-generation` を指定します。
画像検索要否判定もスキップすると、Ollama を呼び出さずに検索だけを実行できます。

```bash
python scripts/search.py "赤い車が雪道を走っている" --skip-image-search-decision --skip-answer-generation
```

画像検索が必要な場合に、Ollama で入力を画像検索向けの短い視覚クエリに変換してから検索したい場合は、`--query-rewrite` を指定します。

```bash
python scripts/search.py "赤い車が雪道を走っている" --query-rewrite
```

Ollama の接続先やモデルを変える場合は、以下のオプションを使います。

```bash
python scripts/search.py "赤い車が雪道を走っている" --query-rewrite --ollama-url http://localhost:11434 --ollama-model qwen3.5:9b
```

thinking を有効にしているため、初回ロードや長い推論で時間がかかる場合があります。タイムアウトする場合は `--ollama-timeout` を大きくしてください。

```bash
python scripts/search.py "以前雪道を走ったよね？" --ollama-timeout 600
```

thinking が長くなりすぎる場合は、ステップごとに概算トークン上限を変更できます。
上限を超えた場合は、その時点までの thinking を直前の句点または改行までで区切り、`think=False` の新しい Ollama リクエストで最終出力だけを生成します。

デフォルト:

```text
--thinking-budget-decision 500
--thinking-budget-rewrite 500
--thinking-budget-answer 500
```

例:

```bash
python scripts/search.py "以前雪道を走ったよね？" --thinking-budget-answer 800
```

0 以下を指定すると、そのステップの thinking 上限を無効化します。

## 検索件数を変更する

`--top-k` を指定すると、保存する検索結果の数を変更できます。

```bash
python scripts/search.py "夕焼けの海辺を走る犬" --top-k 20
```

この場合、上位20件の画像が `results/実行時刻/` にコピーされます。

## 注意点

### 画像を追加したらインデックスを更新する

`images/` に画像を追加した場合は、検索前に以下を再実行してください。
既存の `data/images.faiss` と `data/image_paths.json` がある場合は、未登録の画像だけを末尾に追加します。

```bash
python scripts/build_index.py
```

全画像を最初から作り直す場合は `--rebuild` を指定します。

```bash
python scripts/build_index.py --rebuild
```

キャプションも追加画像分だけ生成する場合は、インデックス更新後に通常どおり実行します。
既存の `data/image_captions.jsonl` に含まれる画像は自動でスキップされます。

```bash
python scripts/generate_captions.py
```

## Reranker をドライブレコーダ画像に特化させる

`Qwen3-VL-Reranker-8B` を BDD100k のドライブレコーダ画像で QLoRA ファインチューニングし、
学習前後の精度を定量比較する一連のスクリプトです。

```text
scripts/build_reranker_dataset.py   学習・評価データセットを作る
scripts/train_reranker_qlora.py     QLoRA で学習する
scripts/evaluate_reranker.py        学習前後の精度を比較する
scripts/reranker_common.py          3つのスクリプトが共有するモデル読み込みとスコア計算
```

3つのスクリプトは `scripts/reranker_common.py` を通して、本番検索 `scripts/search.py` と
同じ命令文 (`DASHCAM_RERANKER_PROMPT`)、同じクエリ正規化、同じスコア計算
(`logit("yes") - logit("no")`) を使います。
学習・評価・本番で入力の作り方が一致するため、測った差はモデルの変化だけに帰属できます。

#### この評価で分かること・分からないこと

評価が測るのは **教師シーンカードの定義に対する適合度** です。本番検索の精度そのものではありません。

- 候補集合は test split (数百枚) の中から教師の構造化事実で選んだもので、
  本番の「10万枚から初段検索が返した上位50件を再ランキングする」設定とは異なります。
- 正解ラベルも学習データと同じ教師モデル・同じ方式で付いているため、
  教師が間違えた点については、その間違いへの適合度を測ることになります。

本番の再ランキング精度を測るには、別途これが必要です。

1. test クエリと正解を人手で確認して凍結する
2. 10万枚に対して本番の初段検索を実行し、実際の上位50候補を取り出す
3. その候補に対する人手の qrels で nDCG / MRR / Recall を算出する
4. 合成データは train / val に限定する

現状のスクリプトは 1〜3 を行いません。合成 test での改善は「学習が効いたか」の一次判定として使い、
本番投入の判断は上記の人手評価で行ってください。

### 1. データセットを作る

Ollama の教師モデルで、画像1枚ごとに「シーンカード」を生成します。
シーンカードは、画像に写っている事実を決められた語彙で構造化したものと、
その画像を検索するための日本語クエリを含みます。

```bash
python scripts/build_reranker_dataset.py --num-images 1000
```

デフォルトでは `images_100k/` から 1000 枚を決定的にサンプリングし、
`hf.co/unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q3_K_XL` でラベリングして
`datasets/dashcam_reranker_ft_v1/` へ書き出します。
RTX 4070 Ti SUPER (16GB) では1枚あたり20〜40秒、1000枚で6〜11時間程度かかります。
1件ずつ追記保存し、既にカードがある画像は自動でスキップするため、途中で止めても同じコマンドで再開できます。

VRAM 16GB では教師モデルのコンテキスト長 4096 が上限の目安です。
`CUDA error: out of memory` が出る場合は下げてください。

```bash
python scripts/build_reranker_dataset.py --num-images 1000 --num-ctx 2048
```

ラベリング済みのカードからペアだけ作り直す場合は、Ollama を呼ばない `pairs` ステージを実行します。

```bash
python scripts/build_reranker_dataset.py --stage pairs
```

#### ラベルの付き方

クエリの制約をすべて満たす画像が正例 (label=1)、満たさない画像が負例 (label=0) です。
判定は構造化された事実集合に対する決定的な照合で行うため、同じシーンカードからは常に同じラベルが再現されます。

制約は「シーン条件」と「対象物条件」に分かれます。
対象物条件は、クエリが言及している対象物ごとに画像内の対象物へ重複なく割り当てられるかを見ます。
これにより「右車線に車、左車線に別の車」のような組み合わせ条件を、対象物単位で正しく判定できます。

負例は2種類です。

- `hard_negative`: 制約の一部だけを満たす画像。満たした割合が高いものを優先して選ぶ
- `random_negative`: 制約をひとつも満たさない画像

hard negative には false negative が混ざり得ます。教師が対象物を見落としたり、
`motion` や車線位置を誤認した場合、本当は関連する画像を負例として学習することになります。
`--hard-negative-max-ratio` を 1.0 未満にすると、制約の充足率がその値を超える画像
(＝正例に近すぎる画像) を負例候補から完全に除外できます。既定の 1.0 は無効です。

```bash
python scripts/build_reranker_dataset.py --stage pairs --hard-negative-max-ratio 0.8
```

除外件数は `reports/dataset_stats.json` の `hard_negatives_dropped_by_margin` に記録されます。
各ペアには `satisfied_ratio` が入っているので、`scores_*.jsonl` と突き合わせれば
誤ラベルが実際に精度へ効いているかを後から確認できます。

#### ラベルを絞り込むオプション (既定は無効)

既定では教師モデルの出力をそのままラベル判定に使います。
まずこの素の状態で学習・評価し、誤りを見てから絞り込みを検討してください。

教師モデルは、クエリ文が述べていない条件まで制約に書き込むことがあります
(例:「自車の前方を走行している車」に `time_of_day=day` が付き、夜の該当画像が負例になる)。
これが誤ラベルとして効いていると分かった場合は `--ground-constraints` で、
クエリ文に対応する日本語表現が現れない条件を落とせます。
ただしこの判定は手書きのキーワード表で行うため、それ自体が誤差源になります
(「金色」を `yellow` に結び付けられない、など)。

`motion` は静止画1枚では「停止中」と「駐車中」を区別できず、
教師が同じ状況へ違うラベルを付けることがあります。
`--drop-motion-constraints` でラベル判定から外せます。

```bash
python scripts/build_reranker_dataset.py --stage pairs --ground-constraints --drop-motion-constraints
```

`pairs` ステージは数秒で終わり、シーンカードを作り直す必要はありません。
出力先を変えて両方の設定でデータセットを作れば、どちらが良いかを評価コードで実測できます。

```bash
python scripts/build_reranker_dataset.py --stage pairs \
  --dataset-dir datasets/dashcam_reranker_ft_v1_grounded \
  --scene-cards datasets/dashcam_reranker_ft_v1/raw_teacher/scene_cards.jsonl \
  --ground-constraints --drop-motion-constraints
```

`--scene-cards` で既存のラベリング結果を再利用するため、教師モデルを呼び直す必要はありません。

#### 分割

train / val / test は **動画グループ単位** で分割します (既定 0.7 / 0.1 / 0.2)。

BDD100k のファイル名は `<動画ID>-<フレームID>.jpg` で、`images_100k/` の 100,003 枚は
60,141 個の動画IDしか持ちません。つまり同じ動画の別フレームが多数含まれます (最大 48 枚)。
画像IDだけでシャッフルすると、ほぼ同じ景色が train と test の両方に現れて差を過大評価します
(既定の 1000 枚サンプルでは 6 グループ 12 枚が該当)。分割は動画ID単位で行い、
同じ動画のフレームは必ず同じ分割へ入れます。

クエリは生成元画像の分割に属し、候補画像も同じ分割のプールからのみ選ぶため、
画像もクエリも動画グループも分割をまたぎません。
リークがないことは `reports/split_isolation.json` に記録され、検出した場合はエラーで停止します。

#### 対象画像の絞り込み

`raw_teacher/scene_cards.jsonl` は追記式なので、`--num-images` や `--seed` を変えて
ラベリングし直すと前回のカードが残ります。`pairs` ステージは
`manifests/sampled_images.jsonl` に載っている画像のカードだけを使い、それ以外は除外します
(除外件数は実行時に表示されます)。複数回のサンプリング結果をまとめて1つのデータセットに
したい場合だけ `--ignore-manifest` を指定してください。

#### キャプション

各ペアには候補画像のキャプションも入り、`--use-caption` を付けたときにモデルへ渡されます。
既定は教師シーンカードの `caption_ja` です。本番の `data/image_captions.jsonl` と揃えたい場合は
`--caption-file` で同じ形式のファイルを指定します (ファイル名で結合し、カバー率を表示します)。

```bash
python scripts/build_reranker_dataset.py --stage pairs --caption-file data/image_captions.jsonl
```

なお `data/image_captions.jsonl` が持つのは `images/` の 1000 枚分だけで、
`images_100k/` から選んだ 1000 枚とは 43 枚しか重なりません。
本番キャプションで学習・評価するには、先に対象画像のキャプションを生成する必要があります。

生成物:

```text
datasets/dashcam_reranker_ft_v1/
├── manifests/sampled_images.jsonl      サンプリングした画像の一覧
├── raw_teacher/scene_cards.jsonl       教師モデルの出力 (再開の基準)
├── raw_teacher/scene_card_errors.jsonl 失敗した画像
├── derived/queries.jsonl               採用したクエリと制約
├── pairs.train.jsonl
├── pairs.val.jsonl
├── pairs.test.jsonl
└── reports/
    ├── dataset_stats.json              件数・正例率・難易度の内訳
    ├── split_isolation.json            分割間のリーク検査結果
    └── fact_coverage.json              事実の出現頻度
```

### 2. QLoRA で学習する

学習の前に、教師モデルを VRAM から降ろしてください。
Ollama はリクエスト後もしばらくモデルを保持するため、16GB では Reranker のロードが
`CUDA error: out of memory` で失敗します。

```bash
curl -s http://localhost:11434/api/chat -d '{"model":"hf.co/unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q3_K_XL","messages":[],"keep_alive":0}'
```

```bash
python scripts/train_reranker_qlora.py
```

bitsandbytes の 4bit NF4 でベースモデルを量子化し、LLM 側の線形層へ LoRA を入れます。
損失は、本番と同じスコア `logit("yes") - logit("no")` を logit とみなした二値交差エントロピーです。
既定の `--pos-weight 1.0` では、この損失は Qwen 公式の
「正例なら `yes`、負例なら `no` を出す非加重 NLL」と数学的に等価です。
クラス不均衡を補正したい場合は `--pos-weight 0` を指定すると 負例数/正例数 を自動計算します。
視覚エンコーダと merger は既定で凍結します。データ量が少ないうちは LLM 側だけで十分です。

学習開始前に val を1回測り、その値 (＝ベースモデルの精度) をチェックポイント選択の下限にします。
エポックごとに val を評価し、nDCG@5 がそれを超えて最良だったアダプタを `--output-dir` の直下へ、
最終エポックのアダプタを `last/` へ保存します。
**一度もベースラインを超えなかった場合、直下には何も保存しません。**
ベースモデル以下のアダプタを「学習済みモデル」として評価してしまうのを防ぐためです
(この場合 `last/` を明示的に指定すれば評価できます)。
実行開始時に、前回の直下アダプタは必ず削除します。

VRAM が足りない場合は画像の解像度を下げます (既定はモデルの `preprocessor_config.json` の値)。

```bash
python scripts/train_reranker_qlora.py --max-pixels 401408 --batch-size 1
```

主なオプション:

```bash
# 既定 (rank 32 / alpha 32、q k v o gate up down)
python scripts/train_reranker_qlora.py --lora-rank 32 --lora-alpha 32

# Qwen 公式が公開している対象層に厳密に合わせる (o_proj を含まない)
python scripts/train_reranker_qlora.py \
  --target-modules q_proj k_proj v_proj gate_proj up_proj down_proj

# ms-swift のサンプルに合わせた軽い設定
python scripts/train_reranker_qlora.py --lora-rank 8 --lora-alpha 32 --target-modules all-linear

# 視覚エンコーダにも LoRA を入れる
python scripts/train_reranker_qlora.py --train-vision-tower

# 数十ペアだけで配線を確認する
python scripts/train_reranker_qlora.py --max-train-pairs 16 --max-val-pairs 8 --epochs 1
```

### 3. 学習前後の精度を比較する

```bash
python scripts/evaluate_reranker.py
```

同じプロセス内で1つのモデルをロードし、LoRA アダプタを無効にした状態 (base) と
有効にした状態 (adapter) の両方で test を採点します。
量子化・画像解像度・命令文・クエリ正規化がすべて共通なので、差分をモデルの変化だけに帰属できます。

出力する指標:

- ランキング: nDCG@1 / @3 / @5 / @10、MRR、MAP、Recall@1 / @3 / @5
- 二値識別: ROC-AUC、PR-AUC、スコア0を閾値とした正解率・適合率・再現率
- `hard_negative_error_rate`: 正例より高いスコアが付いた hard negative の割合

base と adapter の差は、**動画グループ単位** のペアード・クラスタ・ブートストラップで
95% 信頼区間と p 値を出します。信頼区間が 0 をまたぐ場合、その差は統計的に有意ではありません。

1枚の画像から最大2件のクエリが作られ、同じ動画の別フレームも似た候補集合を共有するため、
クエリ単位で復元抽出すると相関を無視して信頼区間が実際より狭くなります
(グループ内が完全相関の合成例では、クエリ単位の CI 幅がクラスタ単位の約 6 割になりました)。
抽出の単位は生成元画像の動画IDです。

学習前のベースラインだけを測る場合:

```bash
python scripts/evaluate_reranker.py --no-adapter
```

`--quantization` は既定で本番検索と同じ 8bit です。base と adapter は必ず同じ設定で測ります。

結果は `datasets/dashcam_reranker_ft_v1/reports/` へ保存されます。

```text
eval_test_20260810_153012.md      比較表と判定
eval_test_20260810_153012.json    全指標とブートストラップ統計
scores_test_20260810_153012.jsonl ペアごとの base / adapter スコア
```

`scores_*.jsonl` を見ると、どのクエリでどの画像の順位が入れ替わったかを個別に確認できます。

### 4. 学習したアダプタを検索で使う

`scripts/search.py` は現時点ではベースモデルだけを読み込みます。
アダプタを本番検索へ入れる場合は、Reranker のロード後に
`reranker_common.attach_adapter(model, adapter_path)` を呼んでください。
transformers の `load_adapter` は内部でモデル全体分の VRAM を先取りしようとするため、
16GB では量子化済みモデルの上で失敗します。`attach_adapter` はこれを避ける実装です。

## よく使うコマンド

インデックス作成:

```bash
python scripts/build_index.py
```

検索:

```bash
python scripts/search.py "赤い車が雪道を走っている"
```

上位20件を検索:

```bash
python scripts/search.py "赤い車が雪道を走っている" --top-k 20
```

## 出力例

```bash
python scripts/search.py "赤い車が雪道を走っている"
```

出力例:

```text
raw query: 赤い車が雪道を走っている
needs image search: Yes
search query: 赤い車が雪道を走っている
results: results/20260424_153012

01  score=0.8123  images/car_001.jpg -> results/20260424_153012/01_0.8123_car_001.jpg
02  score=0.7991  images/snow_045.jpg -> results/20260424_153012/02_0.7991_snow_045.jpg
03  score=0.7814  images/vehicle_120.jpg -> results/20260424_153012/03_0.7814_vehicle_120.jpg
```

## 処理の流れ

1. `scripts/build_index.py` で画像をベクトル化する
2. ベクトルを `data/images.faiss` に保存する
3. 画像パスを `data/image_paths.json` に保存する
4. `scripts/search.py` でユーザ入力を Ollama に渡し、回答に過去画像データベースの参照が必要かを判定する
5. 画像検索が不要な場合は、検索せずに Ollama で通常回答する（`--skip-answer-generation` 指定時は回答生成もスキップする）
6. 画像検索が必要で `--query-rewrite` 指定時は、Ollama で画像検索向けクエリへ変換する
7. 実際に検索へ使うクエリをベクトル化する
8. FAISS で近い画像ベクトルを検索する
9. 上位画像を `results/実行時刻/` にコピーする
10. 最上位の検索結果画像を Ollama に渡して回答を生成し、`llm_response.txt` に保存する（`--skip-answer-generation` 指定時はスキップする）
11. Ollama の thinking を `ollama_thinking.txt` と `ollama_thinking.json` に保存する
12. 元の入力を `raw_query.txt`、検索に使ったクエリを `query.txt` に保存する
