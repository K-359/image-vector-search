# Text-Based Image Retrieval with Qwen3-VL-Embedding-2B

`images/` ディレクトリ内の画像から、入力テキストに意味的に近い画像を検索するプログラムです。

Qwen3-VL-Embedding-2B を使って、画像とテキストを同じ埋め込み空間のベクトルに変換し、FAISS で類似検索します。

## ディレクトリ構成

```text
.
├── images/
│   ├── sample001.jpg
│   ├── sample002.jpg
│   └── ...
├── data/
│   ├── images.faiss
│   └── image_paths.json
├── results/
│   └── 20260424_153012/
│       ├── query.txt
│       ├── 01_0.8123_sample001.jpg
│       └── ...
├── scripts/
│   ├── build_index.py
│   └── search.py
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

画像を追加・削除・変更した場合は、再度このコマンドを実行してください。

## テキストで画像を検索する

以下のように検索します。

```bash
python scripts/search.py "赤い車が雪道を走っている"
```

デフォルトでは上位10件を検索します。

検索結果は標準出力に表示されるだけでなく、`results/` ディレクトリにもコピーされます。

例:

```text
results/
└── 20260424_153012/
    ├── query.txt
    ├── raw_query.txt
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

検索時は、まず Ollama の `qwen3.5:9b` が回答に過去画像データベースの参照が必要かを `Yes` / `No` で判定します。
`Yes` の場合は画像検索し、最上位の検索結果画像を LLM に渡して回答を生成します。
`No` の場合は画像検索を行わず、LLM が通常のテキスト質問として回答します。
検索結果画像を使って回答させるには、`--ollama-model` に画像入力へ対応した Ollama モデルを指定してください。
`--interactive` では LLM 回答がストリーミング表示されます。

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

## 検索件数を変更する

`--top-k` を指定すると、保存する検索結果の数を変更できます。

```bash
python scripts/search.py "夕焼けの海辺を走る犬" --top-k 20
```

この場合、上位20件の画像が `results/実行時刻/` にコピーされます。

## 注意点

### 画像を変更したらインデックスを作り直す

`images/` の中身を変更した場合は、検索前に以下を再実行してください。

```bash
python scripts/build_index.py
```

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
5. 画像検索が不要な場合は、検索せずに Ollama で通常回答する
6. 画像検索が必要で `--query-rewrite` 指定時は、Ollama で画像検索向けクエリへ変換する
7. 実際に検索へ使うクエリをベクトル化する
8. FAISS で近い画像ベクトルを検索する
9. 上位画像を `results/実行時刻/` にコピーする
10. 最上位の検索結果画像を Ollama に渡して回答を生成し、`llm_response.txt` に保存する
11. 元の入力を `raw_query.txt`、検索に使ったクエリを `query.txt` に保存する
