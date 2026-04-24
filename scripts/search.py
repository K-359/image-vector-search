import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def safe_score(score: float) -> str:
    """
    ファイル名に入れやすい形でスコアを文字列化する。
    例: 0.823456 -> 0.8235
    """
    return f"{score:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="検索したい内容。例: 夕焼けの海辺を走る犬")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    index_path = DATA_DIR / "images.faiss"
    paths_path = DATA_DIR / "image_paths.json"

    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} がありません。先に scripts/build_index.py を実行してください。")

    if not paths_path.exists():
        raise FileNotFoundError(f"{paths_path} がありません。先に scripts/build_index.py を実行してください。")

    index = faiss.read_index(str(index_path))

    with open(paths_path, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [args.query],
        prompt="Retrieve images relevant to the user's query.",
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(query_embedding, args.top_k)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    # 入力テキストを保存
    with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
        f.write(args.query + "\n")

    print(f"query: {args.query}")
    print(f"results: {result_dir}")
    print()

    for rank, (score, image_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if image_id < 0:
            continue

        src_path = Path(image_paths[image_id])

        if not src_path.exists():
            print(f"{rank:02d}  score={score:.4f}  MISSING: {src_path}")
            continue

        dst_name = f"{rank:02d}_{safe_score(float(score))}_{src_path.name}"
        dst_path = result_dir / dst_name

        shutil.copy2(src_path, dst_path)

        print(f"{rank:02d}  score={score:.4f}  {src_path} -> {dst_path}")


if __name__ == "__main__":
    main()
