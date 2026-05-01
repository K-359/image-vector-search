import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import readline  # noqa: F401
except ImportError:
    pass


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
    parser.add_argument("query", nargs="?", type=str, help="検索したい内容。例: 夕焼けの海辺を走る犬")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10, help="下位検索結果として出力する件数。")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="モデルとインデックスを読み込んだまま、標準入力から複数回検索する。",
    )
    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("query を指定してください。--interactive の場合は省略できます。")
    if args.top_k < 0:
        parser.error("--top-k は 0 以上を指定してください。")
    if args.bottom_k < 0:
        parser.error("--bottom-k は 0 以上を指定してください。")

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

    if args.interactive:
        print("検索文を入力してください。終了するには空行または Ctrl-D を入力してください。")

    while True:
        if args.interactive:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if not query:
                break
        else:
            query = args.query

        query_embedding = model.encode(
            [query],
            prompt="Retrieve images relevant to the user's query.",
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        result_count = index.ntotal
        scores, ids = index.search(query_embedding, result_count)
        top_results = list(zip(scores[0][: args.top_k], ids[0][: args.top_k]))
        bottom_results = [] if args.bottom_k == 0 else list(zip(scores[0], ids[0]))[-args.bottom_k :]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = RESULTS_DIR / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        # 入力テキストを保存
        with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
            f.write(query + "\n")

        print(f"query: {query}")
        print(f"results: {result_dir}")
        print()

        print("top results:")
        for rank, (score, image_id) in enumerate(top_results, start=1):
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

        print()
        print("bottom results:")
        for rank, (score, image_id) in enumerate(reversed(bottom_results), start=1):
            if image_id < 0:
                continue

            src_path = Path(image_paths[image_id])

            if not src_path.exists():
                print(f"{rank:02d}  score={score:.4f}  MISSING: {src_path}")
                continue

            dst_name = f"bottom_{rank:02d}_{safe_score(float(score))}_{src_path.name}"
            dst_path = result_dir / dst_name

            shutil.copy2(src_path, dst_path)

            print(f"{rank:02d}  score={score:.4f}  {src_path} -> {dst_path}")

        if not args.interactive:
            break
        print()


if __name__ == "__main__":
    main()
