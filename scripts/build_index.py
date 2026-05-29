import argparse
from pathlib import Path
import json


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
IMAGE_DIR = Path("images")
OUT_DIR = Path("data")
INDEX_FILENAME = "images.faiss"
PATHS_FILENAME = "image_paths.json"
BATCH_SIZE = 8


def find_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )


def load_stored_paths(paths_path: Path) -> list[str]:
    with open(paths_path, "r", encoding="utf-8") as f:
        paths = json.load(f)

    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise RuntimeError(f"{paths_path} は画像パス文字列のリストである必要があります。")

    if len(paths) != len(set(paths)):
        raise RuntimeError(f"{paths_path} に重複した画像パスがあります。")

    return paths


def load_existing_index(index_path: Path, paths_path: Path, rebuild: bool):
    import faiss

    if rebuild:
        return None, []

    index_exists = index_path.exists()
    paths_exists = paths_path.exists()
    if not index_exists and not paths_exists:
        return None, []

    if index_exists != paths_exists:
        raise RuntimeError(
            f"{index_path} と {paths_path} は両方存在する必要があります。"
            " 片方だけ存在する状態から作り直す場合は --rebuild を指定してください。"
        )

    index = faiss.read_index(str(index_path))
    stored_paths = load_stored_paths(paths_path)
    if index.ntotal != len(stored_paths):
        raise RuntimeError(
            f"{index_path} の件数 ({index.ntotal}) と {paths_path} の件数"
            f" ({len(stored_paths)}) が一致しません。"
            " 作り直す場合は --rebuild を指定してください。"
        )

    return index, stored_paths


def encode_and_add(index, stored_paths: list[str], pending_paths: list[Path]):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    model = SentenceTransformer(MODEL_NAME)

    for start in tqdm(range(0, len(pending_paths), BATCH_SIZE), desc="Embedding images"):
        batch_paths = pending_paths[start : start + BATCH_SIZE]

        # Qwen3-VL系は画像入力を扱える。ローカル画像は {"image": path} 形式にする。
        batch_inputs = [{"image": str(p)} for p in batch_paths]

        embeddings = model.encode(
            batch_inputs,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if index is None:
            dim = embeddings.shape[1]
            base_index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(base_index)

        ids = np.arange(len(stored_paths), len(stored_paths) + len(batch_paths)).astype("int64")
        index.add_with_ids(embeddings, ids)

        stored_paths.extend(str(p) for p in batch_paths)

    return index


def main():
    parser = argparse.ArgumentParser(
        description="images/ の画像をベクトル化し、FAISSインデックスを作成または追記します。"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_DIR,
        help=f"画像ディレクトリ。デフォルト: {IMAGE_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"出力ディレクトリ。デフォルト: {OUT_DIR}",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="既存のインデックスと画像パス一覧を使わず、全画像を最初から作り直します。",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    index_path = args.out_dir / INDEX_FILENAME
    paths_path = args.out_dir / PATHS_FILENAME

    image_paths = find_image_paths(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"{args.image_dir}/ に画像が見つかりません。")

    index, stored_paths = load_existing_index(index_path, paths_path, args.rebuild)
    stored_path_set = set(stored_paths)
    pending_paths = [path for path in image_paths if str(path) not in stored_path_set]

    print(f"Images found: {len(image_paths)}")
    print(f"Already indexed: {len(stored_paths)}")
    print(f"Pending: {len(pending_paths)}")

    if not pending_paths:
        print("No new images to index.")
        return

    index = encode_and_add(index, stored_paths, pending_paths)

    import faiss

    faiss.write_index(index, str(index_path))

    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(stored_paths, f, ensure_ascii=False, indent=2)

    print(f"Indexed total: {len(stored_paths)}")
    print(f"Added: {len(pending_paths)}")
    print(f"Saved: {index_path}")
    print(f"Saved: {paths_path}")


if __name__ == "__main__":
    main()
