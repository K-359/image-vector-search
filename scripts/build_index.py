from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
IMAGE_DIR = Path("images")
OUT_DIR = Path("data")
BATCH_SIZE = 8


def main():
    OUT_DIR.mkdir(exist_ok=True)

    image_paths = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    if not image_paths:
        raise RuntimeError("images/ に画像が見つかりません。")

    model = SentenceTransformer(MODEL_NAME)

    index = None
    stored_paths = []

    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Embedding images"):
        batch_paths = image_paths[start : start + BATCH_SIZE]

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

    faiss.write_index(index, str(OUT_DIR / "images.faiss"))

    with open(OUT_DIR / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(stored_paths, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(stored_paths)} images")
    print(f"Saved: {OUT_DIR / 'images.faiss'}")
    print(f"Saved: {OUT_DIR / 'image_paths.json'}")


if __name__ == "__main__":
    main()
