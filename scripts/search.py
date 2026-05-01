import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    import readline  # noqa: F401
except ImportError:
    pass


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
OLLAMA_MODEL_NAME = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def normalize_ollama_base_url(base_url: str) -> str:
    """
    OLLAMA_HOST が 127.0.0.1:11434 のようにスキームなしで渡された場合も扱う。
    """
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        return OLLAMA_BASE_URL
    if "://" not in base_url:
        return f"http://{base_url}"
    return base_url


def clean_llm_query(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    prefixes = (
        "検索クエリ:",
        "画像検索クエリ:",
        "query:",
        "Query:",
        "SEARCH QUERY:",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    return text.strip(" \t\r\n\"'「」")


def rewrite_query_with_ollama(
    raw_query: str,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
) -> str:
    system_prompt = """
あなたは画像検索システムのためのクエリ変換器です。
ユーザの生クエリを、画像埋め込み検索に適した短い日本語の視覚描写へ変換してください。

ルール:
- 出力は検索クエリ本文だけにする
- 1文または短い名詞句にする
- 画像に写っていそうな被写体、場所、行動、色、構図、雰囲気を優先する
- 会話上の依頼、挨拶、検索操作への指示、不要な助詞や曖昧な言い回しは取り除く
- 画像に見えない意図、評価、推測、メタ情報は足さない
- 情報が少ない場合は、元クエリの視覚的な語を保ったまま簡潔にする
- 翻訳や説明はしない
""".strip()
    user_prompt = f"ユーザの生クエリ:\n{raw_query}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
    }

    request = urllib.request.Request(
        f"{normalize_ollama_base_url(base_url)}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama で検索クエリを変換できませんでした。"
            f" Ollama が起動しているか、モデル {model_name!r} が利用可能か確認してください。"
        ) from exc

    data = json.loads(response_body)
    rewritten_query = clean_llm_query(data.get("message", {}).get("content", ""))

    if not rewritten_query:
        raise RuntimeError("Ollama の検索クエリ変換結果が空でした。")

    return rewritten_query


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
    parser.add_argument(
        "--query-rewrite",
        action="store_true",
        help="Ollama で入力を画像検索向けクエリに変換してから検索する。",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL_NAME,
        help=f"検索クエリ変換に使う Ollama モデル。デフォルト: {OLLAMA_MODEL_NAME}",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", OLLAMA_BASE_URL),
        help=f"Ollama API のURL。デフォルト: 環境変数 OLLAMA_HOST または {OLLAMA_BASE_URL}",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=60.0,
        help="Ollama API のタイムアウト秒数。",
    )
    args = parser.parse_args()

    if not args.interactive and not args.query:
        parser.error("query を指定してください。--interactive の場合は省略できます。")
    if args.top_k < 0:
        parser.error("--top-k は 0 以上を指定してください。")
    if args.bottom_k < 0:
        parser.error("--bottom-k は 0 以上を指定してください。")
    if args.ollama_timeout <= 0:
        parser.error("--ollama-timeout は 0 より大きい値を指定してください。")

    index_path = DATA_DIR / "images.faiss"
    paths_path = DATA_DIR / "image_paths.json"

    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} がありません。先に scripts/build_index.py を実行してください。")

    if not paths_path.exists():
        raise FileNotFoundError(f"{paths_path} がありません。先に scripts/build_index.py を実行してください。")

    import faiss
    from sentence_transformers import SentenceTransformer

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

        raw_query = query
        search_query = raw_query
        if args.query_rewrite:
            search_query = rewrite_query_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
            )

        query_embedding = model.encode(
            [search_query],
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

        # 後方互換のため、query.txt には実際に検索へ使ったクエリを保存する。
        with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
            f.write(search_query + "\n")

        with open(result_dir / "raw_query.txt", "w", encoding="utf-8") as f:
            f.write(raw_query + "\n")

        print(f"raw query: {raw_query}")
        print(f"search query: {search_query}")
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
