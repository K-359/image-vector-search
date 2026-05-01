import argparse
import base64
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
OLLAMA_TIMEOUT = 300.0
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


def clean_yes_no(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    normalized = text.strip(" \t\r\n\"'「」.。").lower()
    if normalized.startswith("yes"):
        return "Yes"
    if normalized.startswith("no"):
        return "No"

    raise RuntimeError(f"Ollama の画像検索要否判定が Yes/No ではありませんでした: {text!r}")


def chat_with_ollama(
    messages: list[dict],
    *,
    model_name: str,
    base_url: str,
    timeout: float,
    error_context: str,
    stream_callback=None,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream_callback is not None,
        "think": True,
    }

    request = urllib.request.Request(
        f"{normalize_ollama_base_url(base_url)}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if stream_callback is None:
                response_body = response.read().decode("utf-8")
            else:
                content_parts = []
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama の{error_context}中にエラーが発生しました: {chunk['error']}")

                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        content_parts.append(content)
                        stream_callback(content)

                content = "".join(content_parts).strip()
                if not content:
                    raise RuntimeError(f"Ollama の{error_context}結果が空でした。")

                return content
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Ollama で{error_context}できませんでした。"
            f" Ollama が起動しているか、モデル {model_name!r} が利用可能か確認してください。"
            " 推論に時間がかかる場合は --ollama-timeout を大きくしてください。"
        ) from exc

    data = json.loads(response_body)
    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError(f"Ollama の{error_context}結果が空でした。")

    return content


def decide_image_search_with_ollama(
    raw_query: str,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
) -> bool:
    system_prompt = """
あなたは、ユーザクエリへ正しく回答するために過去の画像データベースを参照する必要があるかを判定します。

Yes にする条件:
- ユーザクエリの回答に、過去の画像データベース内の画像内容を参照する必要がある

No にする条件:
- 一般知識、説明、翻訳、文章作成、挨拶、雑談など、過去の画像データベースを参照しなくても答えられる

例:
- 「以前雪道を走ったよね？」 -> Yes
- 「赤い車の画像を探して」 -> Yes
- 「こんにちは」 -> No

出力は Yes または No のどちらか1語だけにしてください。
""".strip()
    user_prompt = f"ユーザクエリ:\n{raw_query}"
    response = chat_with_ollama(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        error_context="画像検索要否判定",
    )

    return clean_yes_no(response) == "Yes"


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

    response = chat_with_ollama(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        error_context="検索クエリ変換",
    )
    rewritten_query = clean_llm_query(response)

    if not rewritten_query:
        raise RuntimeError("Ollama の検索クエリ変換結果が空でした。")

    return rewritten_query


def answer_with_ollama(
    raw_query: str,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
    image_path: Path | None = None,
    image_score: float | None = None,
    stream_callback=None,
) -> str:
    system_prompt = """
あなたはユーザの質問に日本語で答えるアシスタントです。
画像が渡された場合は、その画像を根拠としてユーザの質問に答えてください。
画像が渡されていない場合は、通常のテキスト質問として自然に答えてください。
""".strip()

    if image_path is None:
        user_message = {
            "role": "user",
            "content": f"ユーザクエリ:\n{raw_query}",
        }
    else:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("ascii")

        score_text = "" if image_score is None else f"\n画像検索スコア: {image_score:.4f}"
        user_message = {
            "role": "user",
            "content": (
                "ユーザクエリ:\n"
                f"{raw_query}\n\n"
                "添付画像は画像検索で最もスコアが高かった結果です。"
                f"{score_text}"
            ),
            "images": [image_base64],
        }

    return chat_with_ollama(
        [
            {"role": "system", "content": system_prompt},
            user_message,
        ],
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        error_context="回答生成",
        stream_callback=stream_callback,
    )


def safe_score(score: float) -> str:
    """
    ファイル名に入れやすい形でスコアを文字列化する。
    例: 0.823456 -> 0.8235
    """
    return f"{score:.4f}"


def find_best_existing_image(
    results: list[tuple[float, int]],
    image_paths: list[str],
) -> tuple[float, int, Path] | None:
    for score, image_id in results:
        if image_id < 0:
            continue

        src_path = Path(image_paths[image_id])
        if src_path.exists():
            return float(score), int(image_id), src_path

    return None


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
        help="画像検索が必要な場合に、Ollama で入力を画像検索向けクエリに変換してから検索する。",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL_NAME,
        help=(
            "画像検索要否判定、必要時の検索クエリ変換、回答生成に使う Ollama モデル。"
            f"デフォルト: {OLLAMA_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", OLLAMA_BASE_URL),
        help=f"Ollama API のURL。デフォルト: 環境変数 OLLAMA_HOST または {OLLAMA_BASE_URL}",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=OLLAMA_TIMEOUT,
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

    index = None
    image_paths = None
    model = None

    def load_search_backend():
        nonlocal index, image_paths, model

        if index is not None and image_paths is not None and model is not None:
            return index, image_paths, model

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

        return index, image_paths, model

    if args.interactive:
        load_search_backend()
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
        needs_image_search = decide_image_search_with_ollama(
            raw_query,
            model_name=args.ollama_model,
            base_url=args.ollama_url,
            timeout=args.ollama_timeout,
        )

        print(f"raw query: {raw_query}")
        print(f"needs image search: {'Yes' if needs_image_search else 'No'}")

        if not needs_image_search:
            print()
            print("LLM response:")
            llm_response = answer_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
                stream_callback=(lambda chunk: print(chunk, end="", flush=True)) if args.interactive else None,
            )
            if args.interactive:
                print()
            else:
                print(llm_response)

            if not args.interactive:
                break
            print()
            continue

        if args.query_rewrite:
            search_query = rewrite_query_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
            )

        index, image_paths, model = load_search_backend()

        query_embedding = model.encode(
            [search_query],
            prompt="Retrieve images relevant to the user's query.",
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        result_count = index.ntotal
        scores, ids = index.search(query_embedding, result_count)
        top_results = list(zip(scores[0][: args.top_k], ids[0][: args.top_k]))
        bottom_results = (
            [] if args.bottom_k == 0 else list(zip(scores[0], ids[0]))[-args.bottom_k :]
        )
        best_image = find_best_existing_image(list(zip(scores[0], ids[0])), image_paths)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = RESULTS_DIR / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        # 後方互換のため、query.txt には実際に検索へ使ったクエリを保存する。
        with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
            f.write(search_query + "\n")

        with open(result_dir / "raw_query.txt", "w", encoding="utf-8") as f:
            f.write(raw_query + "\n")

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

        if best_image is None:
            print()
            print("LLM response:")
            llm_response = answer_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
                stream_callback=(lambda chunk: print(chunk, end="", flush=True)) if args.interactive else None,
            )
        else:
            best_score, _, best_path = best_image
            print()
            print("LLM response:")
            llm_response = answer_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
                image_path=best_path,
                image_score=best_score,
                stream_callback=(lambda chunk: print(chunk, end="", flush=True)) if args.interactive else None,
            )

        with open(result_dir / "llm_response.txt", "w", encoding="utf-8") as f:
            f.write(llm_response + "\n")

        if args.interactive:
            print()
        else:
            print(llm_response)

        if not args.interactive:
            break
        print()


if __name__ == "__main__":
    main()
