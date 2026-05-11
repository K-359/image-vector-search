import argparse
import base64
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
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
THINKING_BUDGET_TOKENS = {
    "image_search_decision": 500,
    "query_rewrite": 500,
    "answer_generation": 1000,
}
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
IMAGE_DATES_PATH = DATA_DIR / "image_dates.json"


@dataclass
class OllamaChatResult:
    content: str
    thinking: str
    thinking_budget_exceeded: bool = False
    fallback_used: bool = False


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


def split_think_tags(content: str, thinking: str) -> OllamaChatResult:
    """
    Ollama の現行 API は thinking を message.thinking に分離する。
    古いモデル/テンプレートで <think>...</think> が本文へ混ざる場合も分離する。
    """
    match = re.search(r"<think>\s*(.*?)\s*</think>", content, flags=re.DOTALL)
    if not match:
        return OllamaChatResult(content=content.strip(), thinking=thinking.strip())

    extracted_thinking = match.group(1).strip()
    cleaned_content = (content[: match.start()] + content[match.end() :]).strip()
    combined_thinking = "\n\n".join(
        part for part in (thinking.strip(), extracted_thinking) if part
    )
    return OllamaChatResult(content=cleaned_content, thinking=combined_thinking)


def estimate_token_count(text: str) -> int:
    """
    厳密な Ollama/Qwen トークナイザではなく、ストリーム中に軽く判定するための概算。
    CJK文字は1文字1トークン、それ以外は単語・記号単位で数える。
    """
    cjk_chars = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text)
    non_cjk_text = re.sub(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", " ", text)
    non_cjk_tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_cjk_text)
    return len(cjk_chars) + len(non_cjk_tokens)


def trim_thinking_to_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    sentence_end = max(
        text.rfind(mark)
        for mark in ("。", "．", ".", "!", "?", "！", "？", "\n")
    )
    if sentence_end <= 0:
        return text

    return text[: sentence_end + 1].strip()


def make_fallback_messages(messages: list[dict], thinking: str, error_context: str) -> list[dict]:
    fallback_instruction = f"""
途中までのthinking:
{thinking}

上の途中thinkingを参考にしてください。
これ以上thinkingせず、{error_context}の最終出力だけを返してください。
元の出力形式の制約を必ず守ってください。
""".strip()
    return [*messages, {"role": "user", "content": fallback_instruction}]


def append_thinking_record(
    records: list[dict] | None,
    *,
    step: str,
    result: OllamaChatResult,
) -> None:
    if records is None:
        return

    records.append(
        {
            "step": step,
            "thinking": result.thinking,
            "response": result.content,
            "thinking_budget_exceeded": result.thinking_budget_exceeded,
            "fallback_used": result.fallback_used,
        }
    )


def make_result_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / timestamp
    suffix = 2

    while result_dir.exists():
        result_dir = RESULTS_DIR / f"{timestamp}_{suffix}"
        suffix += 1

    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def write_ollama_thinking_files(result_dir: Path, records: list[dict]) -> None:
    with open(result_dir / "ollama_thinking.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(result_dir / "ollama_thinking.txt", "w", encoding="utf-8") as f:
        for index, record in enumerate(records, start=1):
            if index > 1:
                f.write("\n\n")

            f.write(f"# {record['step']}\n")
            if record.get("thinking_budget_exceeded"):
                f.write(
                    f"(thinking budget exceeded; think=False fallback used: "
                    f"{'yes' if record.get('fallback_used') else 'no'})\n"
                )
            thinking = record.get("thinking") or ""
            if thinking:
                f.write(thinking.strip() + "\n")
            else:
                f.write("(thinking は空でした)\n")


def create_thinking_chunk_writer(result_dir: Path, step: str):
    output_path = result_dir / "ollama_thinking.txt"
    section_started = False

    def write_chunk(chunk: str) -> None:
        nonlocal section_started

        needs_separator = output_path.exists() and output_path.stat().st_size > 0
        with open(output_path, "a", encoding="utf-8") as f:
            if not section_started:
                if needs_separator:
                    f.write("\n\n")
                f.write(f"# {step}\n")
                section_started = True
            f.write(chunk)
            f.flush()

    return write_chunk


def chat_with_ollama(
    messages: list[dict],
    *,
    model_name: str,
    base_url: str,
    timeout: float,
    error_context: str,
    stream_callback=None,
    thinking_callback=None,
    think: bool = True,
    thinking_budget_tokens: int | None = None,
) -> OllamaChatResult:
    stream_response = (
        stream_callback is not None
        or thinking_callback is not None
        or thinking_budget_tokens is not None
    )
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream_response,
        "think": think,
    }

    request = urllib.request.Request(
        f"{normalize_ollama_base_url(base_url)}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not stream_response:
                response_body = response.read().decode("utf-8")
            else:
                content_parts = []
                thinking_parts = []
                thinking_budget_exceeded = False
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(f"Ollama の{error_context}中にエラーが発生しました: {chunk['error']}")

                    message = chunk.get("message", {})
                    thinking = message.get("thinking", "") or chunk.get("thinking", "")
                    if thinking:
                        thinking_parts.append(thinking)
                        if thinking_callback is not None:
                            thinking_callback(thinking)

                        if (
                            thinking_budget_tokens is not None
                            and estimate_token_count("".join(thinking_parts)) >= thinking_budget_tokens
                        ):
                            thinking_budget_exceeded = True
                            break

                    content = message.get("content", "")
                    if content:
                        content_parts.append(content)
                        if stream_callback is not None:
                            stream_callback(content)

                if thinking_budget_exceeded:
                    thinking = trim_thinking_to_sentence("".join(thinking_parts))
                    fallback_result = chat_with_ollama(
                        make_fallback_messages(messages, thinking, error_context),
                        model_name=model_name,
                        base_url=base_url,
                        timeout=timeout,
                        error_context=error_context,
                        stream_callback=stream_callback,
                        thinking_callback=None,
                        think=False,
                    )
                    return OllamaChatResult(
                        content=fallback_result.content,
                        thinking=thinking,
                        thinking_budget_exceeded=True,
                        fallback_used=True,
                    )

                result = split_think_tags(
                    "".join(content_parts).strip(),
                    "".join(thinking_parts).strip(),
                )
                if not result.content:
                    raise RuntimeError(f"Ollama の{error_context}結果が空でした。")

                return result
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Ollama で{error_context}できませんでした。"
            f" Ollama が起動しているか、モデル {model_name!r} が利用可能か確認してください。"
            " 推論に時間がかかる場合は --ollama-timeout を大きくしてください。"
        ) from exc

    data = json.loads(response_body)
    message = data.get("message", {})
    result = split_think_tags(
        message.get("content", "").strip(),
        message.get("thinking", "").strip(),
    )
    if not result.content:
        raise RuntimeError(f"Ollama の{error_context}結果が空でした。")

    return result


def decide_image_search_with_ollama(
    raw_query: str,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
    thinking_log: list[dict] | None = None,
    thinking_callback=None,
    thinking_budget_tokens: int | None = None,
) -> bool:
    system_prompt = """
運転中のユーザが車内で話すクエリを想定してください。
ユーザクエリへの回答に、過去のドライブレコーダから撮影した画像データが必要かどうか判定してください。
迷う場合はYesを優先してください。

例:
- 「さっきのバス危なかったね」 -> Yes
- 「運転疲れた」 -> No

出力は Yes または No のどちらか1語だけにしてください。
""".strip()
    user_prompt = f"ユーザクエリ:\n{raw_query}"
    result = chat_with_ollama(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        error_context="画像検索要否判定",
        thinking_callback=thinking_callback,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    append_thinking_record(thinking_log, step="image_search_decision", result=result)

    return clean_yes_no(result.content) == "Yes"


def rewrite_query_with_ollama(
    raw_query: str,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
    thinking_log: list[dict] | None = None,
    thinking_callback=None,
    thinking_budget_tokens: int | None = None,
) -> str:
    system_prompt = """
ユーザのクエリを、画像埋め込み検索に適したクエリに変換してください。

ルール:
- 出力は検索クエリ本文だけにする
- 1文または短い名詞句にする
- 会話上の依頼、挨拶、検索操作への指示、不要な助詞や曖昧な言い回しは取り除く
""".strip()
    user_prompt = f"ユーザのクエリ:\n{raw_query}"

    result = chat_with_ollama(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        error_context="検索クエリ変換",
        thinking_callback=thinking_callback,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    append_thinking_record(thinking_log, step="query_rewrite", result=result)
    rewritten_query = clean_llm_query(result.content)

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
    image_date: str | None = None,
    stream_callback=None,
    thinking_callback=None,
    thinking_budget_tokens: int | None = None,
) -> OllamaChatResult:
    system_prompt = """
ユーザクエリに日本語で回答してください。
運転中のユーザが車内で話すクエリを想定してください。
ユーザクエリを元に検索された、過去の車外画像が渡される場合があります。
""".strip()

    if image_path is None:
        user_message = {
            "role": "user",
            "content": f"ユーザクエリ:\n{raw_query}",
        }
    else:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("ascii")

        content = f"ユーザクエリ:\n{raw_query}"
        if image_date is not None:
            content = f"検索された画像の日付: {image_date}\n\n{content}"

        user_message = {
            "role": "user",
            "content": content,
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
        thinking_callback=thinking_callback,
        thinking_budget_tokens=thinking_budget_tokens,
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


def load_image_dates(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    dates_by_path = payload.get("dates_by_path") if isinstance(payload, dict) else None
    if not isinstance(dates_by_path, dict):
        raise RuntimeError(f"{path} は dates_by_path を持つJSONオブジェクトである必要があります。")

    invalid_items = [
        (image_path, image_date)
        for image_path, image_date in dates_by_path.items()
        if not isinstance(image_path, str) or not isinstance(image_date, str)
    ]
    if invalid_items:
        raise RuntimeError(f"{path} に文字列ではないパスまたは日付が含まれています。")

    return dates_by_path


def format_date_field(image_path: Path, image_dates: dict[str, str] | None) -> str:
    if image_dates is None:
        return ""

    image_date = image_dates.get(str(image_path))
    if image_date is None:
        return "  date=UNKNOWN"

    return f"  date={image_date}"


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
    parser.add_argument(
        "--thinking-budget-decision",
        type=int,
        default=THINKING_BUDGET_TOKENS["image_search_decision"],
        help="画像検索要否判定の thinking 概算トークン上限。0 以下で無制限。",
    )
    parser.add_argument(
        "--thinking-budget-rewrite",
        type=int,
        default=THINKING_BUDGET_TOKENS["query_rewrite"],
        help="検索クエリ変換の thinking 概算トークン上限。0 以下で無制限。",
    )
    parser.add_argument(
        "--thinking-budget-answer",
        type=int,
        default=THINKING_BUDGET_TOKENS["answer_generation"],
        help="回答生成の thinking 概算トークン上限。0 以下で無制限。",
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

    thinking_budgets = {
        "image_search_decision": (
            None if args.thinking_budget_decision <= 0 else args.thinking_budget_decision
        ),
        "query_rewrite": (
            None if args.thinking_budget_rewrite <= 0 else args.thinking_budget_rewrite
        ),
        "answer_generation": (
            None if args.thinking_budget_answer <= 0 else args.thinking_budget_answer
        ),
    }

    index = None
    image_paths = None
    image_dates = None
    model = None

    def load_search_backend():
        nonlocal index, image_paths, image_dates, model

        if index is not None and image_paths is not None and model is not None:
            return index, image_paths, image_dates, model

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

        if IMAGE_DATES_PATH.exists():
            image_dates = load_image_dates(IMAGE_DATES_PATH)

        model = SentenceTransformer(MODEL_NAME)

        return index, image_paths, image_dates, model

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
        thinking_log = []
        result_dir = make_result_dir()

        with open(result_dir / "raw_query.txt", "w", encoding="utf-8") as f:
            f.write(raw_query + "\n")

        needs_image_search = decide_image_search_with_ollama(
            raw_query,
            model_name=args.ollama_model,
            base_url=args.ollama_url,
            timeout=args.ollama_timeout,
            thinking_log=thinking_log,
            thinking_callback=create_thinking_chunk_writer(result_dir, "image_search_decision"),
            thinking_budget_tokens=thinking_budgets["image_search_decision"],
        )

        print(f"raw query: {raw_query}")
        print(f"needs image search: {'Yes' if needs_image_search else 'No'}")

        if not needs_image_search:
            with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
                f.write(search_query + "\n")

            print(f"results: {result_dir}")
            print()
            print("LLM response:")
            answer_result = answer_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
                stream_callback=(lambda chunk: print(chunk, end="", flush=True)) if args.interactive else None,
                thinking_callback=create_thinking_chunk_writer(result_dir, "answer_generation"),
                thinking_budget_tokens=thinking_budgets["answer_generation"],
            )
            append_thinking_record(thinking_log, step="answer_generation", result=answer_result)
            llm_response = answer_result.content

            with open(result_dir / "llm_response.txt", "w", encoding="utf-8") as f:
                f.write(llm_response + "\n")
            write_ollama_thinking_files(result_dir, thinking_log)

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
                thinking_log=thinking_log,
                thinking_callback=create_thinking_chunk_writer(result_dir, "query_rewrite"),
                thinking_budget_tokens=thinking_budgets["query_rewrite"],
            )

        index, image_paths, image_dates, model = load_search_backend()

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

        # 後方互換のため、query.txt には実際に検索へ使ったクエリを保存する。
        with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
            f.write(search_query + "\n")

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

            date_field = format_date_field(src_path, image_dates)
            print(f"{rank:02d}  score={score:.4f}{date_field}  {src_path} -> {dst_path}")

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

            date_field = format_date_field(src_path, image_dates)
            print(f"{rank:02d}  score={score:.4f}{date_field}  {src_path} -> {dst_path}")

        if best_image is None:
            print()
            print("LLM response:")
            answer_result = answer_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
                stream_callback=(lambda chunk: print(chunk, end="", flush=True)) if args.interactive else None,
                thinking_callback=create_thinking_chunk_writer(result_dir, "answer_generation"),
                thinking_budget_tokens=thinking_budgets["answer_generation"],
            )
        else:
            _, _, best_path = best_image
            best_date = None if image_dates is None else image_dates.get(str(best_path))
            print()
            print("LLM response:")
            answer_result = answer_with_ollama(
                raw_query,
                model_name=args.ollama_model,
                base_url=args.ollama_url,
                timeout=args.ollama_timeout,
                image_path=best_path,
                image_date=best_date,
                stream_callback=(lambda chunk: print(chunk, end="", flush=True)) if args.interactive else None,
                thinking_callback=create_thinking_chunk_writer(result_dir, "answer_generation"),
                thinking_budget_tokens=thinking_budgets["answer_generation"],
            )

        append_thinking_record(thinking_log, step="answer_generation", result=answer_result)
        llm_response = answer_result.content

        with open(result_dir / "llm_response.txt", "w", encoding="utf-8") as f:
            f.write(llm_response + "\n")
        write_ollama_thinking_files(result_dir, thinking_log)

        if args.interactive:
            print()
        else:
            print(llm_response)

        if not args.interactive:
            break
        print()


if __name__ == "__main__":
    main()
