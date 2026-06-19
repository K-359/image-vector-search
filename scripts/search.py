import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import readline  # noqa: F401
except ImportError:
    pass


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
OLLAMA_MODEL_NAME = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0
THINKING_BUDGET_TOKENS = {
    "image_search_decision": 1500,
    "query_rewrite": 500,
    "answer_generation": 500,
}
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
IMAGE_DATES_PATH = DATA_DIR / "image_dates.json"
CAPTIONS_PATH = DATA_DIR / "image_captions.jsonl"
CAPTION_INDEX_PATH = DATA_DIR / "caption_embeddings.faiss"
CAPTION_INDEX_META_PATH = DATA_DIR / "caption_embeddings_meta.json"
CAPTION_BATCH_SIZE = 5
CAPTION_RRF_K = 60
RERANK_CANDIDATES = 50
RERANKER_BATCH_SIZE = 1
DASHCAM_RETRIEVAL_PROMPT = (
    "Retrieve dashcam images that visually show the road scene, traffic participant, "
    "dangerous behavior, collision, or near-miss event described in the user's query."
)
DASHCAM_RERANKER_PROMPT = (
    "Retrieve dashcam images that exactly match all visible facts in the user query. "
    "Pay special attention to traffic direction, lane position, participant orientation, "
    "and relative motion. For wrong-way traffic, require visible evidence that the "
    "participant is moving against the expected traffic flow; do not treat same-direction "
    "traffic as relevant."
)
WRONG_WAY_BICYCLE_QUERY = (
    "A wrong-way cyclist facing and riding toward the dashcam against the direction "
    "of motor traffic."
)
SIMPLE_WRONG_WAY_BICYCLE_PATTERNS = (
    re.compile(
        r"^(?:自転車|サイクリスト|チャリ)(?:が|は|の)?"
        r"(?:(?:道路|車道)(?:を|で)?)?逆走(?:している|してる|する|中)?[。.!！]?$"
    ),
    re.compile(
        r"^逆走(?:している|してる|する|中)?(?:の)?"
        r"(?:自転車|サイクリスト|チャリ)[。.!！]?$"
    ),
)


@dataclass
class OllamaChatResult:
    content: str
    thinking: str
    thinking_budget_exceeded: bool = False
    fallback_used: bool = False


@dataclass
class CaptionRecord:
    image_path: str
    caption: str


@dataclass
class SearchResult:
    score: float
    image_id: int
    retrieval_score: float | None = None
    reranker_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    caption: str | None = None


class BM25Index:
    def __init__(self, documents: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.document_count = len(documents)
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(document) for document in documents]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0.0
        )
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: dict[str, int] = defaultdict(int)

        for document_id, document in enumerate(documents):
            term_counts = Counter(document)
            for term, frequency in term_counts.items():
                self.postings[term].append((document_id, frequency))
                document_frequency[term] += 1

        self.idf = {
            term: math.log(
                1.0
                + (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def score(self, query_tokens: list[str]):
        import numpy as np

        scores = np.zeros(self.document_count, dtype="float32")
        if not query_tokens or not self.document_count or self.avg_doc_length <= 0:
            return scores

        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue

            for document_id, frequency in self.postings[term]:
                doc_length = self.doc_lengths[document_id]
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / self.avg_doc_length
                )
                scores[document_id] += idf * frequency * (self.k1 + 1.0) / denominator

        return scores


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


def build_reranker_query(search_query: str) -> str:
    """
    Qwen3-VL-Reranker は短い日本語の「自転車が逆走している」では、
    自転車の有無を進行方向より強く評価することがある。
    単純な逆走自転車クエリだけ、視覚的に判定できる英語表現へ正規化する。
    """
    normalized_query = re.sub(r"\s+", "", search_query)

    if any(pattern.fullmatch(normalized_query) for pattern in SIMPLE_WRONG_WAY_BICYCLE_PATTERNS):
        return WRONG_WAY_BICYCLE_QUERY

    return search_query


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
                    response.close()

                if not thinking_budget_exceeded:
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

    if stream_response and thinking_budget_exceeded:
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
    image_caption: str | None = None,
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
        if image_caption is not None:
            content = f"検索された画像のキャプション: {image_caption}\n\n{content}"

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
    results: list[SearchResult],
    image_paths: list[str],
) -> tuple[SearchResult, Path] | None:
    for result in results:
        if result.image_id < 0:
            continue

        src_path = Path(image_paths[result.image_id])
        if src_path.exists():
            return result, src_path

    return None


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []

    for chunk in re.findall(
        r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+",
        normalized,
    ):
        if re.fullmatch(r"[a-z0-9]+", chunk):
            tokens.append(chunk)
            continue

        tokens.extend(chunk)
        tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
        tokens.extend(chunk[index : index + 3] for index in range(len(chunk) - 2))

    return tokens


def load_caption_records(path: Path) -> list[CaptionRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} がありません。先に scripts/generate_captions.py を実行してください。"
        )

    records: list[CaptionRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number} がJSONLとして読み込めません。") from exc

            image_path = payload.get("image_path")
            caption = payload.get("caption")
            if not isinstance(image_path, str) or not image_path:
                raise RuntimeError(f"{path}:{line_number} の image_path が不正です。")
            if not isinstance(caption, str) or not caption.strip():
                raise RuntimeError(f"{path}:{line_number} の caption が不正です。")

            records.append(CaptionRecord(image_path=image_path, caption=caption.strip()))

    if not records:
        raise RuntimeError(f"{path} に有効なキャプションがありません。")

    return records


def caption_index_metadata(captions_path: Path, records: list[CaptionRecord]) -> dict:
    stat = captions_path.stat()
    return {
        "schema_version": 2,
        "captions_path": str(captions_path),
        "captions_mtime_ns": stat.st_mtime_ns,
        "captions_size": stat.st_size,
        "caption_count": len(records),
        "model_name": MODEL_NAME,
        "normalize_embeddings": True,
        "records_sha256": caption_records_sha256(records),
    }


def load_caption_index_metadata(meta_path: Path) -> dict | None:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return metadata if isinstance(metadata, dict) else None


def caption_records_sha256(records: list[CaptionRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.caption.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def caption_index_is_current(actual_metadata: dict | None, expected_metadata: dict) -> bool:
    if actual_metadata is None:
        return False

    required_keys = (
        "captions_path",
        "captions_mtime_ns",
        "captions_size",
        "caption_count",
        "model_name",
        "normalize_embeddings",
    )
    if not all(actual_metadata.get(key) == expected_metadata[key] for key in required_keys):
        return False

    actual_schema_version = actual_metadata.get("schema_version")
    if actual_schema_version == 1:
        return True

    return (
        actual_schema_version == expected_metadata["schema_version"]
        and actual_metadata.get("records_sha256") == expected_metadata["records_sha256"]
    )


def caption_index_can_append(
    index,
    actual_metadata: dict | None,
    *,
    captions_path: Path,
    records: list[CaptionRecord],
) -> int | None:
    if actual_metadata is None:
        return None

    if actual_metadata.get("captions_path") != str(captions_path):
        return None
    if actual_metadata.get("model_name") != MODEL_NAME:
        return None
    if actual_metadata.get("normalize_embeddings") is not True:
        return None

    old_count = actual_metadata.get("caption_count")
    old_size = actual_metadata.get("captions_size")
    if not isinstance(old_count, int) or not isinstance(old_size, int):
        return None
    if old_count <= 0 or old_count >= len(records):
        return None
    if old_size > captions_path.stat().st_size:
        return None
    if index.ntotal != old_count:
        return None

    records_sha256 = actual_metadata.get("records_sha256")
    if records_sha256 is not None and records_sha256 != caption_records_sha256(records[:old_count]):
        return None

    return old_count


def add_caption_embeddings_to_index(
    index,
    records: list[CaptionRecord],
    *,
    start_index: int,
    model,
    batch_size: int,
):
    import faiss
    import numpy as np

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    starts = range(start_index, len(records), batch_size)
    iterator = (
        tqdm(starts, desc="Embedding captions")
        if tqdm is not None
        else starts
    )

    for start in iterator:
        batch = records[start : start + batch_size]
        embeddings = model.encode(
            [record.caption for record in batch],
            batch_size=batch_size,
            prompt="Represent this image caption for retrieval.",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if index is None:
            dim = embeddings.shape[1]
            base_index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(base_index)

        ids = np.arange(start, start + len(batch)).astype("int64")
        index.add_with_ids(embeddings, ids)

    return index


def build_caption_vector_index(
    records: list[CaptionRecord],
    *,
    model,
    index_path: Path,
    meta_path: Path,
    metadata: dict,
    batch_size: int,
):
    import faiss

    index = add_caption_embeddings_to_index(
        None,
        records,
        start_index=0,
        model=model,
        batch_size=batch_size,
    )
    if index is None:
        raise RuntimeError("キャプション埋め込みインデックスを作成できませんでした。")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return index


def load_or_build_caption_vector_index(
    records: list[CaptionRecord],
    *,
    captions_path: Path,
    model,
    index_path: Path,
    meta_path: Path,
    batch_size: int,
):
    import faiss

    metadata = caption_index_metadata(captions_path, records)
    actual_metadata = load_caption_index_metadata(meta_path) if meta_path.exists() else None

    if index_path.exists() and caption_index_is_current(actual_metadata, metadata):
        index = faiss.read_index(str(index_path))
        if index.ntotal == len(records):
            return index

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        append_start = caption_index_can_append(
            index,
            actual_metadata,
            captions_path=captions_path,
            records=records,
        )
        if append_start is not None:
            print(
                f"caption vector index に追加分だけを追記します: "
                f"{len(records) - append_start} captions"
            )
            index = add_caption_embeddings_to_index(
                index,
                records,
                start_index=append_start,
                model=model,
                batch_size=batch_size,
            )
            faiss.write_index(index, str(index_path))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                f.write("\n")
            return index

    print(f"caption vector index を作成します: {index_path}")
    return build_caption_vector_index(
        records,
        model=model,
        index_path=index_path,
        meta_path=meta_path,
        metadata=metadata,
        batch_size=batch_size,
    )


def search_image_vectors(index, model, search_query: str) -> list[SearchResult]:
    query_embedding = model.encode(
        [search_query],
        prompt=DASHCAM_RETRIEVAL_PROMPT,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(query_embedding, index.ntotal)
    return [
        SearchResult(
            score=float(score),
            image_id=int(image_id),
            retrieval_score=float(score),
            vector_score=float(score),
        )
        for score, image_id in zip(scores[0], ids[0])
        if image_id >= 0
    ]


def search_caption_hybrid(
    index,
    model,
    bm25_index: BM25Index,
    records: list[CaptionRecord],
    search_query: str,
    *,
    rrf_k: int,
) -> list[SearchResult]:
    import numpy as np

    query_embedding = model.encode(
        [search_query],
        batch_size=1,
        prompt="Retrieve image captions relevant to the user's query.",
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(query_embedding, index.ntotal)
    vector_scores = np.zeros(len(records), dtype="float32")
    for score, image_id in zip(scores[0], ids[0]):
        if 0 <= image_id < len(records):
            vector_scores[image_id] = score

    bm25_scores = bm25_index.score(tokenize_for_bm25(search_query))
    rrf_scores = np.zeros(len(records), dtype="float32")

    for rank, image_id in enumerate(ids[0], start=1):
        if 0 <= image_id < len(records):
            rrf_scores[image_id] += 1.0 / (rrf_k + rank)

    bm25_ranked_ids = np.argsort(-bm25_scores)
    for rank, image_id in enumerate(bm25_ranked_ids, start=1):
        if bm25_scores[image_id] <= 0:
            break
        rrf_scores[image_id] += 1.0 / (rrf_k + rank)

    ordered_ids = np.argsort(-rrf_scores)

    return [
        SearchResult(
            score=float(rrf_scores[image_id]),
            image_id=int(image_id),
            retrieval_score=float(rrf_scores[image_id]),
            vector_score=float(vector_scores[image_id]),
            bm25_score=float(bm25_scores[image_id]),
            caption=records[image_id].caption,
        )
        for image_id in ordered_ids
    ]


def rerank_search_results(
    ranked_results: list[SearchResult],
    image_paths: list[str],
    search_query: str,
    *,
    reranker,
    candidate_count: int,
    batch_size: int,
) -> list[SearchResult]:
    if candidate_count <= 0 or not ranked_results:
        return ranked_results

    candidates = ranked_results[:candidate_count]
    pairs = []
    pair_results = []
    unrereanked_candidates = []

    for result in candidates:
        if not 0 <= result.image_id < len(image_paths):
            unrereanked_candidates.append(result)
            continue

        image_path = Path(image_paths[result.image_id])
        if not image_path.exists():
            unrereanked_candidates.append(result)
            continue

        document = {"image": str(image_path.resolve())}
        if result.caption is not None:
            document["text"] = result.caption

        pairs.append((search_query, document))
        pair_results.append(result)

    if not pairs:
        return ranked_results

    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        prompt=DASHCAM_RERANKER_PROMPT,
        show_progress_bar=len(pairs) > batch_size,
    )
    for result, score in zip(pair_results, scores):
        result.reranker_score = float(score)
        result.score = float(score)

    reranked_candidates = sorted(
        pair_results,
        key=lambda result: result.reranker_score,
        reverse=True,
    )
    return [
        *reranked_candidates,
        *unrereanked_candidates,
        *ranked_results[candidate_count:],
    ]


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


def format_score_fields(result: SearchResult) -> str:
    if result.reranker_score is not None:
        fields = [f"reranker={result.reranker_score:.4f}"]
        if result.retrieval_score is not None:
            fields.append(f"retrieval={result.retrieval_score:.4f}")
    else:
        fields = [f"score={result.score:.4f}"]
    if result.vector_score is not None:
        fields.append(f"vector={result.vector_score:.4f}")
    if result.bm25_score is not None:
        fields.append(f"bm25={result.bm25_score:.4f}")
    return "  ".join(fields)


def shorten_text(text: str, max_length: int = 120) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1] + "…"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", type=str, help="検索したい内容。例: 夕焼けの海辺を走る犬")
    parser.add_argument(
        "--mode",
        choices=("image", "caption"),
        default="image",
        help=(
            "検索方式。image は従来の画像ベクトル検索、caption は"
            "キャプションのベクトル検索とBM25検索のハイブリッド。デフォルト: image"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10, help="下位検索結果として出力する件数。")
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help=(
            "Qwen3-VL-Reranker-2B で再ランキングする初段検索の上位件数。"
            f"0 で無効。デフォルト: {RERANK_CANDIDATES}"
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=RERANKER_MODEL_NAME,
        help=f"再ランキングに使うモデル。デフォルト: {RERANKER_MODEL_NAME}",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=RERANKER_BATCH_SIZE,
        help=f"再ランキング時のバッチサイズ。デフォルト: {RERANKER_BATCH_SIZE}",
    )
    parser.add_argument(
        "--captions-jsonl",
        type=Path,
        default=CAPTIONS_PATH,
        help=f"caption モードで使うキャプションJSONL。デフォルト: {CAPTIONS_PATH}",
    )
    parser.add_argument(
        "--caption-rrf-k",
        type=int,
        default=CAPTION_RRF_K,
        help=f"caption モードのRRFで使う順位定数。デフォルト: {CAPTION_RRF_K}",
    )
    parser.add_argument(
        "--caption-vector-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=CAPTION_BATCH_SIZE,
        help=f"caption モードで初回インデックス作成時に使う埋め込みバッチサイズ。デフォルト: {CAPTION_BATCH_SIZE}",
    )
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
        "--skip-image-search-decision",
        action="store_true",
        help="Ollama による画像検索要否判定をスキップし、画像検索が必要なものとして処理する。",
    )
    parser.add_argument(
        "--skip-answer-generation",
        action="store_true",
        help="Ollama による最終回答生成をスキップする。",
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
    if args.rerank_candidates < 0:
        parser.error("--rerank-candidates は 0 以上を指定してください。")
    if args.reranker_batch_size <= 0:
        parser.error("--reranker-batch-size は 1 以上を指定してください。")
    if args.caption_rrf_k <= 0:
        parser.error("--caption-rrf-k は 1 以上を指定してください。")
    if args.caption_batch_size <= 0:
        parser.error("--caption-batch-size は 1 以上を指定してください。")
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

    image_index = None
    caption_index = None
    image_paths = None
    image_dates = None
    caption_records = None
    caption_bm25_index = None
    model = None
    reranker = None

    def load_embedding_model():
        nonlocal model

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME)

        return model

    def load_reranker():
        nonlocal reranker

        if reranker is None:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(args.reranker_model)

        return reranker

    def load_image_dates_once():
        nonlocal image_dates

        if image_dates is None and IMAGE_DATES_PATH.exists():
            image_dates = load_image_dates(IMAGE_DATES_PATH)

        return image_dates

    def load_image_search_backend():
        nonlocal image_index, image_paths

        index_path = DATA_DIR / "images.faiss"
        paths_path = DATA_DIR / "image_paths.json"

        if image_index is not None and image_paths is not None:
            return image_index, image_paths, load_image_dates_once(), load_embedding_model()

        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} がありません。先に scripts/build_index.py を実行してください。")

        if not paths_path.exists():
            raise FileNotFoundError(f"{paths_path} がありません。先に scripts/build_index.py を実行してください。")

        import faiss
        image_index = faiss.read_index(str(index_path))

        with open(paths_path, "r", encoding="utf-8") as f:
            image_paths = json.load(f)

        return image_index, image_paths, load_image_dates_once(), load_embedding_model()

    def load_caption_search_backend():
        nonlocal caption_index, caption_records, caption_bm25_index

        if caption_index is not None and caption_records is not None and caption_bm25_index is not None:
            caption_paths = [record.image_path for record in caption_records]
            return (
                caption_index,
                caption_records,
                caption_paths,
                load_image_dates_once(),
                load_embedding_model(),
                caption_bm25_index,
            )

        loaded_records = load_caption_records(args.captions_jsonl)
        embedding_model = load_embedding_model()
        caption_index = load_or_build_caption_vector_index(
            loaded_records,
            captions_path=args.captions_jsonl,
            model=embedding_model,
            index_path=CAPTION_INDEX_PATH,
            meta_path=CAPTION_INDEX_META_PATH,
            batch_size=args.caption_batch_size,
        )
        caption_records = loaded_records
        caption_bm25_index = BM25Index(
            [tokenize_for_bm25(record.caption) for record in caption_records]
        )
        caption_paths = [record.image_path for record in caption_records]

        return (
            caption_index,
            caption_records,
            caption_paths,
            load_image_dates_once(),
            embedding_model,
            caption_bm25_index,
        )

    if args.interactive:
        if args.mode == "image":
            load_image_search_backend()
        else:
            load_caption_search_backend()
        if args.rerank_candidates > 0:
            load_reranker()
        print("検索文を入力してください。終了するには空行または Ctrl-D を入力してください。")
    elif args.mode == "caption":
        # 初回のキャプション埋め込み生成はメモリを多く使うため、
        # Reranker と Ollama の判定/回答用モデルをロードする前に済ませる。
        load_caption_search_backend()
        if args.rerank_candidates > 0:
            load_reranker()

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

        if args.skip_image_search_decision:
            needs_image_search = True
        else:
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
            if args.skip_answer_generation:
                write_ollama_thinking_files(result_dir, thinking_log)
                print()
                print("LLM response: skipped")

                if not args.interactive:
                    break
                print()
                continue

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

        caption_records_for_search = None
        if args.mode == "image":
            index, image_paths, image_dates, model = load_image_search_backend()
            ranked_results = search_image_vectors(index, model, search_query)
        else:
            (
                index,
                caption_records_for_search,
                image_paths,
                image_dates,
                model,
                bm25_index,
            ) = load_caption_search_backend()
            ranked_results = search_caption_hybrid(
                index,
                model,
                bm25_index,
                caption_records_for_search,
                search_query,
                rrf_k=args.caption_rrf_k,
            )

        if args.rerank_candidates > 0:
            reranker_query = build_reranker_query(search_query)
            ranked_results = rerank_search_results(
                ranked_results,
                image_paths,
                reranker_query,
                reranker=load_reranker(),
                candidate_count=args.rerank_candidates,
                batch_size=args.reranker_batch_size,
            )
        else:
            reranker_query = None

        top_results = ranked_results[: args.top_k]
        bottom_results = [] if args.bottom_k == 0 else ranked_results[-args.bottom_k :]
        best_image = find_best_existing_image(ranked_results, image_paths)

        # 後方互換のため、query.txt には実際に検索へ使ったクエリを保存する。
        with open(result_dir / "query.txt", "w", encoding="utf-8") as f:
            f.write(search_query + "\n")
        if reranker_query is not None:
            with open(result_dir / "reranker_query.txt", "w", encoding="utf-8") as f:
                f.write(reranker_query + "\n")

        print(f"search mode: {args.mode}")
        print(f"search query: {search_query}")
        if reranker_query is not None and reranker_query != search_query:
            print(f"reranker query: {reranker_query}")
        print(f"results: {result_dir}")
        print()

        print("top results:")
        for rank, result in enumerate(top_results, start=1):
            if result.image_id < 0:
                continue

            src_path = Path(image_paths[result.image_id])

            if not src_path.exists():
                print(f"{rank:02d}  {format_score_fields(result)}  MISSING: {src_path}")
                continue

            dst_name = f"{rank:02d}_{safe_score(result.score)}_{src_path.name}"
            dst_path = result_dir / dst_name

            shutil.copy2(src_path, dst_path)

            date_field = format_date_field(src_path, image_dates)
            print(f"{rank:02d}  {format_score_fields(result)}{date_field}  {src_path} -> {dst_path}")
            if result.caption is not None:
                print(f"    caption: {shorten_text(result.caption)}")

        print()
        print("bottom results:")
        for rank, result in enumerate(reversed(bottom_results), start=1):
            if result.image_id < 0:
                continue

            src_path = Path(image_paths[result.image_id])

            if not src_path.exists():
                print(f"{rank:02d}  {format_score_fields(result)}  MISSING: {src_path}")
                continue

            dst_name = f"bottom_{rank:02d}_{safe_score(result.score)}_{src_path.name}"
            dst_path = result_dir / dst_name

            shutil.copy2(src_path, dst_path)

            date_field = format_date_field(src_path, image_dates)
            print(f"{rank:02d}  {format_score_fields(result)}{date_field}  {src_path} -> {dst_path}")
            if result.caption is not None:
                print(f"    caption: {shorten_text(result.caption)}")

        if args.skip_answer_generation:
            write_ollama_thinking_files(result_dir, thinking_log)
            print()
            print("LLM response: skipped")

            if not args.interactive:
                break
            print()
            continue

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
            best_result, best_path = best_image
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
                image_caption=best_result.caption,
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
