import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        @staticmethod
        def write(message: str) -> None:
            print(message)


OLLAMA_MODEL_NAME = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0
DATA_DIR = Path("data")
IMAGE_DIR = Path("images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CAPTION_PROMPT = (
    "入力画像は車内から撮影されたものです。"
    "車外の状況説明を4文程度の日本語で出力してください。"
    "改行や段落は必要ありません。"
)
HIRAGANA_RE = re.compile(r"[\u3041-\u3096\u309d-\u309f]")


def normalize_ollama_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        return OLLAMA_BASE_URL
    if "://" not in base_url:
        return f"http://{base_url}"
    return base_url


def load_image_paths(paths_json: Path, image_dir: Path) -> list[Path]:
    if paths_json.exists():
        with open(paths_json, "r", encoding="utf-8") as f:
            paths = json.load(f)

        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise RuntimeError(f"{paths_json} は画像パス文字列のリストである必要があります。")

        return [Path(path) for path in paths]

    if not image_dir.exists():
        raise FileNotFoundError(
            f"{paths_json} がなく、{image_dir} も存在しません。画像パスを読み込めません。"
        )

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_completed_paths(captions_jsonl: Path) -> set[str]:
    completed_paths = set()
    if not captions_jsonl.exists():
        return completed_paths

    with open(captions_jsonl, "rb") as f:
        line_number = 0
        while True:
            line_start = f.tell()
            raw_line = f.readline()
            if not raw_line:
                break

            line_number += 1
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            try:
                line = stripped_line.decode("utf-8")
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if f.read(1) == b"":
                    with open(captions_jsonl, "ab") as truncate_f:
                        truncate_f.truncate(line_start)
                    print(
                        f"{captions_jsonl}:{line_number} の不完全な末尾行を削除しました。"
                        " この画像は未処理として再生成します。"
                    )
                    return completed_paths

                raise RuntimeError(
                    f"{captions_jsonl}:{line_number} がJSONLとして読み込めません。"
                ) from exc

            image_path = record.get("image_path")
            caption = record.get("caption")
            if isinstance(image_path, str) and isinstance(caption, str) and caption:
                completed_paths.add(image_path)

    return completed_paths


def clean_caption(text: str) -> str:
    match = re.search(r"<think>\s*.*?\s*</think>", text, flags=re.DOTALL)
    while match:
        text = text[: match.start()] + text[match.end() :]
        match = re.search(r"<think>\s*.*?\s*</think>", text, flags=re.DOTALL)

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return re.sub(r"\s+", " ", text).strip(" \t\r\n\"'「」")


def contains_hiragana(text: str) -> bool:
    return HIRAGANA_RE.search(text) is not None


def caption_image_with_ollama(
    image_path: Path,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
) -> str:
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": CAPTION_PROMPT,
                "images": [image_base64],
            }
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
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Ollama でキャプション生成できませんでした: {image_path}。"
            f" Ollama が起動しているか、モデル {model_name!r} が利用可能か確認してください。"
        ) from exc

    data = json.loads(response_body)
    if data.get("error"):
        raise RuntimeError(f"Ollama のキャプション生成中にエラーが発生しました: {data['error']}")

    message = data.get("message", {})
    caption = clean_caption(message.get("content", ""))
    if not caption:
        raise RuntimeError(f"Ollama のキャプション生成結果が空でした: {image_path}")
    if not contains_hiragana(caption):
        raise RuntimeError(
            f"Ollama のキャプション生成結果にひらがなが含まれていませんでした: {image_path}"
        )

    return caption


def caption_with_retries(
    image_path: Path,
    *,
    model_name: str,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            return caption_image_with_ollama(
                image_path,
                model_name=model_name,
                base_url=base_url,
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep)

    raise RuntimeError(f"{image_path} のキャプション生成に失敗しました。") from last_error


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_jsonl_record(file_obj, record: dict) -> None:
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_obj.flush()
    os.fsync(file_obj.fileno())


def main():
    parser = argparse.ArgumentParser(
        description="Ollama の VLM で画像キャプションを事前生成します。"
    )
    parser.add_argument(
        "--paths-json",
        type=Path,
        default=DATA_DIR / "image_paths.json",
        help="画像パス一覧JSON。存在しない場合は --image-dir を走査します。デフォルト: data/image_paths.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_DIR,
        help="--paths-json が存在しない場合に走査する画像ディレクトリ。デフォルト: images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "image_captions.jsonl",
        help="キャプションJSONLの出力先。デフォルト: data/image_captions.jsonl",
    )
    parser.add_argument(
        "--errors-out",
        type=Path,
        default=DATA_DIR / "image_caption_errors.jsonl",
        help="--continue-on-error 時のエラーJSONL出力先。デフォルト: data/image_caption_errors.jsonl",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL_NAME,
        help=f"キャプション生成に使う Ollama モデル。デフォルト: {OLLAMA_MODEL_NAME}",
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
        "--retries",
        type=int,
        default=2,
        help="画像ごとのリトライ回数。デフォルト: 2",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="リトライ前に待つ秒数。デフォルト: 5.0",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="画像ごとのOllamaリクエスト間隔秒数。デフォルト: 0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理する画像数の上限。動作確認用。",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="画像パス一覧の何番目から処理するか。デフォルト: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存のキャプションJSONLを削除して最初から生成します。",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="画像ごとの最終失敗時に停止せず、エラーJSONLへ記録して次へ進みます。",
    )
    args = parser.parse_args()

    if args.ollama_timeout <= 0:
        parser.error("--ollama-timeout は 0 より大きい値を指定してください。")
    if args.retries < 0:
        parser.error("--retries は 0 以上を指定してください。")
    if args.retry_sleep < 0:
        parser.error("--retry-sleep は 0 以上を指定してください。")
    if args.request_interval < 0:
        parser.error("--request-interval は 0 以上を指定してください。")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit は 0 以上を指定してください。")
    if args.start_index < 0:
        parser.error("--start-index は 0 以上を指定してください。")

    image_paths = load_image_paths(args.paths_json, args.image_dir)
    if args.start_index:
        image_paths = image_paths[args.start_index :]
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError("処理対象の画像がありません。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.errors_out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.out.exists():
        args.out.unlink()

    completed_paths = set() if args.overwrite else load_completed_paths(args.out)
    pending_paths = [path for path in image_paths if str(path) not in completed_paths]

    print(f"Images loaded: {len(image_paths)}")
    print(f"Already captioned: {len(image_paths) - len(pending_paths)}")
    print(f"Pending: {len(pending_paths)}")
    print(f"Output: {args.out}")
    print(f"Ollama: {normalize_ollama_base_url(args.ollama_url)}  model={args.ollama_model}")

    if not pending_paths:
        return

    consecutive_errors = 0
    with ExitStack() as stack:
        out_f = stack.enter_context(open(args.out, "a", encoding="utf-8", buffering=1))
        error_f = None
        if args.continue_on_error:
            error_f = stack.enter_context(
                open(args.errors_out, "a", encoding="utf-8", buffering=1)
            )

        progress = tqdm(pending_paths, desc="Captioning images", unit="image")
        for image_path in progress:
            if not image_path.exists():
                error = FileNotFoundError(f"画像ファイルが存在しません: {image_path}")
            else:
                error = None

            try:
                if error is not None:
                    raise error

                caption = caption_with_retries(
                    image_path,
                    model_name=args.ollama_model,
                    base_url=args.ollama_url,
                    timeout=args.ollama_timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                write_jsonl_record(
                    out_f,
                    {
                        "schema_version": 1,
                        "image_path": str(image_path),
                        "caption": caption,
                        "prompt": CAPTION_PROMPT,
                        "model": args.ollama_model,
                        "created_at": utc_now_iso(),
                    },
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                if not args.continue_on_error:
                    raise

                if error_f is None:
                    raise

                write_jsonl_record(
                    error_f,
                    {
                        "schema_version": 1,
                        "image_path": str(image_path),
                        "error": str(exc),
                        "model": args.ollama_model,
                        "created_at": utc_now_iso(),
                    },
                )
                progress.write(f"ERROR: {image_path}: {exc}")

            if args.request_interval > 0:
                time.sleep(args.request_interval)

    print(f"Saved: {args.out}")
    if args.continue_on_error and consecutive_errors:
        print(f"直近の連続エラー数: {consecutive_errors}")


if __name__ == "__main__":
    main()
