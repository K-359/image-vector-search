"""
Qwen3-VL-Reranker-8B のファインチューニング/評価で共有するユーティリティ。

学習と評価と本番検索で、モデルのロード方法・命令文・スコア計算を完全に一致させるため、
Reranker に関する処理はこのモジュールへ集約する。
本番検索の命令文は scripts/search.py から直接読み込み、コピーによる乖離を防ぐ。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from search import (  # noqa: E402
    DASHCAM_RERANKER_PROMPT,
    RERANKER_MODEL_NAME,
    build_reranker_query,
)
from reranker_adapter import attach_adapter, resolve_adapter_path  # noqa: E402

__all__ = [
    "DASHCAM_RERANKER_PROMPT",
    "RERANKER_MODEL_NAME",
    "build_reranker_query",
    "PairRecord",
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "resolve_image_path",
    "load_reranker",
    "attach_adapter",
    "resolve_adapter_path",
    "score_pairs",
    "build_predict_pairs",
    "validate_pair_captions",
]

# ---------------------------------------------------------------------------
# JSONL ユーティリティ
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL を読み込む。末尾に壊れた行があれば無視する。"""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                # 生成途中で停止した場合の末尾行だけは切り捨てる。
                break
            raise RuntimeError(f"{path} の {index + 1} 行目が JSON として不正です。")

    return records


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for record in read_jsonl(path):
        yield record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# 学習/評価ペア
# ---------------------------------------------------------------------------


@dataclass
class PairRecord:
    """1件の (クエリ, 候補画像) ペア。"""

    pair_id: str
    query_id: str
    query_text: str
    image_id: str
    image_path: str
    label: int
    negative_type: str
    split: str
    caption: str | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "PairRecord":
        return cls(
            pair_id=record["pair_id"],
            query_id=record["query_id"],
            query_text=record["query_text"],
            image_id=record["image_id"],
            image_path=record["image_path"],
            label=int(record["label"]),
            negative_type=record.get("negative_type", "unknown"),
            split=record.get("split", "unknown"),
            caption=record.get("caption"),
            raw=record,
        )


def load_pairs(path: Path) -> list[PairRecord]:
    return [PairRecord.from_dict(record) for record in read_jsonl(path)]


def validate_pair_captions(
    pairs: list[PairRecord],
    *,
    allow_partial: bool = False,
    context: str = "データセット",
) -> tuple[int, int]:
    """caption モードで候補ごとの入力条件が意図せず混在しないことを検証する。"""

    total = len(pairs)
    covered = sum(1 for pair in pairs if pair.caption and pair.caption.strip())
    if covered == 0:
        raise RuntimeError(
            f"{context} の {total} ペアに利用可能な caption がありません。"
            " --use-caption を外すか、caption を含むペアを作成してください。"
        )
    if covered != total and not allow_partial:
        raise RuntimeError(
            f"{context} の caption coverage は {covered}/{total} です。"
            " caption モードでは全ペアの caption が必要です。"
            " 欠損を承知で混在させる場合だけ --allow-partial-captions を指定してください。"
        )
    return covered, total


def relative_to_project(path: Path, project_root: Path) -> str:
    """レポートへ書くための相対パス。プロジェクト外なら絶対パスのまま返す。"""

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def resolve_image_path(image_path: str, project_root: Path) -> Path:
    """データセット内の相対パスを実ファイルへ解決する。"""

    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


# ---------------------------------------------------------------------------
# Reranker のロードとスコアリング
# ---------------------------------------------------------------------------


def load_reranker(
    *,
    model_name: str = RERANKER_MODEL_NAME,
    device: str = "cuda:0",
    quantization: str = "4bit",
    max_pixels: int | None = None,
    torch_dtype: str = "bfloat16",
    attn_implementation: str | None = None,
):
    """
    本番検索 (scripts/search.py) と同じ構成で CrossEncoder を組み立てる。

    Qwen3-VL-Reranker のリポジトリは modules.json が参照する 1_CausalScoreHead の
    設定ファイルを含まないため、Transformer + LogitScore を明示的に組み立てる。
    """

    import torch
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.modules import LogitScore, Transformer
    from transformers import BitsAndBytesConfig

    dtype = getattr(torch, torch_dtype)

    model_kwargs: dict[str, Any] = {"device_map": {"": device}, "dtype": dtype}

    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != "none":
        raise ValueError(f"未知の quantization です: {quantization}")

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    processor_kwargs: dict[str, Any] = {}
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    transformer = Transformer.load(
        model_name,
        model_kwargs=model_kwargs,
        processor_kwargs=processor_kwargs or None,
    )

    true_token_id = transformer.tokenizer.convert_tokens_to_ids("yes")
    false_token_id = transformer.tokenizer.convert_tokens_to_ids("no")
    if true_token_id is None or false_token_id is None:
        raise RuntimeError("再ランカーのトークナイザにスコア計算用の yes/no トークンがありません。")

    class DeviceMappedCrossEncoder(CrossEncoder):
        def to(self, *args, **kwargs):
            # device_map による配置を CrossEncoder 側から変更しない。
            return self

    return DeviceMappedCrossEncoder(
        modules=[
            transformer,
            LogitScore(true_token_id=true_token_id, false_token_id=false_token_id),
        ],
        prompts={"query": "Retrieve text relevant to the user's query."},
        default_prompt_name="query",
    )


def build_predict_pairs(
    pairs: list[PairRecord],
    project_root: Path,
    *,
    use_caption: bool = False,
    normalize_query: bool = True,
) -> list[tuple[str, dict[str, str]]]:
    """
    PairRecord を CrossEncoder.predict / preprocess が受け取る形式へ変換する。

    本番検索と同じく、ドキュメントは {"image": 絶対パス} (キャプション併用時は "text" も) とする。
    """

    predict_pairs: list[tuple[str, dict[str, str]]] = []
    for pair in pairs:
        query = build_reranker_query(pair.query_text) if normalize_query else pair.query_text
        document: dict[str, str] = {"image": str(resolve_image_path(pair.image_path, project_root))}
        if use_caption and pair.caption:
            document["text"] = pair.caption
        predict_pairs.append((query, document))
    return predict_pairs


def score_pairs(
    model,
    pairs: list[PairRecord],
    project_root: Path,
    *,
    batch_size: int = 1,
    use_caption: bool = False,
    normalize_query: bool = True,
    show_progress_bar: bool = True,
) -> list[float]:
    """本番検索と同じ経路で relevance スコア (logit(yes) - logit(no)) を計算する。"""

    predict_pairs = build_predict_pairs(
        pairs,
        project_root,
        use_caption=use_caption,
        normalize_query=normalize_query,
    )
    scores = model.predict(
        predict_pairs,
        batch_size=batch_size,
        prompt=DASHCAM_RERANKER_PROMPT,
        show_progress_bar=show_progress_bar,
    )
    return [float(score) for score in scores]
