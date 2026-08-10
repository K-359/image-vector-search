"""
Qwen3-VL-Reranker-8B のドライブレコーダ特化ファインチューニング用データセットを作る。

処理は2段階に分かれる。

1. cards:  BDD100k からサンプリングした画像 1枚ごとに、Ollama の教師モデル
           (既定 hf.co/unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q3_K_XL) で
           「シーンカード」を生成する。シーンカードは、画像に写っている事実を
           制約付きの語彙で構造化したものと、その画像から答えられる検索クエリを含む。
2. pairs:  シーンカードから (クエリ, 候補画像) ペアを機械的に構成する。
           クエリの制約を満たす画像が正例、満たさない画像が負例になる。
           制約の一部だけを満たす画像を hard negative として優先的に採用する。

ラベルは教師モデルの自由記述ではなく、構造化された事実集合に対する決定的な判定で付く。
そのため同じシーンカードからは常に同じラベルが再現され、学習前後の比較が安定する。

train / val / test は「画像単位」で分割する。クエリはその生成元画像の分割に属し、
候補画像も同じ分割のプールからのみ選ぶ。これにより画像もクエリも分割をまたがない。

使い方:

    python scripts/build_reranker_dataset.py --num-images 1000

    # シーンカードだけ作る (再開可能。既にカードがある画像はスキップ)
    python scripts/build_reranker_dataset.py --stage cards --num-images 1000

    # 既存のシーンカードからペアだけ作り直す (Ollama を呼ばない)
    python scripts/build_reranker_dataset.py --stage pairs
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reranker_common import (  # noqa: E402
    DASHCAM_RERANKER_PROMPT,
    append_jsonl,
    read_jsonl,
    relative_to_project,
    write_json,
    write_jsonl,
)

PROJECT_ROOT = SCRIPTS_DIR.parent

SCHEMA_VERSION = 1
PROMPT_VERSION = "dashcam-scene-card-v1"

DEFAULT_IMAGE_DIR = "images_100k"
DEFAULT_DATASET_DIR = "datasets/dashcam_reranker_ft_v1"
DEFAULT_TEACHER_MODEL = "hf.co/unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q3_K_XL"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 900.0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

UNSPECIFIED = "unspecified"
UNKNOWN = "unknown"

# --- 事実の語彙 -------------------------------------------------------------
# 教師モデルの出力をこの語彙へ強制することで、画像間で事実集合を比較できるようにする。

ROAD_TYPES = [
    "highway",
    "city_street",
    "residential_street",
    "rural_road",
    "parking_area",
    "tunnel",
    "bridge",
]
TIME_OF_DAY = ["day", "night", "dawn_dusk"]
WEATHER = ["clear", "partly_cloudy", "overcast", "rain", "snow", "fog"]
SCENE_EXTRA = [
    "intersection",
    "crosswalk_marking",
    "roadwork",
    "traffic_jam",
    "curve",
    "slope",
    "railway_crossing",
    "one_way_sign",
    "prohibition_sign",
    "lane_arrow",
    "traffic_light_red",
    "traffic_light_green",
    "traffic_light_yellow",
    "wet_road",
    "snow_on_road",
    "flooding",
    "tunnel_entrance",
    "toll_gate",
]
PARTICIPANT_TYPES = [
    "car",
    "bus",
    "truck",
    "van",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "emergency_vehicle",
    "train",
    "animal",
]
COLORS = [
    "white",
    "black",
    "gray",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "brown",
]
POSITIONS = [
    "foreground_left",
    "foreground_center",
    "foreground_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "background_left",
    "background_center",
    "background_right",
]
LANE_RELATIONS = [
    "same_lane",
    "left_lane",
    "right_lane",
    "oncoming_lane",
    "crosswalk",
    "sidewalk",
    "roadside_left",
    "roadside_right",
    "parking_area",
    "off_road",
]
ORIENTATIONS = [
    "away_from_camera",
    "toward_camera",
    "facing_left",
    "facing_right",
    "oblique",
]
MOTIONS = [
    "moving_away",
    "moving_toward_camera",
    "crossing_road",
    "turning",
    "stopped",
    "parked",
]

PARTICIPANT_ATTRIBUTES = ("type", "color", "position", "lane_relation", "orientation", "motion")

DIFFICULTIES = ["simple", "compositional"]


def _enum(values: Iterable[str], *, extra: Iterable[str] = ()) -> dict[str, Any]:
    return {"type": "string", "enum": [*values, *extra]}


def build_scene_card_schema(max_participants: int, max_queries: int) -> dict[str, Any]:
    """Ollama の structured outputs (format) へ渡す JSON Schema。"""

    participant_properties = {
        "type": _enum(PARTICIPANT_TYPES),
        "color": _enum(COLORS, extra=[UNKNOWN]),
        "position": _enum(POSITIONS),
        "lane_relation": _enum(LANE_RELATIONS),
        "orientation": _enum(ORIENTATIONS, extra=[UNKNOWN]),
        "motion": _enum(MOTIONS, extra=[UNKNOWN]),
    }
    required_participant_properties = {
        "type": _enum(PARTICIPANT_TYPES),
        "color": _enum(COLORS, extra=[UNSPECIFIED]),
        "position": _enum(POSITIONS, extra=[UNSPECIFIED]),
        "lane_relation": _enum(LANE_RELATIONS, extra=[UNSPECIFIED]),
        "orientation": _enum(ORIENTATIONS, extra=[UNSPECIFIED]),
        "motion": _enum(MOTIONS, extra=[UNSPECIFIED]),
    }

    return {
        "type": "object",
        "properties": {
            "caption_ja": {"type": "string"},
            "scene": {
                "type": "object",
                "properties": {
                    "road_type": {"type": "array", "items": _enum(ROAD_TYPES), "maxItems": 2},
                    "time_of_day": _enum(TIME_OF_DAY),
                    "weather": _enum(WEATHER),
                    "extra": {"type": "array", "items": _enum(SCENE_EXTRA), "maxItems": 6},
                },
                "required": ["road_type", "time_of_day", "weather", "extra"],
            },
            "participants": {
                "type": "array",
                "maxItems": max_participants,
                "items": {
                    "type": "object",
                    "properties": participant_properties,
                    "required": list(PARTICIPANT_ATTRIBUTES),
                },
            },
            "queries_ja": {
                "type": "array",
                "maxItems": max_queries,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "difficulty": _enum(DIFFICULTIES),
                        "required_scene": {
                            "type": "object",
                            "properties": {
                                "road_type": {
                                    "type": "array",
                                    "items": _enum(ROAD_TYPES),
                                    "maxItems": 2,
                                },
                                "time_of_day": _enum(TIME_OF_DAY, extra=[UNSPECIFIED]),
                                "weather": _enum(WEATHER, extra=[UNSPECIFIED]),
                                "extra": {
                                    "type": "array",
                                    "items": _enum(SCENE_EXTRA),
                                    "maxItems": 3,
                                },
                            },
                            "required": ["road_type", "time_of_day", "weather", "extra"],
                        },
                        "required_participants": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": required_participant_properties,
                                "required": list(PARTICIPANT_ATTRIBUTES),
                            },
                        },
                    },
                    "required": ["text", "difficulty", "required_scene", "required_participants"],
                },
            },
        },
        "required": ["caption_ja", "scene", "participants", "queries_ja"],
    }


SCENE_CARD_PROMPT = """\
入力画像は自動車のドライブレコーダ (車載カメラ) が撮影した交通シーンです。
この画像を検索対象とするための「シーンカード」を、指定された JSON スキーマで出力してください。

出力する内容は以下の3つです。

1. caption_ja
   画像に見える事実だけを述べた日本語のキャプションを2文から4文で書いてください。
   道路環境、写っている対象物、自車から見た位置、向き、動きを具体的に書いてください。
   見えないことや推測は書かないでください。

2. scene と participants
   画像に実際に写っている事実だけを、指定された語彙から選んで構造化してください。
   participants には、画像の中で明確に識別できる対象物だけを、最大6件まで挙げてください。
   小さすぎて属性が判断できない対象は挙げないでください。
   lane_relation は自車から見た位置関係です。
   same_lane は自車と同じ車線、oncoming_lane は対向車線、
   roadside_left / roadside_right は車道の外側の路肩を指します。
   orientation は対象物が向いている方向です。
   away_from_camera は自車と同じ進行方向 (後ろ姿が見える)、
   toward_camera は自車へ正面を向けている状態です。
   判断できない属性は color / orientation / motion に限り "unknown" を選んでください。

3. queries_ja
   この画像を検索するための日本語クエリを2件作ってください。
   1件目は difficulty を "simple" にし、条件を1つか2つだけ含む短いクエリにしてください。
   2件目は difficulty を "compositional" にし、対象物の種類と、位置または向きまたは動きを
   組み合わせた条件を含むクエリにしてください。
   クエリ文は 10文字以上 60文字以内の自然な日本語の名詞句または文にしてください。

   各クエリには、そのクエリ文が要求している条件を required_scene と required_participants で
   機械可読に書いてください。ここが最も重要です。
   - required_scene と required_participants には、クエリ文が明示的に述べている条件だけを書いてください。
   - クエリ文が触れていない属性には必ず "unspecified" を指定し、road_type と extra は空配列にしてください。
   - required_participants の各要素は、クエリ文が言及している対象物ひとつに対応します。
     クエリ文が1台の車だけに言及しているなら要素は1件です。
   - required_scene と required_participants に書いた条件は、必ずこの画像が満たしている必要があります。
     画像が満たしていない条件は絶対に書かないでください。

説明や前置きは出力せず、JSON だけを出力してください。
"""


# ---------------------------------------------------------------------------
# Ollama 呼び出し
# ---------------------------------------------------------------------------


def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def ollama_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_show(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    """教師モデルの同一性を記録するため /api/show の要約を取る。"""

    try:
        payload = ollama_request(f"{base_url}/api/show", {"model": model}, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
        return {"model": model, "error": str(error)}

    details = payload.get("details") or {}
    return {
        "model": model,
        "modified_at": payload.get("modified_at"),
        "details": {
            "family": details.get("family"),
            "format": details.get("format"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
        },
        "digest": payload.get("digest"),
    }


def generate_scene_card(
    image_path: Path,
    *,
    base_url: str,
    model: str,
    schema: dict[str, Any],
    timeout: float,
    num_ctx: int,
    think: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "think": think,
        "format": schema,
        "options": {"temperature": 0.0, "num_ctx": num_ctx},
        "messages": [
            {
                "role": "user",
                "content": SCENE_CARD_PROMPT,
                "images": [encode_image_base64(image_path)],
            }
        ],
    }
    response = ollama_request(f"{base_url}/api/chat", payload, timeout)
    content = (response.get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("教師モデルが空の応答を返しました。")

    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"教師モデルの出力が JSON ではありません: {content[:300]!r}") from error


# ---------------------------------------------------------------------------
# シーンカードの正規化と検証
# ---------------------------------------------------------------------------


def _clamp_enum(value: Any, allowed: Iterable[str], fallback: str | None) -> str | None:
    if isinstance(value, str) and value in set(allowed):
        return value
    return fallback


def normalize_participant(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None

    participant_type = _clamp_enum(raw.get("type"), PARTICIPANT_TYPES, None)
    if participant_type is None:
        return None

    return {
        "type": participant_type,
        "color": _clamp_enum(raw.get("color"), COLORS, UNKNOWN) or UNKNOWN,
        "position": _clamp_enum(raw.get("position"), POSITIONS, UNKNOWN) or UNKNOWN,
        "lane_relation": _clamp_enum(raw.get("lane_relation"), LANE_RELATIONS, UNKNOWN) or UNKNOWN,
        "orientation": _clamp_enum(raw.get("orientation"), ORIENTATIONS, UNKNOWN) or UNKNOWN,
        "motion": _clamp_enum(raw.get("motion"), MOTIONS, UNKNOWN) or UNKNOWN,
    }


def normalize_scene(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    road_type = [value for value in raw.get("road_type", []) if value in set(ROAD_TYPES)]
    extra = [value for value in raw.get("extra", []) if value in set(SCENE_EXTRA)]
    return {
        "road_type": sorted(set(road_type)),
        "time_of_day": _clamp_enum(raw.get("time_of_day"), TIME_OF_DAY, UNKNOWN) or UNKNOWN,
        "weather": _clamp_enum(raw.get("weather"), WEATHER, UNKNOWN) or UNKNOWN,
        "extra": sorted(set(extra)),
    }


def normalize_required_scene(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    road_type = [value for value in raw.get("road_type", []) if value in set(ROAD_TYPES)]
    extra = [value for value in raw.get("extra", []) if value in set(SCENE_EXTRA)]
    return {
        "road_type": sorted(set(road_type)),
        "time_of_day": _clamp_enum(raw.get("time_of_day"), TIME_OF_DAY, UNSPECIFIED) or UNSPECIFIED,
        "weather": _clamp_enum(raw.get("weather"), WEATHER, UNSPECIFIED) or UNSPECIFIED,
        "extra": sorted(set(extra)),
    }


def normalize_required_participant(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None

    participant_type = _clamp_enum(raw.get("type"), PARTICIPANT_TYPES, None)
    if participant_type is None:
        return None

    return {
        "type": participant_type,
        "color": _clamp_enum(raw.get("color"), COLORS, UNSPECIFIED) or UNSPECIFIED,
        "position": _clamp_enum(raw.get("position"), POSITIONS, UNSPECIFIED) or UNSPECIFIED,
        "lane_relation": _clamp_enum(raw.get("lane_relation"), LANE_RELATIONS, UNSPECIFIED) or UNSPECIFIED,
        "orientation": _clamp_enum(raw.get("orientation"), ORIENTATIONS, UNSPECIFIED) or UNSPECIFIED,
        "motion": _clamp_enum(raw.get("motion"), MOTIONS, UNSPECIFIED) or UNSPECIFIED,
    }


def normalize_scene_card(raw: dict[str, Any], *, max_participants: int) -> dict[str, Any]:
    participants = []
    for item in raw.get("participants", [])[:max_participants]:
        participant = normalize_participant(item)
        if participant is not None:
            participants.append(participant)

    queries = []
    for item in raw.get("queries_ja", []):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        required_participants = []
        for candidate in item.get("required_participants", [])[:3]:
            required = normalize_required_participant(candidate)
            if required is not None:
                required_participants.append(required)
        queries.append(
            {
                "text": text,
                "difficulty": _clamp_enum(item.get("difficulty"), DIFFICULTIES, "simple") or "simple",
                "required_scene": normalize_required_scene(item.get("required_scene")),
                "required_participants": required_participants,
            }
        )

    return {
        "caption_ja": (raw.get("caption_ja") or "").strip(),
        "scene": normalize_scene(raw.get("scene")),
        "participants": participants,
        "queries_ja": queries,
    }


# ---------------------------------------------------------------------------
# 制約の充足判定
# ---------------------------------------------------------------------------


def count_scene_constraints(required_scene: dict[str, Any]) -> int:
    count = len(required_scene["road_type"]) + len(required_scene["extra"])
    if required_scene["time_of_day"] != UNSPECIFIED:
        count += 1
    if required_scene["weather"] != UNSPECIFIED:
        count += 1
    return count


def count_participant_constraints(required_participants: list[dict[str, str]]) -> int:
    count = 0
    for required in required_participants:
        count += 1  # type は必ず指定される
        for attribute in PARTICIPANT_ATTRIBUTES:
            if attribute == "type":
                continue
            if required[attribute] != UNSPECIFIED:
                count += 1
    return count


def count_query_constraints(query: dict[str, Any]) -> int:
    return count_scene_constraints(query["required_scene"]) + count_participant_constraints(
        query["required_participants"]
    )


def satisfied_scene_constraints(required_scene: dict[str, Any], scene: dict[str, Any]) -> int:
    satisfied = 0
    for road_type in required_scene["road_type"]:
        if road_type in scene["road_type"]:
            satisfied += 1
    for extra in required_scene["extra"]:
        if extra in scene["extra"]:
            satisfied += 1
    if required_scene["time_of_day"] != UNSPECIFIED and required_scene["time_of_day"] == scene["time_of_day"]:
        satisfied += 1
    if required_scene["weather"] != UNSPECIFIED and required_scene["weather"] == scene["weather"]:
        satisfied += 1
    return satisfied


def participant_match_score(required: dict[str, str], participant: dict[str, str]) -> int | None:
    """
    required の指定属性のうち、participant が満たす数を返す。
    type が一致しない場合は対応付け自体が成立しないため None を返す。
    """

    if required["type"] != participant["type"]:
        return None

    satisfied = 1
    for attribute in PARTICIPANT_ATTRIBUTES:
        if attribute == "type":
            continue
        value = required[attribute]
        if value == UNSPECIFIED:
            continue
        if participant[attribute] == value:
            satisfied += 1
    return satisfied


# --- 制約の接地 (grounding) ------------------------------------------------
# 既定では無効。--ground-constraints を付けたときだけ働く。
#
# 教師モデルは、クエリ文が述べていない条件まで required_scene / required_participants へ
# 書き込むことがある (例:「自車の前方を走行している車」に time_of_day=day が付く)。
# これが誤ラベルとして効いている場合に、クエリ文へ対応する表現が現れない条件を落とす。
# 条件を減らす方向にしか働かないため、正例が負例へ変わることはない。
#
# ただしこの語彙表は手書きなので、それ自体が誤差源になる
# (「金色」を yellow に結び付けられない、など)。
# まず素の教師ラベルで学習・評価し、誤りを確認してから有効にすること。
# シーンカードを作り直す必要はなく、pairs ステージだけを両方の設定で回して比較できる。

CONSTRAINT_KEYWORDS: dict[str, tuple[str, ...]] = {
    # road_type
    "highway": ("高速", "自動車専用", "ハイウェイ", "幹線"),
    "city_street": ("市街", "街中", "街なか", "都市", "繁華", "街路", "市内"),
    "residential_street": ("住宅", "生活道路", "路地"),
    "rural_road": ("郊外", "田舎", "田園", "山道", "農道"),
    "parking_area": ("駐車場", "パーキング"),
    "tunnel": ("トンネル",),
    "bridge": ("橋", "高架"),
    # time_of_day
    "day": ("昼", "日中", "昼間", "明るい"),
    "night": ("夜", "ナイト", "暗い"),
    "dawn_dusk": ("夕", "朝", "薄暮", "日暮れ", "明け方"),
    # weather
    "clear": ("晴", "快晴"),
    "partly_cloudy": ("晴れ間", "薄曇"),
    "overcast": ("曇", "くもり"),
    "rain": ("雨", "雨天"),
    "snow": ("雪",),
    "fog": ("霧", "もや"),
    # scene extra
    "intersection": ("交差点", "十字路", "辻"),
    "crosswalk_marking": ("横断歩道", "ゼブラ"),
    "roadwork": ("工事", "regulation", "規制"),
    "traffic_jam": ("渋滞", "混雑"),
    "curve": ("カーブ", "曲がり"),
    "slope": ("坂", "勾配"),
    "railway_crossing": ("踏切", "線路"),
    "one_way_sign": ("一方通行",),
    "prohibition_sign": ("禁止", "規制標識"),
    "lane_arrow": ("矢印",),
    "traffic_light_red": ("赤信号", "信号が赤", "信号は赤"),
    "traffic_light_green": ("青信号", "信号が青", "信号は青", "緑信号"),
    "traffic_light_yellow": ("黄信号", "信号が黄"),
    "wet_road": ("濡れた", "路面が濡"),
    "snow_on_road": ("積雪", "雪道", "雪が積"),
    "flooding": ("冠水", "水没"),
    "tunnel_entrance": ("トンネルの入", "トンネル入口"),
    "toll_gate": ("料金所", "ゲート"),
    # color
    "white": ("白",),
    "black": ("黒",),
    "gray": ("グレー", "灰"),
    "silver": ("シルバー", "銀"),
    "red": ("赤",),
    "blue": ("青",),
    "green": ("緑", "グリーン"),
    "yellow": ("黄",),
    "orange": ("オレンジ", "橙"),
    "brown": ("茶", "ブラウン"),
    # position
    "foreground_left": ("手前の左", "左手前", "近くの左"),
    "foreground_center": ("すぐ前", "直前", "目の前", "手前"),
    "foreground_right": ("手前の右", "右手前", "近くの右"),
    "middle_left": ("左",),
    "middle_center": ("中央", "正面", "真ん中"),
    "middle_right": ("右",),
    "background_left": ("奥の左", "遠くの左"),
    "background_center": ("奥", "遠く", "前方遠"),
    "background_right": ("奥の右", "遠くの右"),
    # lane_relation
    "same_lane": ("同じ車線", "同一車線", "自車線", "前方", "直前", "目の前", "前を"),
    "left_lane": ("左車線", "左の車線", "左側の車線", "隣の左"),
    "right_lane": ("右車線", "右の車線", "右側の車線", "隣の右"),
    "oncoming_lane": ("対向", "反対車線", "逆方向の車線"),
    "crosswalk": ("横断歩道", "横断"),
    "sidewalk": ("歩道",),
    "roadside_left": ("左の路肩", "左側の路肩", "左端", "左の路側"),
    "roadside_right": ("右の路肩", "右側の路肩", "右端", "右の路側"),
    "off_road": ("路外", "道路外"),
    # orientation
    "away_from_camera": (
        "後ろ姿",
        "同じ方向",
        "前を走",
        "前方を走",
        "前方へ進",
        "遠ざか",
        "背面",
        "後方から",
    ),
    "toward_camera": ("対向", "こちらを向", "正面を向", "向かって", "正面から", "こちらに向"),
    "facing_left": ("左を向", "左向き", "左に向"),
    "facing_right": ("右を向", "右向き", "右に向"),
    "oblique": ("斜め",),
    # motion
    "moving_away": ("走行", "進行", "進む", "走って", "遠ざか", "前を走", "先を走"),
    "moving_toward_camera": ("向かって", "接近", "近づ", "対向"),
    "crossing_road": ("横断", "渡っ", "横切"),
    "turning": ("曲が", "右折", "左折", "旋回"),
    "stopped": ("停止", "止ま", "停まっ"),
    "parked": ("駐車", "停車"),
}

# lane_relation と parking_area は同名の値が別カテゴリにも存在するため個別に補う。
CONSTRAINT_KEYWORDS.setdefault("parking_area", ("駐車場", "パーキング"))


# 静止画1枚では確実に判定できない属性。--drop-motion-constraints で条件から外せる。
# 「停止中」と「駐車中」は1フレームでは区別できないため、教師が同じ状況へ違うラベルを
# 付けることがある。それが誤ラベルとして効いていると分かった場合に外す。
UNRELIABLE_ATTRIBUTES = ("motion",)


def constraint_is_grounded(value: str, query_text: str) -> bool:
    keywords = CONSTRAINT_KEYWORDS.get(value)
    if not keywords:
        # 語彙表にない値は落とさない (判定できないものを勝手に削らない)。
        return True
    return any(keyword in query_text for keyword in keywords)


def refine_query_constraints(
    query: dict[str, Any], *, ground: bool, drop_unreliable: bool
) -> tuple[dict[str, Any], int]:
    """
    教師が付けた条件を絞り込んだ新しいクエリと、落とした条件数を返す。

    ground / drop_unreliable がどちらも False なら、教師の出力をそのまま使う。
    どちらも既定では無効で、ラベルの素性を確かめてから有効にする想定。
    """

    if not ground and not drop_unreliable:
        return query, 0

    text = query["text"]
    dropped = 0

    required_scene = dict(query["required_scene"])
    if ground:
        kept_road_type = [
            value for value in required_scene["road_type"] if constraint_is_grounded(value, text)
        ]
        dropped += len(required_scene["road_type"]) - len(kept_road_type)
        required_scene["road_type"] = kept_road_type

        kept_extra = [
            value for value in required_scene["extra"] if constraint_is_grounded(value, text)
        ]
        dropped += len(required_scene["extra"]) - len(kept_extra)
        required_scene["extra"] = kept_extra

        for key in ("time_of_day", "weather"):
            value = required_scene[key]
            if value != UNSPECIFIED and not constraint_is_grounded(value, text):
                required_scene[key] = UNSPECIFIED
                dropped += 1

    required_participants = []
    for required in query["required_participants"]:
        refined = dict(required)
        for attribute in PARTICIPANT_ATTRIBUTES:
            if attribute == "type":
                # 対象物の種類はクエリの主語そのものなので落とさない。
                continue
            value = refined[attribute]
            if value == UNSPECIFIED:
                continue
            if drop_unreliable and attribute in UNRELIABLE_ATTRIBUTES:
                refined[attribute] = UNSPECIFIED
                dropped += 1
                continue
            if ground and not constraint_is_grounded(value, text):
                refined[attribute] = UNSPECIFIED
                dropped += 1
        required_participants.append(refined)

    return (
        {**query, "required_scene": required_scene, "required_participants": required_participants},
        dropped,
    )


def best_participant_assignment(
    required_participants: list[dict[str, str]],
    participants: list[dict[str, str]],
    *,
    max_participants: int = 12,
) -> int:
    """
    required_participants を participants へ重複なく割り当てたときの、
    満たされる制約数の最大値を返す。

    クエリの対象物は最大3件、画像側は最大 max_participants 件に制限しているため、
    全探索でも十分に速い。
    """

    if not required_participants:
        return 0

    pool = participants[:max_participants]
    if not pool:
        return 0

    best = 0
    indices = range(len(pool))
    for assignment in itertools.permutations(indices, min(len(required_participants), len(pool))):
        total = 0
        for required, participant_index in zip(required_participants, assignment):
            score = participant_match_score(required, pool[participant_index])
            if score is not None:
                total += score
        best = max(best, total)
    return best


def evaluate_query_against_card(query: dict[str, Any], card: dict[str, Any]) -> tuple[int, int]:
    """(満たされた制約数, 制約総数) を返す。"""

    total = count_query_constraints(query)
    satisfied = satisfied_scene_constraints(query["required_scene"], card["scene"])
    satisfied += best_participant_assignment(query["required_participants"], card["participants"])
    return satisfied, total


def query_is_satisfied_by(query: dict[str, Any], card: dict[str, Any]) -> bool:
    satisfied, total = evaluate_query_against_card(query, card)
    return total > 0 and satisfied == total


def flatten_facts(card: dict[str, Any]) -> list[str]:
    """統計・レポート用のフラットな事実集合。ラベル判定には使わない。"""

    facts: set[str] = set()
    scene = card["scene"]
    facts.update(scene["road_type"])
    facts.update(scene["extra"])
    if scene["time_of_day"] != UNKNOWN:
        facts.add(scene["time_of_day"])
    if scene["weather"] != UNKNOWN:
        facts.add(scene["weather"])
    for participant in card["participants"]:
        for attribute in PARTICIPANT_ATTRIBUTES:
            value = participant[attribute]
            if value != UNKNOWN:
                facts.add(value)
    return sorted(facts)


def query_facts(query: dict[str, Any]) -> list[str]:
    facts: set[str] = set()
    required_scene = query["required_scene"]
    facts.update(required_scene["road_type"])
    facts.update(required_scene["extra"])
    if required_scene["time_of_day"] != UNSPECIFIED:
        facts.add(required_scene["time_of_day"])
    if required_scene["weather"] != UNSPECIFIED:
        facts.add(required_scene["weather"])
    for required in query["required_participants"]:
        for attribute in PARTICIPANT_ATTRIBUTES:
            value = required[attribute]
            if value != UNSPECIFIED:
                facts.add(value)
    return sorted(facts)


# ---------------------------------------------------------------------------
# ステージ1: 画像サンプリングとシーンカード生成
# ---------------------------------------------------------------------------


def sample_images(image_dir: Path, num_images: int, seed: int) -> list[Path]:
    if not image_dir.is_dir():
        raise RuntimeError(f"画像ディレクトリが見つかりません: {image_dir}")

    paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"{image_dir} に画像がありません。")

    if num_images <= 0 or num_images >= len(paths):
        return paths

    rng = random.Random(seed)
    return sorted(rng.sample(paths, num_images))


def image_id_for(path: Path) -> str:
    return f"bdd100k:{path.stem}"


def acquire_cards_lock(dataset_dir: Path) -> Path:
    """
    同じデータセットへ cards ステージを二重に走らせないためのロック。

    教師モデルの呼び出しは1000枚で数時間かかるため、二重起動に気づかないまま
    同じ画像を2回ラベリングして時間を捨てることがないようにする。
    """

    lock_path = dataset_dir / "state" / "cards.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{lock_path} が既に存在します。同じデータセットで cards ステージが実行中の可能性があります。"
            " 実行中でないことを確認したうえで、このファイルを削除してから再実行してください。"
        ) from None

    with os.fdopen(descriptor, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\t{datetime.now(timezone.utc).isoformat()}\n")
    return lock_path


def run_cards_stage(args: argparse.Namespace, dataset_dir: Path) -> None:
    image_dir = (PROJECT_ROOT / args.image_dir).resolve()
    images = sample_images(image_dir, args.num_images, args.seed)

    manifest_path = dataset_dir / "manifests" / "sampled_images.jsonl"
    write_jsonl(
        manifest_path,
        (
            {
                "schema_version": SCHEMA_VERSION,
                "image_id": image_id_for(path),
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
            }
            for path in images
        ),
    )
    print(f"サンプリング画像: {len(images)} 件 -> {manifest_path}")

    cards_path = dataset_dir / "raw_teacher" / "scene_cards.jsonl"
    errors_path = dataset_dir / "raw_teacher" / "scene_card_errors.jsonl"

    if args.overwrite:
        for path in (cards_path, errors_path):
            if path.exists():
                path.unlink()

    done_image_ids = {record.get("image_id") for record in read_jsonl(cards_path)}
    pending = [path for path in images if image_id_for(path) not in done_image_ids]
    print(f"生成済み: {len(done_image_ids)} 件 / 未生成: {len(pending)} 件")

    if not pending:
        return

    schema = build_scene_card_schema(args.max_participants, args.max_queries_per_image)
    prompt_sha256 = hashlib.sha256(SCENE_CARD_PROMPT.encode("utf-8")).hexdigest()

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "teacher": ollama_show(args.ollama_url, args.teacher_model, 60.0),
        "num_pending": len(pending),
        "options": {"temperature": 0.0, "num_ctx": args.num_ctx, "think": args.think},
    }
    append_jsonl(dataset_dir / "raw_teacher" / "scene_cards.runs.jsonl", provenance)

    started_at = time.monotonic()
    failure_count = 0

    for index, path in enumerate(pending, start=1):
        try:
            raw_card = generate_scene_card(
                path,
                base_url=args.ollama_url,
                model=args.teacher_model,
                schema=schema,
                timeout=args.ollama_timeout,
                num_ctx=args.num_ctx,
                think=args.think,
            )
            card = normalize_scene_card(raw_card, max_participants=args.max_participants)
        except (
            RuntimeError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
        ) as error:
            failure_count += 1
            append_jsonl(
                errors_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "image_id": image_id_for(path),
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "error": f"{type(error).__name__}: {error}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"[{index}/{len(pending)}] 失敗: {path.name}: {error}")
            if not args.continue_on_error:
                raise
            continue

        append_jsonl(
            cards_path,
            {
                "schema_version": SCHEMA_VERSION,
                "image_id": image_id_for(path),
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "teacher_model": args.teacher_model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                **card,
            },
        )

        elapsed = time.monotonic() - started_at
        rate = index / elapsed if elapsed > 0 else 0.0
        remaining = (len(pending) - index) / rate if rate > 0 else 0.0
        print(
            f"[{index}/{len(pending)}] {path.name} "
            f"participants={len(card['participants'])} queries={len(card['queries_ja'])} "
            f"({rate * 60:.1f} 件/分, 残り約 {remaining / 60:.0f} 分)"
        )

    print(f"シーンカード生成完了。失敗 {failure_count} 件 -> {cards_path}")


# ---------------------------------------------------------------------------
# ステージ2: クエリ選定・分割・ペア構成
# ---------------------------------------------------------------------------


def normalize_query_text(text: str) -> str:
    return "".join(text.split()).strip("。.!！?？")


def assign_splits(
    cards: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """画像単位で train / val / test を割り当てる。"""

    image_ids = sorted(card["image_id"] for card in cards)
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    total = len(image_ids)
    train_end = int(round(total * ratios[0]))
    val_end = train_end + int(round(total * ratios[1]))
    val_end = min(val_end, total)

    split_by_image: dict[str, str] = {}
    for index, image_id in enumerate(image_ids):
        if index < train_end:
            split_by_image[image_id] = "train"
        elif index < val_end:
            split_by_image[image_id] = "val"
        else:
            split_by_image[image_id] = "test"
    return split_by_image


def select_queries(
    cards: list[dict[str, Any]],
    *,
    split_by_image: dict[str, str],
    queries_per_image: int,
    min_constraints: int,
    min_query_length: int,
    max_query_length: int,
    ground_constraints: bool = False,
    drop_unreliable_attributes: bool = False,
) -> tuple[list[dict[str, Any]], Counter]:
    """
    各シーンカードから採用するクエリを選ぶ。

    採用条件:
      - 文字数が範囲内
      - 制約数が min_constraints 以上
      - 生成元画像自身がそのクエリの制約をすべて満たす (自己整合性)
      - 正規化したクエリ文がデータセット内で未出現
    """

    rejected = Counter()
    seen_texts: set[str] = set()
    queries: list[dict[str, Any]] = []
    dropped_constraints = 0

    for card in cards:
        accepted_for_card: list[dict[str, Any]] = []
        # simple を先に、次に compositional を採用し、難易度が偏らないようにする。
        ordered = sorted(
            card["queries_ja"],
            key=lambda query: (query["difficulty"] != "simple", query["text"]),
        )
        for query in ordered:
            if len(accepted_for_card) >= queries_per_image:
                break

            text = query["text"].strip()
            if not (min_query_length <= len(text) <= max_query_length):
                rejected["length"] += 1
                continue

            query, dropped = refine_query_constraints(
                query,
                ground=ground_constraints,
                drop_unreliable=drop_unreliable_attributes,
            )
            dropped_constraints += dropped

            constraint_count = count_query_constraints(query)
            if constraint_count < min_constraints:
                rejected["too_few_constraints"] += 1
                continue

            if not query_is_satisfied_by(query, card):
                rejected["not_self_consistent"] += 1
                continue

            normalized = normalize_query_text(text)
            if normalized in seen_texts:
                rejected["duplicate_text"] += 1
                continue

            seen_texts.add(normalized)
            query_id = "q:" + hashlib.sha1(
                f"{card['image_id']}|{normalized}".encode("utf-8")
            ).hexdigest()[:20]
            accepted_for_card.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "query_id": query_id,
                    "query_text": text,
                    "difficulty": query["difficulty"],
                    "constraint_count": constraint_count,
                    "required_scene": query["required_scene"],
                    "required_participants": query["required_participants"],
                    "supported_facts": query_facts(query),
                    "source_image_id": card["image_id"],
                    "split": split_by_image[card["image_id"]],
                }
            )

        queries.extend(accepted_for_card)

    rejected["_dropped_ungrounded_constraints"] = dropped_constraints
    return queries, rejected


def mine_pairs_for_query(
    query: dict[str, Any],
    cards_by_split: dict[str, list[dict[str, Any]]],
    cards_by_id: dict[str, dict[str, Any]],
    *,
    positives_per_query: int,
    hard_negatives_per_query: int,
    random_negatives_per_query: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    同じ分割のプールから、正例・hard negative・random negative を選ぶ。

    hard negative は「制約の一部だけを満たす」画像で、満たした割合が高いものを優先する。
    random negative は「制約をひとつも満たさない」画像から選ぶ。
    """

    split = query["split"]
    pool = cards_by_split[split]
    source_id = query["source_image_id"]

    positives: list[tuple[str, float]] = []
    hard_negatives: list[tuple[str, float]] = []
    easy_negatives: list[str] = []

    for card in pool:
        if card["image_id"] == source_id:
            continue
        satisfied, total = evaluate_query_against_card(query, card)
        ratio = satisfied / total if total else 0.0
        if total > 0 and satisfied == total:
            positives.append((card["image_id"], ratio))
        elif satisfied > 0:
            hard_negatives.append((card["image_id"], ratio))
        else:
            easy_negatives.append(card["image_id"])

    # 決定的にするため、スコア降順・image_id 昇順で並べる。
    hard_negatives.sort(key=lambda item: (-item[1], item[0]))
    positives.sort(key=lambda item: item[0])
    easy_negatives.sort()

    selected_positive_ids = [source_id]
    for image_id, _ in positives:
        if len(selected_positive_ids) >= positives_per_query:
            break
        selected_positive_ids.append(image_id)

    selected_hard_ids = [image_id for image_id, _ in hard_negatives[:hard_negatives_per_query]]

    random_count = min(random_negatives_per_query, len(easy_negatives))
    selected_random_ids = rng.sample(easy_negatives, random_count) if random_count else []

    # hard negative が足りない場合は random negative で補い、候補数を揃える。
    shortage = hard_negatives_per_query - len(selected_hard_ids)
    if shortage > 0:
        remaining = [
            image_id for image_id in easy_negatives if image_id not in set(selected_random_ids)
        ]
        if remaining:
            selected_random_ids.extend(rng.sample(remaining, min(shortage, len(remaining))))

    pairs: list[dict[str, Any]] = []

    def add_pair(image_id: str, label: int, negative_type: str) -> None:
        card = cards_by_id[image_id]
        satisfied, total = evaluate_query_against_card(query, card)
        pair_id = "pair:" + hashlib.sha1(
            f"{query['query_id']}|{image_id}".encode("utf-8")
        ).hexdigest()[:20]
        pairs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_id": pair_id,
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "instruction": DASHCAM_RERANKER_PROMPT,
                "difficulty": query["difficulty"],
                "constraint_count": query["constraint_count"],
                "supported_facts": query["supported_facts"],
                "split": split,
                "image_id": image_id,
                "image_path": card["relative_path"],
                "label": label,
                "negative_type": negative_type,
                "satisfied_constraints": satisfied,
                "total_constraints": total,
                "satisfied_ratio": round(satisfied / total, 4) if total else 0.0,
                "is_source_image": image_id == source_id,
                "label_source": "scene_card_constraint_match_v1",
            }
        )

    for image_id in selected_positive_ids:
        add_pair(image_id, 1, "positive")
    for image_id in selected_hard_ids:
        add_pair(image_id, 0, "hard_negative")
    for image_id in selected_random_ids:
        add_pair(image_id, 0, "random_negative")

    return pairs


def run_pairs_stage(args: argparse.Namespace, dataset_dir: Path) -> None:
    # --scene-cards を使うと、既存のラベリング結果から別設定のデータセットを作れる。
    # 教師モデルの呼び出しは数時間かかるので、設定を変えた比較ではカードを共有する。
    if args.scene_cards:
        cards_path = (PROJECT_ROOT / args.scene_cards).resolve()
    else:
        cards_path = dataset_dir / "raw_teacher" / "scene_cards.jsonl"

    cards = read_jsonl(cards_path)
    if not cards:
        raise RuntimeError(
            f"{cards_path} にシーンカードがありません。先に --stage cards を実行してください。"
        )
    print(f"シーンカード: {cards_path}")

    # 同じ画像が複数回書かれている場合は最後の1件を採用する。
    cards_by_id = {card["image_id"]: card for card in cards}
    cards = sorted(cards_by_id.values(), key=lambda card: card["image_id"])
    print(f"シーンカード: {len(cards)} 件")

    ratios = tuple(args.split_ratios)
    split_by_image = assign_splits(cards, ratios, args.split_seed)

    cards_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for card in cards:
        cards_by_split[split_by_image[card["image_id"]]].append(card)

    queries, rejected = select_queries(
        cards,
        split_by_image=split_by_image,
        queries_per_image=args.queries_per_image,
        min_constraints=args.min_constraints,
        min_query_length=args.min_query_length,
        max_query_length=args.max_query_length,
        ground_constraints=args.ground_constraints,
        drop_unreliable_attributes=args.drop_unreliable_attributes,
    )
    print(f"採用クエリ: {len(queries)} 件 (不採用: {dict(rejected)})")

    if args.max_queries > 0 and len(queries) > args.max_queries:
        rng = random.Random(args.split_seed)
        queries = sorted(rng.sample(queries, args.max_queries), key=lambda query: query["query_id"])
        print(f"--max-queries により {len(queries)} 件へ削減しました。")

    write_jsonl(dataset_dir / "derived" / "queries.jsonl", queries)

    pairs_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(args.split_seed)

    for index, query in enumerate(queries, start=1):
        is_eval = query["split"] in ("val", "test")
        pairs = mine_pairs_for_query(
            query,
            cards_by_split,
            cards_by_id,
            positives_per_query=args.positives_per_query,
            hard_negatives_per_query=(
                args.eval_hard_negatives_per_query if is_eval else args.hard_negatives_per_query
            ),
            random_negatives_per_query=(
                args.eval_random_negatives_per_query if is_eval else args.random_negatives_per_query
            ),
            rng=rng,
        )
        pairs_by_split[query["split"]].extend(pairs)
        if index % 100 == 0:
            print(f"ペア構成: {index}/{len(queries)} クエリ")

    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": relative_to_project(dataset_dir, PROJECT_ROOT),
        "scene_cards": len(cards),
        "queries_total": len(queries),
        "query_rejections": dict(rejected),
        "constraint_grounding": args.ground_constraints,
        "drop_unreliable_attributes": (
            list(UNRELIABLE_ATTRIBUTES) if args.drop_unreliable_attributes else []
        ),
        "split_ratios": list(ratios),
        "split_seed": args.split_seed,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        pairs = sorted(pairs_by_split[split], key=lambda pair: (pair["query_id"], pair["pair_id"]))
        path = dataset_dir / f"pairs.{split}.jsonl"
        write_jsonl(path, pairs)

        split_queries = [query for query in queries if query["split"] == split]
        positives = sum(1 for pair in pairs if pair["label"] == 1)
        stats["splits"][split] = {
            "images": len(cards_by_split[split]),
            "queries": len(split_queries),
            "pairs": len(pairs),
            "positives": positives,
            "negatives": len(pairs) - positives,
            "hard_negatives": sum(1 for pair in pairs if pair["negative_type"] == "hard_negative"),
            "random_negatives": sum(
                1 for pair in pairs if pair["negative_type"] == "random_negative"
            ),
            "difficulty": dict(Counter(query["difficulty"] for query in split_queries)),
            "mean_constraints": (
                round(
                    sum(query["constraint_count"] for query in split_queries) / len(split_queries),
                    3,
                )
                if split_queries
                else 0.0
            ),
        }
        print(
            f"{split}: 画像 {stats['splits'][split]['images']} / "
            f"クエリ {len(split_queries)} / ペア {len(pairs)} "
            f"(正例 {positives}) -> {path}"
        )

    # 分割間のリークがないことを明示的に検証して記録する。
    images_by_split = {
        split: {card["image_id"] for card in cards_by_split[split]} for split in cards_by_split
    }
    queries_by_split = {
        split: {normalize_query_text(query["query_text"]) for query in queries if query["split"] == split}
        for split in cards_by_split
    }
    isolation: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "image_overlap": {}, "query_text_overlap": {}}
    for left, right in itertools.combinations(("train", "val", "test"), 2):
        isolation["image_overlap"][f"{left}|{right}"] = len(
            images_by_split[left] & images_by_split[right]
        )
        isolation["query_text_overlap"][f"{left}|{right}"] = len(
            queries_by_split[left] & queries_by_split[right]
        )
    for split, pairs in pairs_by_split.items():
        used_images = {pair["image_id"] for pair in pairs}
        isolation.setdefault("candidate_images_outside_split", {})[split] = len(
            used_images - images_by_split[split]
        )

    write_json(dataset_dir / "reports" / "split_isolation.json", isolation)
    write_json(dataset_dir / "reports" / "dataset_stats.json", stats)

    fact_counter = Counter()
    for card in cards:
        fact_counter.update(flatten_facts(card))
    write_json(
        dataset_dir / "reports" / "fact_coverage.json",
        {
            "schema_version": SCHEMA_VERSION,
            "images": len(cards),
            "fact_counts": dict(fact_counter.most_common()),
        },
    )

    leaked = sum(isolation["image_overlap"].values()) + sum(isolation["query_text_overlap"].values())
    leaked += sum(isolation.get("candidate_images_outside_split", {}).values())
    if leaked:
        raise RuntimeError(f"分割間のリークを検出しました: {isolation}")
    print("分割間のリークなし。レポートを reports/ に書き出しました。")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Reranker-8B のドライブレコーダ特化学習/評価データセットを作る。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["all", "cards", "pairs"], default="all")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-images", type=int, default=1000, help="ラベリングする画像枚数。")
    parser.add_argument("--seed", type=int, default=42, help="画像サンプリングの乱数シード。")

    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_OLLAMA_TIMEOUT)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="教師モデルのコンテキスト長。VRAM 16GB では 4096 が上限の目安。",
    )
    parser.add_argument("--think", action="store_true", help="教師モデルの thinking を有効にする。")
    parser.add_argument("--max-participants", type=int, default=6)
    parser.add_argument("--max-queries-per-image", type=int, default=2)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument(
        "--stop-on-error",
        dest="continue_on_error",
        action="store_false",
        help="画像1件の失敗で処理を停止する。",
    )
    parser.add_argument("--overwrite", action="store_true", help="既存のシーンカードを破棄して作り直す。")

    parser.add_argument("--queries-per-image", type=int, default=2)
    parser.add_argument(
        "--scene-cards",
        default=None,
        help=(
            "pairs ステージが読むシーンカードのパス。"
            " 既定は --dataset-dir 配下。既存のラベリング結果を再利用して"
            " 別設定のデータセットを作るときに指定する。"
        ),
    )
    parser.add_argument("--max-queries", type=int, default=0, help="0 なら制限しない。")
    parser.add_argument("--min-constraints", type=int, default=2)
    parser.add_argument(
        "--ground-constraints",
        action="store_true",
        default=False,
        help=(
            "クエリ文に対応する日本語表現が現れない条件を、ラベル判定から落とす。"
            " 教師の過剰指定が誤ラベルの原因だと確認できてから使う。"
        ),
    )
    parser.add_argument(
        "--drop-motion-constraints",
        dest="drop_unreliable_attributes",
        action="store_true",
        default=False,
        help="静止画1枚では判定しづらい motion 条件を、ラベル判定から外す。",
    )
    parser.add_argument("--min-query-length", type=int, default=8)
    parser.add_argument("--max-query-length", type=int, default=60)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.1, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--positives-per-query", type=int, default=2)
    parser.add_argument("--hard-negatives-per-query", type=int, default=3)
    parser.add_argument("--random-negatives-per-query", type=int, default=2)
    parser.add_argument("--eval-hard-negatives-per-query", type=int, default=5)
    parser.add_argument("--eval-random-negatives-per-query", type=int, default=4)

    args = parser.parse_args()

    if abs(sum(args.split_ratios) - 1.0) > 1e-6:
        parser.error("--split-ratios の合計は 1.0 にしてください。")
    if args.positives_per_query < 1:
        parser.error("--positives-per-query は 1 以上を指定してください。")

    return args


def main() -> None:
    args = parse_args()
    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "cards"):
        lock_path = acquire_cards_lock(dataset_dir)
        try:
            run_cards_stage(args, dataset_dir)
        finally:
            lock_path.unlink(missing_ok=True)
    if args.stage in ("all", "pairs"):
        run_pairs_stage(args, dataset_dir)


if __name__ == "__main__":
    main()
