"""検索要求に対するシーンカードの記載を集計する。正解ラベルは生成しない。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RIGHT = ["foreground_right", "middle_right", "background_right"]
LEFT = ["foreground_left", "middle_left", "background_left"]


def requirements() -> list[dict]:
    rows = []

    def add(rid, query, people=(), scene=None, note=""):
        rows.append(dict(id=rid, query=query, people=list(people), scene=scene or {},
                         note=note, mode="and"))

    for i, (kind, name) in enumerate([
        ("car", "車"), ("bus", "バス"), ("truck", "トラック"), ("van", "バン"),
        ("pedestrian", "歩行者"), ("bicycle", "自転車"), ("motorcycle", "バイク"),
    ], 1):
        add(f"S{i:02}", f"{name}がいる画像", [{"type": kind}])
    add("S08", "赤い車", [{"type": "car", "color": "red"}])
    add("S09", "白いバス", [{"type": "bus", "color": "white"}])
    position_note = "既存の位置語彙による近似。画面3等分の規約で再確認が必要"
    add("S10", "画面右側の自転車", [{"type": "bicycle", "position": RIGHT}], note=position_note)
    add("S11", "画面左側の歩行者", [{"type": "pedestrian", "position": LEFT}], note=position_note)
    add("S12", "同じ車線の車", [{"type": "car", "lane_relation": "same_lane"}])
    add("S13", "歩道上の歩行者", [{"type": "pedestrian", "lane_relation": "sidewalk"}])
    add("S14", "横断歩道上の歩行者", [{"type": "pedestrian", "lane_relation": "crosswalk"}])
    add("S15", "正面を向けた自転車", [{"type": "bicycle", "orientation": "toward_camera"}])
    add("S16", "後ろ姿の車", [{"type": "car", "orientation": "away_from_camera"}])
    add("S17", "トンネル内", scene={"road_type": "tunnel"}, note="トンネル内と入口の区別を画像で確認")
    add("S18", "赤信号", scene={"extra": "traffic_light_red"})
    add("S19", "濡れた路面", scene={"extra": "wet_road"})
    add("S20", "路面の雪", scene={"extra": "snow_on_road"})
    add("S21", "工事区間", scene={"extra": "roadwork"})
    add("S22", "夜の道路", scene={"time_of_day": "night"})
    add("S23", "霧のある道路", scene={"weather": "fog"}, note="霧と煙・レンズの曇りを再確認")
    add("S24", "降雨中", scene={"weather": "rain"}, note="降雨の直接的な根拠を再確認")
    add("S25", "バスと自転車", [{"type": "bus"}, {"type": "bicycle"}])
    add("S26", "赤い車と白いバス", [{"type": "car", "color": "red"}, {"type": "bus", "color": "white"}])
    add("S27", "夜の横断歩道上の歩行者", [{"type": "pedestrian", "lane_relation": "crosswalk"}], {"time_of_day": "night"})
    add("S28", "路面に雪があり車がいる", [{"type": "car"}], {"extra": "snow_on_road"})
    add("S29", "画面右側のバイク", [{"type": "motorcycle", "position": RIGHT}], note=position_note)
    add("S30", "同じ車線の正面向き自転車", [{"type": "bicycle", "lane_relation": "same_lane", "orientation": "toward_camera"}], note="逆走判定ではない")
    add("S31", "車道上の歩行者", [{"type": "pedestrian", "lane_relation": ["same_lane", "left_lane", "right_lane", "oncoming_lane", "crosswalk"]}], note="車道の直接項目がないため車線等の記載から近似。路肩は含めない")
    add("S32", "赤信号と車", [{"type": "car"}], {"extra": "traffic_light_red"}, note="信号無視判定ではない")
    for rid, query, note in [
        ("X01", "トラックより左の歩行者", "対象間の座標・関係の注釈が必要"),
        ("X02", "車がちょうど2台", "最大6対象のカードから総数は確定できない"),
        ("X03", "歩行者が写っていない", "非網羅的なカードから不在は確定できない"),
    ]:
        rows.append(dict(id=rid, query=query, mode="unsupported", note=note))
    add("X04", "バスまたはトラック", [{"type": "bus"}, {"type": "truck"}], note="記載の有無はOR集計可能。既存学習パイプラインのOR対応とは別")
    rows[-1]["mode"] = "or"
    for i, query in enumerate(["逆走自転車", "飛び出し", "急な横断", "急停止", "急接近", "信号無視"], 1):
        rows.append(dict(id=f"D{i:02}", query=query, mode="unsupported", note="時系列・道路情報・行動判定規約が必要"))
    return rows


def matches(actual, expected) -> bool:
    choices = expected if isinstance(expected, list) else [expected]
    if isinstance(actual, list):
        return any(value in actual for value in choices)
    return actual in choices


def evidence(req: dict, card: dict) -> tuple[int, int] | None:
    """一致する明示条件の最大数。0は不在・不適合を意味しない。"""
    if req["mode"] == "unsupported":
        return None
    people = card.get("participants", [])
    if req["mode"] == "or":
        found = any(all(matches(p.get(k), v) for k, v in condition.items())
                    for condition in req["people"] for p in people)
        return int(found), 1
    scene_score = sum(matches(card.get("scene", {}).get(k), v) for k, v in req["scene"].items())
    required = req["people"]
    total = len(req["scene"]) + sum(len(p) for p in required)
    # 未記載の対象を埋めるダミーを用意し、同じ対象を2回使わない。
    pool = people + [{} for _ in required]
    best = 0
    for assignment in itertools.permutations(range(len(pool)), len(required)):
        score = sum(sum(matches(pool[index].get(k), v) for k, v in condition.items())
                    for condition, index in zip(required, assignment))
        best = max(best, score)
    return scene_score + best, total


def classify(req: dict, card: dict) -> str:
    result = evidence(req, card)
    if result is None:
        return "unsupported"
    score, total = result
    if score == total:
        return "supported_candidate"
    if total > 1 and score == total - 1:
        return "one_condition_short"
    return "unconfirmed"


def read_unique(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"重複image_id: {path}")
    return rows


def audit(dataset: Path) -> dict:
    path = dataset / "raw_teacher/scene_cards.jsonl"
    cards = read_unique(path)
    manifest_path = dataset / "manifests/sampled_images.jsonl"
    manifest = read_unique(manifest_path)
    ids = {row["image_id"] for row in manifest}
    by_id = {row["image_id"]: row for row in cards}
    if ids - by_id.keys():
        raise ValueError("manifestに含まれる画像のカードが不足しています")
    selected = [by_id[key] for key in sorted(ids)]
    if not selected:
        raise ValueError("manifestが空です")
    results = []
    for req in requirements():
        groups = {key: [] for key in ["supported_candidate", "one_condition_short", "unconfirmed", "unsupported"]}
        for card in selected:
            groups[classify(req, card)].append(card["image_id"])
        results.append({**req, "counts": {key: len(value) for key, value in groups.items()},
                        "image_ids": {key: groups[key] for key in ["supported_candidate", "one_condition_short"]}})
    return {
        "schema_version": 1,
        "dataset": str(dataset.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "images": len(selected),
        "ignored_cards_outside_manifest": len(cards) - len(selected),
        "teacher_models": sorted({c["teacher_model"] for c in selected}),
        "rules_sha256": hashlib.sha256(json.dumps(requirements(), sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_paths": {c["image_id"]: c["relative_path"] for c in selected},
        "weather_counts": dict(Counter(c["scene"]["weather"] for c in selected)),
        "pedestrian_lane_counts": dict(Counter(p["lane_relation"] for c in selected for p in c["participants"] if p["type"] == "pedestrian")),
        "all_participants_sidewalk_count": sum(p["lane_relation"] == "sidewalk" for c in selected for p in c["participants"]),
        "requirements": results,
    }


def markdown(report: dict) -> str:
    lines = ["# 検索要求に対する既存カードの充足状況", "",
             f"対象: `{report['dataset']}` / {report['images']}画像。", "",
             "再実行: `python3 scripts/audit_search_coverage.py`", "",
             "これは教師カードの明示記載による候補集計です。実画像の正解数・精度ではありません。",
             "全条件一致も画像確認が必要です。1条件不足は負例を探すための確認候補であり、負例ラベルではありません。",
             "未確認は残りのカードです。カードにない対象・設備の不在を意味せず、条件不一致と情報欠落も混在します。",
             "既存カードの位置・天候等の語彙を使うため、新しい注釈規約への適合は保証されません。",
             "対応不可の行はゼロ件ではなく集計不能（—）です。", "",
             "| ID | 要求 | 全条件一致候補 | 1条件不足候補 | 未確認 | 備考 |",
             "|---|---|---:|---:|---:|---|"]
    for req in report["requirements"]:
        c = req["counts"]
        counts = "— | — | —" if req["mode"] == "unsupported" else f"{c['supported_candidate']} | {c['one_condition_short']} | {c['unconfirmed']}"
        lines.append(f"| {req['id']} | {req['query']} | {counts} | {req['note']} |")
    lines += ["", "全画像IDと照合条件は同名のJSONレポートに保存しています。",
              "JSONの要求別ID一覧は全条件一致・1条件不足の候補を保存し、未確認は全画像IDとの差集合です（対応不可の要求は全画像が集計不能）。",
              "要求ごとの件数は重複するため、合計を独立した画像数として使えません。",
              "必要件数・不足数の確定には、実画像の確認と学習・評価の配分設計が必要です。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/dashcam_reranker_ft_v2_qwen38")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/reports/search-coverage-qwen38")
    args = parser.parse_args()
    report = audit(args.dataset_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"{report['images']}画像 / {len(report['requirements'])}要求 → {args.output_prefix}.md / .json")


if __name__ == "__main__":
    main()
