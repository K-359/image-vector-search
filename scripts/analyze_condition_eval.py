"""
v3 条件指定データの評価結果を、条件の種類別・未学習の組み合わせ (H) 別に集計する。

evaluate_reranker.py が出力した scores_*.jsonl を読むだけで、モデルは呼ばない。
v3 では1クエリ = 1条件で、1条件に正例が10件以上あることが多いため MRR はほぼ飽和する。
主指標は次の2つにする。

  - クエリ内 AUC: 同じ条件の (正例, 負例) の組のうち、正例のスコアが高い割合
  - 検索上位負例の誤り率: 検索上位から選んだ不適合画像 (hard_negative) が
    同じ条件の正例以上のスコアを取った組の割合 (低いほど良い)

使い方:

    python scripts/analyze_condition_eval.py \\
        datasets/dashcam_reranker_v3_conditions/reports/scores_test_v3-conditions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_reranker import paired_bootstrap, per_query_metrics  # noqa: E402

VARIANTS = ("base", "adapter")
CATEGORY_NAMES = {
    "OBJ": "対象物の存在",
    "COL": "対象物の色",
    "POS": "画面内の位置",
    "LOC": "道路・自車との位置",
    "ORI": "向き",
    "ROAD": "道路・設備・路面",
    "ENV": "時間帯・天候",
    "CO": "2対象の共存",
    "CE": "環境＋対象",
    "CA": "1対象の複数属性",
    "CB": "複数対象への属性の結び付け",
    "R": "希少対象",
    "X": "追加の判定",
    "H": "未学習の組み合わせ (test専用)",
}


def category_of(condition_id: str) -> str:
    return re.match(r"[A-Z]+", condition_id).group(0)


def ordered_fraction(high: list[float], low: list[float]) -> float | None:
    """high 側が low 側より高い組の割合。同点は 0.5 と数える。"""

    if not high or not low:
        return None
    wins = sum(1.0 if h > l else 0.5 if h == l else 0.0 for h in high for l in low)
    return wins / (len(high) * len(low))


def condition_metrics(rows: list[dict], variant: str) -> dict[str, float]:
    key = f"score_{variant}"
    ordered = sorted(rows, key=lambda row: (-row[key], row["pair_id"]))
    metrics = per_query_metrics([row["label"] for row in ordered], [row["label"] for row in rows])

    positives = [row[key] for row in rows if row["label"] == 1]
    negatives = [row[key] for row in rows if row["label"] == 0]
    hard = [row[key] for row in rows if row["negative_type"] == "hard_negative"]
    metrics["auc"] = ordered_fraction(positives, negatives)
    hard_ordered = ordered_fraction(positives, hard)
    metrics["hard_error"] = None if hard_ordered is None else 1.0 - hard_ordered
    metrics["accuracy@0"] = sum((row[key] > 0) == (row["label"] == 1) for row in rows) / len(rows)
    return metrics


def mean(values: list[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(conditions: dict[str, dict], ids: list[str], *, iterations: int, seed: int) -> dict:
    summary: dict = {"conditions": len(ids), "pairs": sum(conditions[i]["pairs"] for i in ids)}
    for metric in ("auc", "hard_error", "ndcg@5", "map", "accuracy@0"):
        entry = {v: mean([conditions[i][v][metric] for i in ids]) for v in VARIANTS}
        usable = [i for i in ids if conditions[i]["base"][metric] is not None]
        if len(usable) >= 2:
            results = {v: {i: {"metrics": conditions[i][v]} for i in usable} for v in VARIANTS}
            stats = paired_bootstrap(
                results["base"], results["adapter"], metric,
                clusters={}, iterations=iterations, seed=seed,
            )
            entry.update(delta=stats["delta"], ci=[stats["ci_lower"], stats["ci_upper"]])
        summary[metric] = entry
    return summary


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def format_markdown(payload: dict) -> str:
    lines = [
        "# v3 条件別の評価集計",
        "",
        f"- 入力: `{payload['scores']}`",
        f"- 条件数: {payload['overall']['conditions']} / ペア数: {payload['overall']['pairs']}",
        "- 信頼区間は条件単位のペアード・ブートストラップ (95%)。同じ画像が複数条件に現れる相関は考慮していない。",
        "- ラベルはすべて教師モデル (Qwen3.8) の判定であり、人手で確認した正解ではない。",
        "",
    ]
    for metric, title, note in (
        ("auc", "クエリ内 AUC", "高いほど良い"),
        ("hard_error", "検索上位負例の誤り率", "低いほど良い"),
    ):
        lines += [
            f"## {title} ({note})",
            "",
            "| 区分 | 条件数 | base | adapter | 差分 | 95%CI |",
            "| --- | ---: | ---: | ---: | ---: | :---: |",
        ]
        for name, summary in payload["groups"].items():
            entry = summary[metric]
            ci = entry.get("ci")
            ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
            lines.append(
                f"| {name} | {summary['conditions']} | {fmt(entry['base'])} | "
                f"{fmt(entry['adapter'])} | {fmt(entry.get('delta'), True)} | {ci_text} |"
            )
        lines.append("")

    lines += [
        "## adapter で AUC が下がった条件",
        "",
        "| 条件 | クエリ | base | adapter | 差分 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["worsened"]:
        lines.append(
            f"| {row['condition_id']} | {row['query_text']} | {fmt(row['base'])} | "
            f"{fmt(row['adapter'])} | {fmt(row['delta'], True)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path, help="evaluate_reranker.py の scores_*.jsonl")
    parser.add_argument("--train-pairs", type=Path, default=None, help="既定は scores と同じデータセットの pairs.train.jsonl")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    by_condition: dict[str, list[dict]] = defaultdict(list)
    with args.scores.open() as f:
        for line in f:
            row = json.loads(line)
            by_condition[row["query_id"].rsplit(":", 1)[-1]].append(row)

    train_pairs = args.train_pairs or args.scores.parent.parent / "pairs.train.jsonl"
    with train_pairs.open() as f:
        seen_in_train = {json.loads(line)["condition_id"] for line in f}

    conditions = {}
    for condition_id, rows in by_condition.items():
        if not any(row["label"] for row in rows) or all(row["label"] for row in rows):
            continue  # 正例・負例の片方しかない条件は順位を評価できない
        conditions[condition_id] = {
            "query_text": rows[0]["query_text"],
            "pairs": len(rows),
            "seen_in_train": condition_id in seen_in_train,
            **{v: condition_metrics(rows, v) for v in VARIANTS},
        }

    options = {"iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed}
    all_ids = sorted(conditions)
    groups = {
        "全体": all_ids,
        "学習済みの条件": [i for i in all_ids if conditions[i]["seen_in_train"]],
        "学習に無い条件": [i for i in all_ids if not conditions[i]["seen_in_train"]],
    }
    for category, name in CATEGORY_NAMES.items():
        groups[f"{category} {name}"] = [i for i in all_ids if category_of(i) == category]

    worsened = sorted(
        (
            {
                "condition_id": i,
                "query_text": conditions[i]["query_text"],
                "base": conditions[i]["base"]["auc"],
                "adapter": conditions[i]["adapter"]["auc"],
                "delta": conditions[i]["adapter"]["auc"] - conditions[i]["base"]["auc"],
            }
            for i in all_ids
        ),
        key=lambda row: row["delta"],
    )
    payload = {
        "scores": str(args.scores),
        "overall": summarize(conditions, all_ids, **options),
        "groups": {name: summarize(conditions, ids, **options) for name, ids in groups.items() if ids},
        "worsened": [row for row in worsened if row["delta"] < 0],
        "conditions": conditions,
    }

    stem = args.scores.stem.replace("scores_", "conditions_", 1)
    json_path = args.scores.with_name(f"{stem}.json")
    md_path = args.scores.with_name(f"{stem}.md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(format_markdown(payload))
    print(md_path.read_text())
    print(f"保存しました: {md_path}")


if __name__ == "__main__":
    main()
