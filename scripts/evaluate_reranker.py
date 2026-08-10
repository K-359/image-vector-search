"""
Qwen3-VL-Reranker-8B の学習前後の精度を、同一条件で定量比較する。

同じプロセス内で1つのモデルをロードし、LoRA アダプタを無効化した状態 (base) と
有効化した状態 (adapter) の両方でテストセットを採点する。
量子化・画像解像度・命令文・クエリ正規化はすべて本番検索 (scripts/search.py) と同じ経路を通る。
比較条件が完全に揃うため、差分をモデルの変化だけに帰属できる。

出力する指標:

  ランキング (クエリ単位で候補を並べ替える。本番の再ランキングと同じ設定)
    - nDCG@1 / @5 / @10
    - MRR
    - Recall@1 / @3 / @5
    - MAP
  二値識別 (ペア単位)
    - ROC-AUC / PR-AUC
    - スコア 0 を閾値としたときの正解率・適合率・再現率

base と adapter の差については、動画グループ単位のペアード・クラスタ・ブートストラップで
95% 信頼区間と p 値を出す。信頼区間が 0 をまたぐ場合、その差は有意ではない。

この評価が測るのは「教師シーンカードの定義に対する適合度」であり、本番検索の精度そのものではない。
候補集合は test split (数百枚) の中から教師の構造化事実で選んだものなので、
本番の「10万枚から初段検索が返した上位50件を再ランキングする」設定とは異なる。
本番精度を測るには、初段検索の実際の上位候補と人手で作った正解集合が必要になる。

使い方:

    python scripts/evaluate_reranker.py \\
        --dataset-dir datasets/dashcam_reranker_ft_v1 \\
        --adapter-path models/qwen3-vl-reranker-8b-dashcam-v1

    # 学習前のベースラインだけ測る
    python scripts/evaluate_reranker.py --no-adapter
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reranker_common import (  # noqa: E402
    DASHCAM_RERANKER_PROMPT,
    RERANKER_MODEL_NAME,
    PairRecord,
    attach_adapter,
    load_pairs,
    load_reranker,
    relative_to_project,
    score_pairs,
    write_json,
    write_jsonl,
)

PROJECT_ROOT = SCRIPTS_DIR.parent

DEFAULT_DATASET_DIR = "datasets/dashcam_reranker_ft_v1"
DEFAULT_ADAPTER_PATH = "models/qwen3-vl-reranker-8b-dashcam-v1"
ADAPTER_NAME = "dashcam"

RANK_CUTOFFS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------


def dcg(labels: list[int], cutoff: int) -> float:
    return sum(
        label / math.log2(index + 1) for index, label in enumerate(labels[:cutoff], start=1)
    )


def per_query_metrics(labels_in_rank_order: list[int], all_labels: list[int]) -> dict[str, float]:
    """1クエリ分のランキング指標。labels_in_rank_order はスコア降順に並べたラベル列。"""

    metrics: dict[str, float] = {}
    ideal = sorted(all_labels, reverse=True)
    positive_count = sum(all_labels)

    for cutoff in RANK_CUTOFFS:
        idcg = dcg(ideal, cutoff)
        metrics[f"ndcg@{cutoff}"] = dcg(labels_in_rank_order, cutoff) / idcg if idcg else 0.0
        hits = sum(labels_in_rank_order[:cutoff])
        metrics[f"recall@{cutoff}"] = hits / positive_count if positive_count else 0.0

    rank = next(
        (index for index, label in enumerate(labels_in_rank_order, start=1) if label == 1), None
    )
    metrics["mrr"] = 1.0 / rank if rank else 0.0

    hits = 0
    precision_sum = 0.0
    for index, label in enumerate(labels_in_rank_order, start=1):
        if label == 1:
            hits += 1
            precision_sum += hits / index
    metrics["map"] = precision_sum / positive_count if positive_count else 0.0

    return metrics


def rank_pairs(
    pairs: list[PairRecord], scores: list[float]
) -> dict[str, dict[str, Any]]:
    """クエリごとにスコア降順で並べ、ランキング指標を計算する。"""

    by_query: dict[str, list[tuple[float, PairRecord]]] = {}
    for pair, score in zip(pairs, scores):
        by_query.setdefault(pair.query_id, []).append((score, pair))

    results: dict[str, dict[str, Any]] = {}
    for query_id, candidates in by_query.items():
        all_labels = [pair.label for _, pair in candidates]
        if not any(all_labels):
            # 正例のない候補集合ではランキング指標が定義できないため除外する。
            continue

        # 同点のときに元の並び順で有利にならないよう、pair_id で決定的にタイブレークする。
        ordered = sorted(candidates, key=lambda item: (-item[0], item[1].pair_id))
        ordered_labels = [pair.label for _, pair in ordered]

        metrics = per_query_metrics(ordered_labels, all_labels)
        metrics["candidates"] = float(len(candidates))
        top_pair = ordered[0][1]
        results[query_id] = {
            "metrics": metrics,
            "top1_negative_type": top_pair.negative_type if top_pair.label == 0 else None,
            "difficulty": (top_pair.raw or {}).get("difficulty"),
        }
    return results


def binary_metrics(pairs: list[PairRecord], scores: list[float]) -> dict[str, float]:
    labels = [pair.label for pair in pairs]
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return {}

    metrics: dict[str, float] = {}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    except ImportError:
        pass

    predictions = [1 if score > 0 else 0 for score in scores]
    true_positive = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    false_positive = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    false_negative = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    correct = sum(1 for label, pred in zip(labels, predictions) if label == pred)

    metrics["accuracy@0"] = correct / len(labels)
    metrics["precision@0"] = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    metrics["recall@0"] = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    metrics["positive_rate"] = positives / len(labels)
    return metrics


def hard_negative_error_rate(pairs: list[PairRecord], scores: list[float]) -> float:
    """
    「正例より高いスコアを付けられた hard negative」の割合。

    ドライブレコーダ検索で問題になるのは、条件の一部だけ合う画像を上位に出す誤りなので、
    この指標を個別に見る。
    """

    by_query: dict[str, list[tuple[float, PairRecord]]] = {}
    for pair, score in zip(pairs, scores):
        by_query.setdefault(pair.query_id, []).append((score, pair))

    total = 0
    inverted = 0
    for candidates in by_query.values():
        positive_scores = [score for score, pair in candidates if pair.label == 1]
        if not positive_scores:
            continue
        best_positive = max(positive_scores)
        for score, pair in candidates:
            if pair.negative_type != "hard_negative":
                continue
            total += 1
            if score >= best_positive:
                inverted += 1
    return inverted / total if total else 0.0


def aggregate(query_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    if not query_results:
        return {}
    keys = next(iter(query_results.values()))["metrics"].keys()
    return {
        key: sum(result["metrics"][key] for result in query_results.values()) / len(query_results)
        for key in keys
    }


def cluster_of(pair: PairRecord) -> str:
    """
    ブートストラップで独立とみなす単位。

    同じ画像から作られたクエリ (既定で最大2件) も、同じ動画の別フレームから作られたクエリも、
    ほぼ同じ景色を見ているので互いに独立ではない。クエリ単位で復元抽出すると
    この相関を無視して信頼区間が実際より狭くなる。動画グループを単位にする。
    """

    source = (pair.raw or {}).get("source_image_id")
    if not source:
        # 旧形式のデータセット (source_image_id なし) ではクエリ単位へ退避する。
        return pair.query_id
    return source.split(":")[-1].split("-")[0]


def paired_bootstrap(
    base_results: dict[str, dict[str, Any]],
    tuned_results: dict[str, dict[str, Any]],
    metric: str,
    *,
    clusters: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """動画グループを復元抽出して、指標差の 95% 信頼区間と両側 p 値を求める。"""

    query_ids = sorted(set(base_results) & set(tuned_results))
    if not query_ids:
        return {}

    deltas = [
        tuned_results[query_id]["metrics"][metric] - base_results[query_id]["metrics"][metric]
        for query_id in query_ids
    ]
    observed = sum(deltas) / len(deltas)

    # クラスタごとに delta をまとめ、クラスタ単位で復元抽出する (cluster bootstrap)。
    by_cluster: dict[str, list[float]] = {}
    for query_id, delta in zip(query_ids, deltas):
        by_cluster.setdefault(clusters.get(query_id, query_id), []).append(delta)
    cluster_deltas = [by_cluster[key] for key in sorted(by_cluster)]

    rng = random.Random(seed)
    count = len(cluster_deltas)
    samples: list[float] = []
    for _ in range(iterations):
        total = 0.0
        size = 0
        for _ in range(count):
            group = cluster_deltas[rng.randrange(count)]
            total += sum(group)
            size += len(group)
        samples.append(total / size if size else 0.0)

    samples.sort()
    lower = samples[int(0.025 * iterations)]
    upper = samples[min(int(0.975 * iterations), iterations - 1)]

    # 帰無仮説 (差が0) のもとでの両側 p 値をブートストラップ分布から近似する。
    centered = [sample - observed for sample in samples]
    extreme = sum(1 for value in centered if abs(value) >= abs(observed))
    p_value = min(1.0, (extreme + 1) / (iterations + 1))

    improved = sum(1 for delta in deltas if delta > 0)
    worsened = sum(1 for delta in deltas if delta < 0)

    return {
        "delta": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "clusters": count,
        "queries_improved": improved,
        "queries_worsened": worsened,
        "queries_unchanged": len(deltas) - improved - worsened,
    }


# ---------------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------------


def format_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3-VL-Reranker-8B ファインチューニング前後の比較")
    lines.append("")
    lines.append(f"- 生成日時: {payload['created_at']}")
    lines.append(f"- データセット: `{payload['dataset_dir']}` ({payload['split']} split)")
    lines.append(f"- ベースモデル: `{payload['model']}` ({payload['quantization']})")
    adapter = payload.get("adapter_path")
    lines.append(f"- アダプタ: `{adapter}`" if adapter else "- アダプタ: なし (base のみ)")
    lines.append(
        f"- 評価規模: クエリ {payload['num_queries']} 件 / ペア {payload['num_pairs']} 件"
        f" / 動画グループ {payload.get('num_clusters', '-')} 件"
    )
    lines.append("")
    lines.append(
        "> この評価が測るのは、教師シーンカードの定義に対する適合度である。"
        "候補集合は test split 内から構造化事実で選んだものなので、"
        "本番検索 (10万枚の初段検索の上位50件を再ランキング) の精度とは一致しない。"
    )
    lines.append("")

    variants = payload["variants"]
    has_tuned = "adapter" in variants

    lines.append("## ランキング精度 (クエリ単位平均)")
    lines.append("")
    header = "| 指標 | base |" + (" adapter | 差分 | 95%CI | p |" if has_tuned else "")
    separator = "| --- | ---: |" + (" ---: | ---: | :---: | ---: |" if has_tuned else "")
    lines.append(header)
    lines.append(separator)

    comparison = payload.get("comparison", {})
    for metric in [
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "mrr",
        "map",
        "recall@1",
        "recall@3",
        "recall@5",
    ]:
        base_value = variants["base"]["ranking"].get(metric)
        if base_value is None:
            continue
        row = f"| {metric} | {base_value:.4f} |"
        if has_tuned:
            tuned_value = variants["adapter"]["ranking"][metric]
            stats = comparison.get(metric, {})
            delta = stats.get("delta", tuned_value - base_value)
            ci = (
                f"[{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]"
                if "ci_lower" in stats
                else "-"
            )
            p_value = f"{stats['p_value']:.4f}" if "p_value" in stats else "-"
            row += f" {tuned_value:.4f} | {delta:+.4f} | {ci} | {p_value} |"
        lines.append(row)
    lines.append("")

    lines.append("## 二値識別精度 (ペア単位)")
    lines.append("")
    lines.append("| 指標 | base |" + (" adapter | 差分 |" if has_tuned else ""))
    lines.append("| --- | ---: |" + (" ---: | ---: |" if has_tuned else ""))
    for metric in ["roc_auc", "pr_auc", "accuracy@0", "precision@0", "recall@0"]:
        base_value = variants["base"]["binary"].get(metric)
        if base_value is None:
            continue
        row = f"| {metric} | {base_value:.4f} |"
        if has_tuned:
            tuned_value = variants["adapter"]["binary"][metric]
            row += f" {tuned_value:.4f} | {tuned_value - base_value:+.4f} |"
        lines.append(row)
    lines.append("")

    lines.append("## hard negative の混入")
    lines.append("")
    lines.append("正例より高いスコアが付いた hard negative の割合 (低いほど良い)。")
    lines.append("")
    lines.append("| 指標 | base |" + (" adapter | 差分 |" if has_tuned else ""))
    lines.append("| --- | ---: |" + (" ---: | ---: |" if has_tuned else ""))
    base_value = variants["base"]["hard_negative_error_rate"]
    row = f"| hard_negative_error_rate | {base_value:.4f} |"
    if has_tuned:
        tuned_value = variants["adapter"]["hard_negative_error_rate"]
        row += f" {tuned_value:.4f} | {tuned_value - base_value:+.4f} |"
    lines.append(row)
    lines.append("")

    if has_tuned:
        stats = comparison.get("ndcg@5", {})
        if stats:
            lines.append("## 判定")
            lines.append("")
            significant = stats["ci_lower"] > 0 or stats["ci_upper"] < 0
            if stats["delta"] > 0:
                direction = f"{stats['delta']:+.4f} の改善"
            elif stats["delta"] < 0:
                direction = f"{stats['delta']:+.4f} の劣化"
            else:
                direction = "変化なし"
            verdict = (
                "この差は統計的に有意です。"
                if significant
                else "この差は統計的に有意ではありません。"
            )
            lines.append(
                f"nDCG@5 は {direction}でした。"
                f"95% 信頼区間は [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}] で、{verdict}"
            )
            lines.append(
                f"クエリ単位では 改善 {stats['queries_improved']} 件 / "
                f"劣化 {stats['queries_worsened']} 件 / 変化なし {stats['queries_unchanged']} 件でした。"
            )
            lines.append(
                f"信頼区間は動画グループ {stats.get('clusters', '-')} 件を単位とした"
                "クラスタ・ブートストラップで求めた。"
            )
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Reranker-8B の学習前後精度を定量比較する。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--no-adapter",
        dest="use_adapter",
        action="store_false",
        default=True,
        help="学習前のベースラインだけを測る。",
    )
    parser.add_argument("--model", default=RERANKER_MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--quantization",
        choices=["8bit", "4bit", "none"],
        default="8bit",
        help="本番検索と同じ 8bit が既定。base と adapter は必ず同じ設定で測る。",
    )
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=0, help="0 なら全件。動作確認用。")
    parser.add_argument("--use-caption", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None, help="既定はデータセット配下の reports/")
    parser.add_argument("--tag", default=None, help="レポートファイル名につける識別子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    pairs_path = dataset_dir / f"pairs.{args.split}.jsonl"
    pairs = load_pairs(pairs_path)
    if not pairs:
        raise RuntimeError(f"{pairs_path} が空です。先に build_reranker_dataset.py を実行してください。")

    if args.max_queries > 0:
        selected = sorted({pair.query_id for pair in pairs})[: args.max_queries]
        keep = set(selected)
        pairs = [pair for pair in pairs if pair.query_id in keep]

    query_count = len({pair.query_id for pair in pairs})
    print(f"{args.split}: クエリ {query_count} 件 / ペア {len(pairs)} 件")

    adapter_path = (PROJECT_ROOT / args.adapter_path).resolve() if args.use_adapter else None
    if adapter_path is not None and not (adapter_path / "adapter_config.json").exists():
        hint = ""
        if (adapter_path / "last" / "adapter_config.json").exists():
            hint = (
                f" 学習で val がベースラインを超えなかった場合、best は保存されません。"
                f" 最終エポックのアダプタを評価するなら --adapter-path {args.adapter_path}/last を指定してください。"
            )
        raise RuntimeError(
            f"{adapter_path} に adapter_config.json がありません。"
            " 学習前のベースラインだけを測る場合は --no-adapter を指定してください。" + hint
        )

    print(f"モデルをロードします: {args.model} ({args.quantization})")
    model = load_reranker(
        model_name=args.model,
        device=args.device,
        quantization=args.quantization,
        max_pixels=args.max_pixels,
    )

    if adapter_path is not None:
        attach_adapter(model, adapter_path, adapter_name=ADAPTER_NAME)
        print(f"アダプタをロードしました: {adapter_path}")

    def run_variant(name: str, prepare: Callable[[], None]) -> dict[str, Any]:
        prepare()
        started_at = time.monotonic()
        scores = score_pairs(
            model,
            pairs,
            PROJECT_ROOT,
            batch_size=args.batch_size,
            use_caption=args.use_caption,
        )
        elapsed = time.monotonic() - started_at
        print(f"{name}: {len(pairs)} ペアを {elapsed / 60:.1f} 分で採点しました。")

        query_results = rank_pairs(pairs, scores)
        return {
            "scores": scores,
            "query_results": query_results,
            "ranking": aggregate(query_results),
            "binary": binary_metrics(pairs, scores),
            "hard_negative_error_rate": hard_negative_error_rate(pairs, scores),
            "scored_queries": len(query_results),
            "elapsed_seconds": round(elapsed, 1),
        }

    variants: dict[str, dict[str, Any]] = {}

    def disable() -> None:
        if adapter_path is not None:
            model.disable_adapters()

    variants["base"] = run_variant("base", disable)

    if adapter_path is not None:
        variants["adapter"] = run_variant("adapter", model.enable_adapters)

    clusters = {pair.query_id: cluster_of(pair) for pair in pairs}
    num_clusters = len(set(clusters.values()))

    comparison: dict[str, dict[str, float]] = {}
    if "adapter" in variants:
        for metric in variants["base"]["ranking"]:
            if metric == "candidates":
                continue
            comparison[metric] = paired_bootstrap(
                variants["base"]["query_results"],
                variants["adapter"]["query_results"],
                metric,
                clusters=clusters,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            )

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        (PROJECT_ROOT / args.output_dir).resolve()
        if args.output_dir
        else dataset_dir / "reports"
    )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": relative_to_project(dataset_dir, PROJECT_ROOT),
        "split": args.split,
        "model": args.model,
        "quantization": args.quantization,
        "max_pixels": args.max_pixels,
        "use_caption": args.use_caption,
        "instruction": DASHCAM_RERANKER_PROMPT,
        "adapter_path": relative_to_project(adapter_path, PROJECT_ROOT) if adapter_path else None,
        "num_queries": query_count,
        "num_pairs": len(pairs),
        "num_clusters": num_clusters,
        "bootstrap": {
            "unit": "video_group",
            "clusters": num_clusters,
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
        },
        "variants": {
            name: {
                key: value
                for key, value in variant.items()
                if key not in ("scores", "query_results")
            }
            for name, variant in variants.items()
        },
        "comparison": comparison,
    }

    write_json(output_dir / f"eval_{args.split}_{tag}.json", payload)

    markdown = format_markdown(payload)
    markdown_path = output_dir / f"eval_{args.split}_{tag}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    # 個別ペアのスコアは、誤りの目視確認に使えるよう別ファイルへ残す。
    score_records = []
    for index, pair in enumerate(pairs):
        record = {
            "pair_id": pair.pair_id,
            "query_id": pair.query_id,
            "query_text": pair.query_text,
            "image_path": pair.image_path,
            "label": pair.label,
            "negative_type": pair.negative_type,
            "score_base": variants["base"]["scores"][index],
        }
        if "adapter" in variants:
            record["score_adapter"] = variants["adapter"]["scores"][index]
        score_records.append(record)
    write_jsonl(output_dir / f"scores_{args.split}_{tag}.jsonl", score_records)

    print()
    print(markdown)
    print(f"レポート: {markdown_path}")


if __name__ == "__main__":
    main()
