"""Compare several adapters on one test set, per condition, with paired bootstrap intervals.

Each adapter was scored by evaluate_reranker.py on the same pairs (scores_*.jsonl). A variant is
given as NAME=SCORES_FILE:FIELD, where FIELD is score_base or score_adapter. As in
analyze_condition_eval.py, every phrasing is ranked on its own and the metrics are averaged per
condition (query_id is condition:<split>:<condition id>[:<phrasing>]); conditions are the
bootstrap unit.

    python scripts/compare_adapters.py \\
        base=datasets/dashcam_reranker_v4_extension/reports/scores_test_ext-v4.jsonl:score_base \\
        v3言換=datasets/dashcam_reranker_v4_extension/reports/scores_test_ext-v3paraphrase.jsonl:score_adapter \\
        v4=datasets/dashcam_reranker_v4_extension/reports/scores_test_ext-v4.jsonl:score_adapter \\
        --contrast base:v4 v3言換:v4
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_condition_eval import condition_metrics, mean  # noqa: E402
from evaluate_reranker import paired_bootstrap  # noqa: E402


def load(variants):
    """{condition_id: [rows per phrasing]} with one score_<name> field per variant."""
    rows = {}
    for name, (path, field) in variants.items():
        with path.open() as f:
            scores = {r['pair_id']: r for r in map(json.loads, f)}
        if rows and set(scores) != set(rows): raise SystemExit(f'{path} covers different pairs')
        for pair_id, r in scores.items():
            row = rows.setdefault(pair_id, {k: r[k] for k in ('pair_id', 'query_id', 'label', 'negative_type')})
            row[f'score_{name}'] = r[field]
    by_query = defaultdict(list)
    for row in rows.values(): by_query[row['query_id']].append(row)
    by_condition = defaultdict(list)
    for query_id, qrows in by_query.items():
        if any(r['label'] for r in qrows) and not all(r['label'] for r in qrows):
            by_condition[query_id.split(':')[2]].append(qrows)
    return by_condition


def run(args):
    variants = {}
    for spec in args.variants:
        name, rest = spec.split('=', 1)
        path, field = rest.rsplit(':', 1)
        variants[name] = (Path(path), field)
    by_condition = load(variants)
    results = {v: {cid: {'metrics': {k: mean([m[k] for m in ms]) for k in ms[0]}}
                   for cid, queries in by_condition.items()
                   for ms in [[condition_metrics(q, v) for q in queries]]}
               for v in variants}
    groups = {'全体': sorted(by_condition)}
    for cid in sorted(by_condition):
        groups.setdefault(re.match(r'[A-Z]+', cid).group(0), []).append(cid)
    for pattern in args.group or []:
        name, regex = pattern.split('=', 1)
        groups[name] = [c for c in sorted(by_condition) if re.fullmatch(regex, c)]
    contrasts = [c.split(':') for c in args.contrast]
    for metric in args.metrics:
        print(f'\n## {metric}\n\n| 区分 | 条件 | ' + ' | '.join(variants) + ' | '
              + ' | '.join(f'{b}−{a}' for a, b in contrasts) + ' |\n|---|' + '---:|'*(1 + len(variants)) + '---|'*len(contrasts))
        for group, ids in groups.items():
            if not ids: continue
            means = [mean([results[v][c]['metrics'][metric] for c in ids]) for v in variants]
            cells = []
            for a, b in contrasts:
                if len(ids) < 2: cells.append('—'); continue
                s = paired_bootstrap({c: results[a][c] for c in ids}, {c: results[b][c] for c in ids}, metric,
                                     clusters={}, iterations=args.iterations, seed=args.seed)
                cells.append(f"{s['delta']:+.3f} [{s['ci_lower']:+.3f}, {s['ci_upper']:+.3f}]")
            print(f'| {group} | {len(ids)} | ' + ' | '.join(f'{m:.3f}' for m in means) + ' | ' + ' | '.join(cells) + ' |')
    if args.per_condition:
        metric = args.metrics[0]
        print(f'\n## 条件別 {metric}\n\n| 条件 | ' + ' | '.join(variants) + ' |\n|---|' + '---:|'*len(variants))
        for cid in sorted(by_condition):
            print(f'| {cid} | ' + ' | '.join(f"{results[v][cid]['metrics'][metric]:.3f}" for v in variants) + ' |')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('variants', nargs='+', help='NAME=SCORES_FILE:FIELD')
    parser.add_argument('--contrast', nargs='+', default=[], help='A:B pairs; reports B minus A')
    parser.add_argument('--metrics', nargs='+', default=['ndcg@5', 'auc'])
    parser.add_argument('--group', nargs='*', help='NAME=REGEX over condition IDs, added to the category groups')
    parser.add_argument('--per-condition', action='store_true')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
