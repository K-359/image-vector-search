"""Compare base / v2 / v3 / v3 paraphrase on the 3-4 constraint BDD test by kind of combination.

Reads the scores that evaluate_reranker.py wrote for each adapter on
datasets/dashcam_reranker_bdd_combo (tags combo-v2, combo-v3, combo-v3paraphrase); the base
scores come from the combo-v2 run. Queries are grouped by the number of constraints, by whether
they name objects or positions, and by the scene terms that behaved differently (clear, night).
Confidence intervals are paired bootstraps over conditions.

    python scripts/analyze_combo_eval.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_condition_eval import condition_metrics  # noqa: E402
from condition_data import ROOT, read_jsonl  # noqa: E402
from evaluate_reranker import paired_bootstrap  # noqa: E402

# Variant name -> (score tag, score field).
VARIANTS = {'base': ('combo-v2', 'score_base'), 'v2': ('combo-v2', 'score_adapter'),
            'v3': ('combo-v3', 'score_adapter'), 'v3言換': ('combo-v3paraphrase', 'score_adapter')}
CONTRASTS = [('base', 'v3言換'), ('v2', 'v3言換'), ('v3', 'v3言換')]


def terms(condition):
    e = condition['expression']
    return e['terms'] if e['op'] == 'all' else [e]


def objects(condition):
    return [o for t in terms(condition) if t['op'] == 'exists' for o in t['objects']]


def scenes(condition):
    return {t['name'] for t in terms(condition) if t['op'] == 'scene'}


GROUPS = {
    '全体': lambda c: True,
    '3条件': lambda c: c['constraints'] == 3,
    '4条件': lambda c: c['constraints'] == 4,
    '場面の条件だけ': lambda c: not objects(c),
    '対象が1つ': lambda c: len(objects(c)) == 1,
    '対象が2つ': lambda c: len(objects(c)) == 2,
    '位置を含む': lambda c: any('position' in o for o in objects(c)),
    '晴れを含む': lambda c: 'clear' in scenes(c),
    '晴れを含まない': lambda c: 'clear' not in scenes(c),
    '夜を含む': lambda c: 'night' in scenes(c),
}


def run(args):
    conditions = {c['id']: c for c in read_jsonl(args.dataset_dir/'conditions.jsonl')}
    scores = {tag: {r['pair_id']: r for r in read_jsonl(args.dataset_dir/f'reports/scores_test_{tag}.jsonl')}
              for tag in {t for t, _ in VARIANTS.values()}}
    pair_ids = set(scores['combo-v2'])
    if any(set(s) != pair_ids for s in scores.values()): raise SystemExit('score files cover different pairs')
    by_query = defaultdict(list)
    for pair_id in pair_ids:
        r = scores['combo-v2'][pair_id]
        row = {'pair_id': pair_id, 'label': r['label'], 'negative_type': r['negative_type']}
        for name, (tag, field) in VARIANTS.items(): row[f'score_{name}'] = scores[tag][pair_id][field]
        by_query[r['query_id']].append(row)
    results = {name: {q: {'metrics': condition_metrics(rows, name)} for q, rows in by_query.items()} for name in VARIANTS}
    condition_of = lambda q: conditions[q.split(':')[2]]
    for metric in args.metrics:
        print(f'\n## {metric}\n\n| 区分 | クエリ | ' + ' | '.join(VARIANTS) + ' | '
              + ' | '.join(f'{b}−{a}' for a, b in CONTRASTS) + ' |\n|---|' + '---:|'*(1 + len(VARIANTS)) + '---|'*len(CONTRASTS))
        for group, keep in GROUPS.items():
            qs = [q for q in by_query if keep(condition_of(q))]
            means = [sum(results[v][q]['metrics'][metric] for q in qs)/len(qs) for v in VARIANTS]
            cells = []
            for a, b in CONTRASTS:
                s = paired_bootstrap({q: results[a][q] for q in qs}, {q: results[b][q] for q in qs}, metric,
                                     clusters={}, iterations=args.iterations, seed=args.seed)
                cells.append(f"{s['delta']:+.3f} [{s['ci_lower']:+.3f}, {s['ci_upper']:+.3f}]")
            print(f'| {group} | {len(qs)} | ' + ' | '.join(f'{m:.3f}' for m in means) + ' | ' + ' | '.join(cells) + ' |')
    print(f'\n## v3言換 のクエリ内 AUC が低い条件\n\n| 条件 | 文 | ' + ' | '.join(VARIANTS) + ' |\n|---|---|' + '---:|'*len(VARIANTS))
    for q in sorted(by_query, key=lambda q: results['v3言換'][q]['metrics']['auc'])[:args.worst]:
        c = condition_of(q)
        print(f"| {c['id']} | {c['query']} | " + ' | '.join(f"{results[v][q]['metrics']['auc']:.2f}" for v in VARIANTS) + ' |')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT/'datasets/dashcam_reranker_bdd_combo')
    parser.add_argument('--metrics', nargs='+', default=['ndcg@10', 'auc'])
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--worst', type=int, default=8)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
