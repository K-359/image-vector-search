"""Compare what the training queries ask for: v2 (free generation) and v3 (catalog conditions).

v2 queries were written freely by the teacher for 1,000 images; each query records the facts it
constrains (`supported_facts`). v3 queries come from catalog conditions; the object kinds are
read from each condition's expression. The table shows how concentrated each set is on cars and
on a few facts, which was the original motivation for v3.

    python scripts/compare_query_distribution.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from .condition_data import ROOT, load_conditions, read_jsonl
except ImportError:
    from condition_data import ROOT, load_conditions, read_jsonl

KINDS = ['car', 'truck', 'bus', 'van', 'pedestrian', 'bicycle', 'motorcycle']
KIND_JA = {'car': '車', 'truck': 'トラック', 'bus': 'バス', 'van': 'バン', 'pedestrian': '歩行者',
           'bicycle': '自転車', 'motorcycle': 'バイク'}


def queries(path):
    """{query_id: first pair} and the number of distinct images in a pairs file."""
    rows = read_jsonl(path)
    return {r['query_id']: r for r in reversed(rows)}, len({r['image_id'] for r in rows})


def v2_kinds(pair):
    return set(pair['supported_facts']) & set(KINDS)


def v3_kinds(expression):
    text = json.dumps(expression)
    return {k for k in KINDS if f'"kind": "{k}"' in text}


def summary(name, qs, images, kinds, top_facts):
    n = len(qs)
    share = Counter(k for q in qs for k in kinds[q])
    return {'name': name, 'queries': n, 'images': images,
            'kinds': {k: share[k]/n for k in KINDS},
            'car_only': sum(kinds[q] == {'car'} for q in qs)/n, 'no_object': sum(not kinds[q] for q in qs)/n,
            'top_facts': top_facts}


def run(args):
    v2, v2_images = queries(args.v2/'pairs.train.jsonl')
    facts = Counter(f for p in v2.values() for f in p['supported_facts'])
    v2_row = summary('v2', v2, v2_images, {q: v2_kinds(p) for q, p in v2.items()},
                     [(f, c/len(v2)) for f, c in facts.most_common(5)])
    expressions = {c['id']: c['expression'] for c in load_conditions()}
    v3, v3_images = queries(args.v3/'pairs.train.jsonl')
    conditions = Counter(p['condition_id'] for p in v3.values())
    v3_row = summary('v3', v3, v3_images, {q: v3_kinds(expressions[p['condition_id']]) for q, p in v3.items()},
                     [(c, n/len(v3)) for c, n in conditions.most_common(5)])
    v3_row['conditions'] = len(conditions)
    rows = [v2_row, v3_row]
    print('| | ' + ' | '.join(r['name'] for r in rows) + ' |\n|---|' + '---:|'*len(rows))
    print('| 学習クエリ | ' + ' | '.join(str(r['queries']) for r in rows) + ' |')
    print('| 学習画像 | ' + ' | '.join(str(r['images']) for r in rows) + ' |')
    for k in KINDS:
        print(f'| {KIND_JA[k]}を含む | ' + ' | '.join(f"{r['kinds'][k]:.1%}" for r in rows) + ' |')
    print('| 対象が車だけ | ' + ' | '.join(f"{r['car_only']:.1%}" for r in rows) + ' |')
    print('| 対象なし（場面だけ） | ' + ' | '.join(f"{r['no_object']:.1%}" for r in rows) + ' |')
    print('\nv2 で多い事実:', ', '.join(f'{f} {s:.0%}' for f, s in v2_row['top_facts']))
    print(f"v3 の条件数: {v3_row['conditions']}、1条件あたり最大 {v3_row['top_facts'][0][1]:.1%} のクエリ")
    if args.json: args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--v2', type=Path, default=ROOT/'datasets/dashcam_reranker_ft_v2_qwen38')
    parser.add_argument('--v3', type=Path, default=ROOT/'datasets/dashcam_reranker_v3_paraphrase')
    parser.add_argument('--json', type=Path, help='also write the numbers here')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
