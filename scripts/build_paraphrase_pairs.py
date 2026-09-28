"""Rewrite the v3 condition pairs with verified paraphrased queries.

Images, labels and splits are copied from the v3 condition pairs unchanged; only the query
text changes. Each condition's verified paraphrases are split once into a training pool
(plus the catalog query) and test-only phrasings:

  train: every pair gets one phrasing from the training pool, so the pair count is unchanged.
  val:   one phrasing from the training pool per condition, with that condition's full candidates.
  test:  every test-only phrasing is scored against that condition's full candidate set.

The catalog-query test set stays in the v3 dataset, so seen and unseen phrasings are
evaluated on identical images and labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

try:
    from .condition_data import ROOT, digest, read_jsonl, stable_key, write_json, write_jsonl
except ImportError:
    from condition_data import ROOT, digest, read_jsonl, stable_key, write_json, write_jsonl

SOURCE = ROOT/'datasets/dashcam_reranker_v3_conditions'
OUT = ROOT/'datasets/dashcam_reranker_v3_paraphrase'
HELDOUT_FRACTION = 0.3
MAX_HELDOUT = 3
# Words the parser treats loosely: 車両 spans car/van/truck/bus, 見つけ is a request.
LOOSE_WORDS = re.compile(r'(?<!緊急)車両|見つけ')


def normalize(text):
    return re.sub(r'[\s、。,.]+', '', text)


def assign_phrasings(rows, *, seed):
    """{condition_id: {'train': [...], 'heldout': [...]}}; the catalog query is always in train."""
    owners = defaultdict(set)
    for r in rows:
        for q in r['queries']:
            if q['passed']: owners[normalize(q['text'])].add(r['condition_id'])
    phrasings, dropped = {}, Counter()
    for r in rows:
        catalog = next(q['text'] for q in r['queries'] if q['source'] == 'catalog')
        generated = []
        for q in r['queries']:
            if q['source'] != 'generated' or not q['passed']: continue
            if LOOSE_WORDS.search(q['text']): dropped['loose_words'] += 1; continue
            if len(owners[normalize(q['text'])]) > 1: dropped['shared_across_conditions'] += 1; continue
            generated.append(q['text'])
        generated.sort(key=lambda text: stable_key(seed, f"{r['condition_id']}:{text}"))
        count = min(MAX_HELDOUT, max(1, math.ceil(len(generated) * HELDOUT_FRACTION))) if generated else 0
        phrasings[r['condition_id']] = {'train': [catalog] + generated[count:], 'heldout': generated[:count]}
    return phrasings, dropped


def rewrite(pair, query_id, text, phrasing_role, suffix=''):
    return {**pair, 'pair_id': pair['pair_id'] + suffix, 'query_id': query_id, 'query_text': text,
            'catalog_query_text': pair['query_text'], 'phrasing_role': phrasing_role}


def run(args):
    rows = read_jsonl(args.queries)
    config = json.loads((args.queries.parent/'config.json').read_text())
    phrasings, dropped = assign_phrasings(rows, seed=args.seed)
    source = {s: read_jsonl(args.source/f'pairs.{s}.jsonl') for s in ('train', 'val', 'test')}
    by_condition = {s: defaultdict(list) for s in source}
    for s, pairs in source.items():
        for p in pairs: by_condition[s][p['condition_id']].append(p)

    out = {s: [] for s in source}
    missing = Counter()
    for s in ('train', 'val'):
        for cid, pairs in sorted(by_condition[s].items()):
            # X conditions have no paraphrases and keep their catalog query.
            pool = phrasings.get(cid, {'train': [pairs[0]['query_text']]})['train']
            if cid not in phrasings: missing[s] += 1
            if s == 'train':
                for p in pairs:
                    k = int(stable_key(args.seed, p['pair_id']), 16) % len(pool)
                    out[s].append(rewrite(p, f'condition:{s}:{cid}:p{k}', pool[k], 'catalog' if k == 0 else 'train'))
            else:
                k = int(stable_key(args.seed, f'val:{cid}'), 16) % len(pool)
                out[s] += [rewrite(p, f'condition:{s}:{cid}:p{k}', pool[k], 'catalog' if k == 0 else 'train') for p in pairs]
    for cid, pairs in sorted(by_condition['test'].items()):
        heldout = phrasings.get(cid, {'heldout': []})['heldout']
        if not heldout: missing['test'] += 1
        for k, text in enumerate(heldout):
            out['test'] += [rewrite(p, f'condition:test:{cid}:h{k}', text, 'heldout', f':h{k}') for p in pairs]

    # Held-out phrasings must never appear as training or validation text.
    seen = {normalize(p['query_text']) for s in ('train', 'val') for p in out[s]}
    leaked = sorted({p['query_text'] for p in out['test'] if normalize(p['query_text']) in seen})
    if leaked: raise SystemExit(f'held-out phrasings leaked into train/val: {leaked[:5]}')
    for s in source:
        if s != 'test' and len(out[s]) != len(source[s]): raise SystemExit(f'{s} pair count changed')

    args.out.mkdir(parents=True, exist_ok=True)
    for s, pairs in out.items(): write_jsonl(args.out/f'pairs.{s}.jsonl', pairs)
    write_jsonl(args.out/'derived/phrasings.jsonl',
                [{'condition_id': cid, **v} for cid, v in sorted(phrasings.items())])
    stats = {
        'source_dataset': str(args.source.relative_to(ROOT)),
        'source_pair_sha256': {s: digest(args.source/f'pairs.{s}.jsonl') for s in source},
        'queries_sha256': digest(args.queries), 'query_config': config, 'seed': args.seed,
        'heldout_fraction': HELDOUT_FRACTION, 'max_heldout': MAX_HELDOUT, 'dropped_phrasings': dict(dropped),
        'conditions_without_paraphrases': dict(missing),
        'pairs': {s: len(p) for s, p in out.items()},
        'queries': {s: len({p['query_id'] for p in pairs}) for s, pairs in out.items()},
        'train_phrasing_roles': dict(Counter(p['phrasing_role'] for p in out['train'])),
        'train_distinct_texts': len({p['query_text'] for p in out['train']}),
        'heldout_per_condition': dict(Counter(len(v['heldout']) for v in phrasings.values())),
        'train_pool_per_condition': dict(Counter(len(v['train']) for v in phrasings.values())),
    }
    write_json(args.out/'reports/paraphrase_pair_stats.json', stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--queries', type=Path, default=OUT/'queries/queries.jsonl')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, default=20260926)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
