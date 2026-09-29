"""Build the v4 dataset: the v3 paraphrase data plus the extension catalog conditions.

The extension conditions (docs/search-condition-catalog-extension.md) are judged on the same
5,000 v3 images, from the v3 facts merged with the extension facts
(annotate_extension_facts.py). Everything after that reuses the v3 steps unchanged:

  retrieve  rank the 100k-image index for each extension query, as v3 did for hard negatives
  pairs     judge every image, then sample pairs with build_condition_pairs.build_pairs
  combine   rewrite the extension pairs with verified paraphrases
            (build_paraphrase_pairs.assign_phrasings) and add them to the v3 paraphrase pairs

Output (datasets/dashcam_reranker_v4_extension):
  pairs.train.jsonl / pairs.val.jsonl  v3 paraphrase pairs + extension pairs
  pairs.test.jsonl                     extension conditions only, test-only paraphrases
  catalog/pairs.*.jsonl                extension pairs with the catalog query

    python scripts/build_extension_dataset.py retrieve
    python scripts/build_extension_dataset.py pairs
    python scripts/build_extension_dataset.py combine
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

try:
    from .annotate_extension_facts import OUT, SOURCE, load_merged
    from .build_condition_dataset import MODEL, QUERY_PROMPT, REVISION, load_index, now
    from .build_condition_pairs import SPLITS, build_pairs, excluded_conditions
    from .build_paraphrase_pairs import assign_phrasings, normalize, rewrite
    from .condition_data import ROOT, digest, evaluate, load_extension_conditions, read_jsonl, stable_key, write_json, write_jsonl
except ImportError:
    from annotate_extension_facts import OUT, SOURCE, load_merged
    from build_condition_dataset import MODEL, QUERY_PROMPT, REVISION, load_index, now
    from build_condition_pairs import SPLITS, build_pairs, excluded_conditions
    from build_paraphrase_pairs import assign_phrasings, normalize, rewrite
    from condition_data import ROOT, digest, evaluate, load_extension_conditions, read_jsonl, stable_key, write_json, write_jsonl

PARAPHRASE = ROOT/'datasets/dashcam_reranker_v3_paraphrase'
SEARCH_DEPTH = 1000  # same depth as the v3 rankings (build_condition_dataset.run_retrieve)


def run_retrieve(args):
    import torch
    from sentence_transformers import SentenceTransformer
    conditions = load_extension_conditions()
    index, paths = load_index()
    model = SentenceTransformer(MODEL, revision=REVISION, local_files_only=True, device=args.device,
                                model_kwargs={'torch_dtype': torch.bfloat16})
    vectors = model.encode([c['query'] for c in conditions], prompt=QUERY_PROMPT, batch_size=8, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype('float32')
    scores, ids = index.search(vectors, SEARCH_DEPTH)
    rankings = [{'condition_id': c['id'], 'query': c['query'],
                 'hits': [{'index_id': int(i), 'image_id': Path(paths[i]).stem, 'rank': rank, 'score': float(s)}
                          for rank, (s, i) in enumerate(zip(cs, ci), 1) if i >= 0]}
                for c, cs, ci in zip(conditions, scores, ids)]
    write_jsonl(args.out/'selection/rankings.jsonl', rankings)
    write_json(args.out/'selection/retrieval.json', {
        'created_at': now(), 'model': MODEL, 'revision': REVISION, 'query_prompt': QUERY_PROMPT, 'depth': SEARCH_DEPTH,
        'device': args.device, 'index_sha256': digest(ROOT/'data_100k/images.faiss'),
        'catalog_sha256': digest(ROOT/'docs/search-condition-catalog-extension.md'), 'conditions': len(conditions)})
    print('retrieval complete', len(conditions))


def run_pairs(args):
    conditions = load_extension_conditions()
    manifest = read_jsonl(SOURCE/'manifests/selected_5000.jsonl')
    records = load_merged()
    excluded = excluded_conditions(conditions)
    judgments, buckets = [], defaultdict(lambda: defaultdict(list))
    for row in manifest:
        facts = records[row['image_id']]['facts']
        for c in conditions:
            value = evaluate(c['expression'], facts)
            usage = 'excluded' if excluded[(row['split'], c['id'])] else 'unknown' if value == 'unknown' else 'teacher_candidate'
            judgments.append({'image_id': row['image_id'], 'condition_id': c['id'], 'split': row['split'],
                              'judgment': value, 'usage': usage})
            if usage != 'excluded': buckets[(row['split'], c['id'])][value].append(row['image_id'])
    write_jsonl(args.out/'annotations/condition_judgments.jsonl', judgments)
    pairs, queries, coverage = build_pairs(manifest, conditions, buckets, read_jsonl(args.out/'selection/rankings.jsonl'),
                                           seed=args.seed)
    for split in SPLITS:
        for p in pairs[split]: p['label_source'] = 'v4_extension/annotations/condition_judgments.jsonl'
        write_jsonl(args.out/f'catalog/pairs.{split}.jsonl', pairs[split])
    write_jsonl(args.out/'catalog/queries.jsonl', queries)
    write_json(args.out/'reports/extension_pair_coverage.json', coverage)
    stats = {'created_at': now(), 'seed': args.seed,
             'sources_sha256': {'v3_facts': digest(SOURCE/'annotations/facts.jsonl'),
                                'extension_facts': digest(args.out/'annotations/extension_facts.jsonl'),
                                'manifest': digest(SOURCE/'manifests/selected_5000.jsonl'),
                                'catalog': digest(ROOT/'docs/search-condition-catalog-extension.md'),
                                'allocation': digest(ROOT/'docs/search-condition-allocation-extension.csv')},
             'judgments': {s: dict(Counter(j['judgment'] for j in judgments if j['split'] == s)) for s in SPLITS},
             'pairs': {s: len(pairs[s]) for s in SPLITS},
             'queries': {s: sum(q['split'] == s for q in queries) for s in SPLITS},
             'statuses': {s: dict(Counter(c['status'] for c in coverage if c['split'] == s)) for s in SPLITS}}
    write_json(args.out/'reports/extension_pair_stats.json', stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    for c in coverage:
        print(c['split'], c['condition_id'], c['status'], f"yes={c['available_positive']} no={c['available_negative']} unknown={c['unknown']}")


def run_combine(args):
    rows = read_jsonl(args.out/'queries/queries.jsonl')
    phrasings, dropped = assign_phrasings(rows, seed=args.paraphrase_seed)
    catalog = {s: read_jsonl(args.out/f'catalog/pairs.{s}.jsonl') for s in SPLITS}
    by_condition = {s: defaultdict(list) for s in SPLITS}
    for s, pairs in catalog.items():
        for p in pairs: by_condition[s][p['condition_id']].append(p)
    # Same rewriting as build_paraphrase_pairs.run, applied to the extension pairs.
    extension = {s: [] for s in SPLITS}
    for s in ('train', 'val'):
        for cid, pairs in sorted(by_condition[s].items()):
            pool = phrasings[cid]['train']
            if s == 'train':
                for p in pairs:
                    k = int(stable_key(args.paraphrase_seed, p['pair_id']), 16) % len(pool)
                    extension[s].append(rewrite(p, f'condition:{s}:{cid}:p{k}', pool[k], 'catalog' if k == 0 else 'train'))
            else:
                k = int(stable_key(args.paraphrase_seed, f'val:{cid}'), 16) % len(pool)
                extension[s] += [rewrite(p, f'condition:{s}:{cid}:p{k}', pool[k], 'catalog' if k == 0 else 'train') for p in pairs]
    for cid, pairs in sorted(by_condition['test'].items()):
        for k, text in enumerate(phrasings[cid]['heldout']):
            extension['test'] += [rewrite(p, f'condition:test:{cid}:h{k}', text, 'heldout', f':h{k}') for p in pairs]
    base = {s: read_jsonl(PARAPHRASE/f'pairs.{s}.jsonl') for s in ('train', 'val')}
    out = {'train': base['train'] + extension['train'], 'val': base['val'] + extension['val'], 'test': extension['test']}
    seen = {normalize(p['query_text']) for s in ('train', 'val') for p in out[s]}
    leaked = sorted({p['query_text'] for p in out['test'] if normalize(p['query_text']) in seen})
    if leaked: raise SystemExit(f'held-out phrasings leaked into train/val: {leaked[:5]}')
    images = {s: {p['image_id'] for p in out[s]} for s in SPLITS}
    if images['train'] & images['val'] or images['train'] & images['test'] or images['val'] & images['test']:
        raise SystemExit('image leakage across splits')
    for s, pairs in out.items(): write_jsonl(args.out/f'pairs.{s}.jsonl', pairs)
    write_jsonl(args.out/'derived/phrasings.jsonl', [{'condition_id': cid, **v} for cid, v in sorted(phrasings.items())])
    stats = {'created_at': now(), 'paraphrase_seed': args.paraphrase_seed, 'dropped_phrasings': dict(dropped),
             'base_pairs_sha256': {s: digest(PARAPHRASE/f'pairs.{s}.jsonl') for s in ('train', 'val')},
             'pairs': {s: len(p) for s, p in out.items()},
             'extension_pairs': {s: len(p) for s, p in extension.items()},
             'queries': {s: len({p['query_id'] for p in pairs}) for s, pairs in out.items()},
             'heldout_per_condition': {cid: len(v['heldout']) for cid, v in sorted(phrasings.items())}}
    write_json(args.out/'reports/combine_stats.json', stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['retrieve', 'pairs', 'combine'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cpu', help='the teacher usually holds the GPU while this runs')
    parser.add_argument('--seed', type=int, default=20260924, help='pair sampling seed (same as v3)')
    parser.add_argument('--paraphrase-seed', type=int, default=20260926, help='phrasing split seed (same as v3)')
    args = parser.parse_args()
    {'retrieve': run_retrieve, 'pairs': run_pairs, 'combine': run_combine}[args.command](args)


if __name__ == '__main__':
    main()
