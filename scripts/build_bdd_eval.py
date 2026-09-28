"""Build a test set whose labels come from the official BDD100K human annotations.

The teacher model is not used anywhere. BDD labels are converted into the same three-valued
fact format as the v3 dataset, and the catalog conditions are judged by the same evaluator
(condition_data.evaluate). Conditions that BDD cannot decide (colour, orientation, lane, van,
bridge, ...) evaluate to unknown and drop out on their own.

Candidates follow the production setting: each query retrieves the 100k index with the
first-stage embedding model and keeps the top candidates that were never used by v1-v3.

  - Objects whose box is lower than MIN_BOX_HEIGHT px are too small to decide. Every condition
    is judged twice, with all boxes and with only large boxes; if the two disagree the label
    is unknown.
  - Unknown candidates are dropped, so each query ranks only decided candidates.
  - The query is the first test-only paraphrase of the condition, which no adapter has seen.
    Conditions without paraphrases (X01-X04) use the catalog query and are marked as such.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

try:
    from .build_condition_dataset import MODEL, QUERY_PROMPT, REVISION, load_index, now
    from .condition_data import (
        INVENTORY_KINDS, ROOT, SCENES, digest, evaluate, load_conditions, read_jsonl, stable_key, write_json, write_jsonl,
    )
except ImportError:
    from build_condition_dataset import MODEL, QUERY_PROMPT, REVISION, load_index, now
    from condition_data import (
        INVENTORY_KINDS, ROOT, SCENES, digest, evaluate, load_conditions, read_jsonl, stable_key, write_json, write_jsonl,
    )

LABELS = Path.home()/'Downloads/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{}.json'
OUT = ROOT/'datasets/dashcam_reranker_bdd_eval'
PHRASINGS = ROOT/'datasets/dashcam_reranker_v3_paraphrase/derived/phrasings.jsonl'
USED_DATASETS = ('dashcam_reranker_ft_v1', 'dashcam_reranker_ft_v2_qwen38', 'dashcam_reranker_v3_conditions')
MIN_BOX_HEIGHT = 20
TOP_K = 50
SEARCH_DEPTH = 1000
DECIDABILITY_SAMPLE = 10000
SEED = 20260927
KIND = {'car': 'car', 'bus': 'bus', 'truck': 'truck', 'person': 'pedestrian', 'bike': 'bicycle',
        'motor': 'motorcycle', 'train': 'train'}  # rider is not a pedestrian, as in the catalog
BDD_KINDS = set(KIND.values())
# BDD gives one exclusive label per attribute, so a label is used as evidence against a
# condition only where the two cannot co-occur. The catalog lets 市街地 and 住宅街 overlap, and a
# tunnel on a highway or city street is labelled with the road type, so those stay unknown.
SCENE = {
    'urban': {'city street': 'yes', 'highway': 'no'},
    'residential': {'residential': 'yes', 'highway': 'no'},
    'highway': {'highway': 'yes', 'city street': 'no', 'residential': 'no', 'parking lot': 'no', 'gas stations': 'no'},
    'tunnel': {'tunnel': 'yes'},
}
TIME = {'day': 'daytime', 'night': 'night', 'twilight': 'dawn/dusk'}
# Annotators rarely chose foggy and often labelled rain seen through a wet windscreen as
# overcast, so only a clear (or partly cloudy) sky rules out rain and fog. BDD snowy marks snow
# lying on the ground rather than falling snow, so snowfall (and the road surface) stay unknown.
WEATHER = {
    'clear': {'clear': 'yes', 'overcast': 'no', 'rainy': 'no', 'snowy': 'no', 'foggy': 'no'},
    'overcast': {'overcast': 'yes', 'clear': 'no'},
    'rain': {'rainy': 'yes', 'clear': 'no', 'partly cloudy': 'no'},
    'fog': {'foggy': 'yes', 'clear': 'no', 'partly cloudy': 'no'},
}


def to_facts(record, width, height, *, large_only):
    """BDD annotation -> v3 fact dict. Only what BDD annotates is decided; the rest is unknown."""
    a = record['attributes']
    scene = dict.fromkeys(SCENES, 'unknown')
    for name, table in SCENE.items(): scene[name] = table.get(a['scene'], 'unknown')
    if a['timeofday'] != 'undefined':
        for name, value in TIME.items(): scene[name] = 'yes' if a['timeofday'] == value else 'no'
    for name, table in WEATHER.items(): scene[name] = table.get(a['weather'], 'unknown')
    labels = record['labels']
    lights = {l['attributes'].get('trafficLightColor') for l in labels if l['category'] == 'traffic light'}
    for color in ('red', 'green', 'yellow'): scene[f'{color}_signal'] = 'yes' if color in lights else 'no'
    lanes = [l for l in labels if l['category'] == 'lane']
    if lanes: scene['crosswalk'] = 'yes' if any(l['attributes'].get('laneType') == 'crosswalk' for l in lanes) else 'no'

    objects = []
    for l in labels:
        if l['category'] not in KIND: continue
        b = l['box2d']
        if large_only and b['y2'] - b['y1'] < MIN_BOX_HEIGHT: continue
        objects.append({
            'id': f"o{l['id']}", 'kind': KIND[l['category']],
            'bbox': [b['x1'] * 1000 / width, b['y1'] * 1000 / height, b['x2'] * 1000 / width, b['y2'] * 1000 / height],
            'color': 'unknown', 'orientation': 'unknown', 'lane': 'unknown', 'sidewalk': 'unknown',
            'on_crosswalk': 'unknown', 'roadway': 'unknown', 'emergency_vehicle': 'unknown',
        })
    # BDD boxes every instance of its classes; other kinds were not annotated.
    present = {o['kind'] for o in objects}
    inventory = {k: ({'presence': 'yes' if k in present else 'no', 'complete': True} if k in BDD_KINDS
                     else {'presence': 'unknown', 'complete': False}) for k in INVENTORY_KINDS}
    return {'scene': scene, 'objects': objects, 'inventory': inventory}


def both_facts(record, image):
    return tuple(to_facts(record, image['width'], image['height'], large_only=v) for v in (False, True))


def judge(expr, facts):
    """Decided only when the answer does not depend on the small boxes."""
    full, large = (evaluate(expr, f) for f in facts)
    return full if full == large else 'unknown'


def load_pool():
    """Annotated images never used by v1-v3 (exact duplicates of used images are excluded too)."""
    images = {r['image_id']: r for r in read_jsonl(ROOT/'datasets/dashcam_reranker_v3_conditions/inventory/images.jsonl')}
    used = set()
    for name in USED_DATASETS:
        for split in ('train', 'val', 'test'):
            for p in read_jsonl(ROOT/'datasets'/name/f'pairs.{split}.jsonl'):
                used.add(images[Path(p['image_path']).stem]['sha256'])
    records = {}
    for split in ('train', 'val'):
        for r in json.loads(Path(str(LABELS).format(split)).read_text()):
            image = images.get(Path(r['name']).stem)
            if image and image['error'] is None and image['sha256'] not in used: records[image['image_id']] = r
    return images, records


def load_phrasings(args):
    """{condition_id: {'heldout': [...]}} from the v3 split, or all verified texts of a generation log."""
    if not args.queries: return {r['condition_id']: r for r in read_jsonl(PHRASINGS)}
    return {r['condition_id']: {'heldout': sorted((q['text'] for q in r['queries'] if q['source'] == 'generated' and q['passed']),
                                                  key=lambda text: stable_key(SEED, f"{r['condition_id']}:{text}"))}
            for r in read_jsonl(args.queries)}


def load_queries(conditions, phrasings):
    queries = []
    for c in conditions:
        heldout = phrasings.get(c['id'], {}).get('heldout', [])
        queries.append({'condition': c, 'text': heldout[0] if heldout else c['query'],
                        'phrasing_role': 'heldout' if heldout else 'catalog'})
    return queries


def retrieve(texts, device):
    import torch
    from sentence_transformers import SentenceTransformer
    index, paths = load_index()
    model = SentenceTransformer(MODEL, revision=REVISION, local_files_only=True, device=device,
                                model_kwargs={'torch_dtype': torch.bfloat16})
    vectors = model.encode(texts, prompt=QUERY_PROMPT, batch_size=8, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype('float32')
    scores, ids = index.search(vectors, SEARCH_DEPTH)
    return [[(Path(paths[i]).stem, float(s)) for s, i in zip(row_s, row_i) if i >= 0] for row_s, row_i in zip(scores, ids)]


def run(args):
    started = time.monotonic()
    # Extra conditions (build_combo_conditions.py) replace the catalog when given.
    conditions = read_jsonl(args.conditions) if args.conditions else [c for c in load_conditions() if c['expression']['op'] != 'deferred']
    images, records = load_pool()
    print(f'pool: {len(records)} annotated images never used by v1-v3', flush=True)

    # Keep only conditions BDD can decide, before spending retrieval on them. A fixed sample of
    # the pool is enough to see whether a condition ever evaluates to yes and to no.
    sample = [both_facts(records[i], images[i]) for i in sorted(records, key=lambda i: stable_key(SEED, i))[:DECIDABILITY_SAMPLE]]
    decided = {c['id']: dict(Counter(judge(c['expression'], f) for f in sample)) for c in conditions}
    usable = [c for c in conditions if decided[c['id']].get('yes', 0) and decided[c['id']].get('no', 0)]
    print(f'conditions decidable from BDD labels: {len(usable)}/{len(conditions)}', flush=True)

    phrasings = load_phrasings(args)
    if args.queries:
        # Extra conditions have no natural reference query to fall back on.
        usable = [c for c in usable if phrasings.get(c['id'], {}).get('heldout')]
    queries = load_queries(usable, phrasings)
    rankings = retrieve([q['text'] for q in queries], args.device)
    pairs, per_query = [], []
    for q, hits in zip(queries, rankings):
        c = q['condition']; qid = f"condition:test:{c['id']}:{'h0' if q['phrasing_role'] == 'heldout' else 'c'}"
        candidates = [(rank, image_id, score) for rank, (image_id, score) in enumerate(hits, 1) if image_id in records][:TOP_K]
        labels = Counter()
        rows = []
        for rank, image_id, score in candidates:
            image = images[image_id]
            label = judge(c['expression'], both_facts(records[image_id], image))
            labels[label] += 1
            if label == 'unknown': continue
            rows.append({
                'pair_id': f"bdd:{c['id']}:{image_id}", 'query_id': qid, 'condition_id': c['id'],
                'query_text': q['text'], 'catalog_query_text': c['query'], 'phrasing_role': q['phrasing_role'],
                'image_id': image_id, 'image_path': image['image_path'], 'label': int(label == 'yes'),
                'negative_type': 'positive' if label == 'yes' else 'hard_negative', 'split': 'test', 'caption': None,
                'retrieval_rank': rank, 'retrieval_score': score, 'label_source': 'bdd100k_official_labels',
            })
        rankable = any(r['label'] for r in rows) and not all(r['label'] for r in rows)
        per_query.append({'condition_id': c['id'], 'query_text': q['text'], 'phrasing_role': q['phrasing_role'],
                          'candidates': len(candidates), 'labels': dict(labels), 'rankable': rankable})
        if rankable: pairs += rows

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out/'pairs.test.jsonl', pairs)
    write_json(args.out/'reports/build_stats.json', {
        'created_at': now(), 'labels': {s: digest(Path(str(LABELS).format(s))) for s in ('train', 'val')},
        'phrasings_sha256': digest(args.queries or PHRASINGS), 'catalog_sha256': digest(ROOT/'docs/search-condition-catalog.md'),
        'conditions_sha256': digest(args.conditions) if args.conditions else None,
        'index_sha256': digest(ROOT/'data_100k/images.faiss'), 'embedding_model': MODEL, 'embedding_revision': REVISION,
        'min_box_height': MIN_BOX_HEIGHT, 'top_k': TOP_K, 'excluded_datasets': USED_DATASETS,
        'pool_images': len(records), 'conditions_total': len(conditions), 'conditions_decidable': len(usable),
        'queries_rankable': sum(q['rankable'] for q in per_query), 'pairs': len(pairs),
        'positive_pairs': sum(p['label'] for p in pairs), 'decidability_sample': len(sample), 'sample_label_counts': decided, 'queries': per_query,
        'elapsed_seconds': time.monotonic() - started,
    })
    print(f"queries {len(per_query)} (rankable {sum(q['rankable'] for q in per_query)}), pairs {len(pairs)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conditions', type=Path, help='conditions JSONL used instead of the catalog')
    parser.add_argument('--queries', type=Path, help='generate_condition_queries.py log used instead of the v3 test-only paraphrases')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
