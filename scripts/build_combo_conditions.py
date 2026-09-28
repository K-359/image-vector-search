"""Sample 3-4 constraint conditions that BDD100K labels can decide, for testing longer queries.

The catalog conditions have one to three constraints. To see whether adapters trained on them
handle longer queries, this script combines constraints that the BDD conversion in
build_bdd_eval.py decides (time of day, clear/rain, road type, crosswalk, signal colour, object
kinds and screen positions). A constraint is one scene term, one object kind, or one position.

Combinations are drawn from real pool images, so every one has at least one match, and are kept
only when they are neither too common nor too rare in the pool. Combinations equal to a catalog
condition are skipped. The output uses the catalog's expression format, so the same query
generator and evaluator apply.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

try:
    from .build_bdd_eval import DECIDABILITY_SAMPLE, SEED, both_facts, judge, load_pool
    from .condition_data import ROOT, load_conditions, stable_key, write_json, write_jsonl
    from .generate_condition_queries import canonical_target
except ImportError:
    from build_bdd_eval import DECIDABILITY_SAMPLE, SEED, both_facts, judge, load_pool
    from condition_data import ROOT, load_conditions, stable_key, write_json, write_jsonl
    from generate_condition_queries import canonical_target

OUT = ROOT/'datasets/dashcam_reranker_bdd_combo'
# At most one term per group; the groups are the scene terms build_bdd_eval decides with negatives.
SCENE_GROUPS = [['day', 'night', 'twilight'], ['clear', 'rain'], ['urban', 'residential', 'highway'],
                ['crosswalk'], ['red_signal', 'green_signal', 'yellow_signal']]
# Cars appear in 97% of BDD images (and at almost every screen position), so a car constraint
# would add length without narrowing the search; cars are left out.
KINDS = ['bus', 'truck', 'pedestrian', 'bicycle', 'motorcycle']
POSITIONS = ['left', 'center', 'right']
SCENE_JA = {'day': '昼', 'night': '夜', 'twilight': '薄明', 'clear': '晴れ', 'rain': '雨が降っている',
            'urban': '市街地', 'residential': '住宅街', 'highway': '高速道路', 'crosswalk': '横断歩道が見える',
            'red_signal': '赤信号が見える', 'green_signal': '青信号が見える', 'yellow_signal': '黄信号が見える'}
KIND_JA = {'bus': 'バス', 'truck': 'トラック', 'pedestrian': '歩行者', 'bicycle': '自転車', 'motorcycle': 'バイク'}
POSITION_JA = {'left': '画面左', 'center': '画面中央', 'right': '画面右'}
PER_SIZE = 40
# Share of pool images that match. Above the upper bound the retrieved top-50 is almost all
# positive; below the lower bound there are too few matches in the whole pool.
MIN_RATE, MAX_RATE = 0.0005, 0.15


def size(scenes, objects):
    return len(scenes) + sum(len(o) for o in objects)


def expression(scenes, objects):
    terms = [{'op': 'scene', 'name': s} for s in scenes]
    if objects: terms.append({'op': 'exists', 'objects': objects})
    return terms[0] if len(terms) == 1 else {'op': 'all', 'terms': terms}


def sentence(scenes, objects):
    """Plain scene description for the query generator, e.g. '夜・市街地。画面左に歩行者、バスがいる'."""
    parts = []
    if scenes: parts.append('・'.join(SCENE_JA[s] for s in scenes))
    if objects:
        names = [f"{POSITION_JA[o['position']]}に{KIND_JA[o['kind']]}" if 'position' in o else KIND_JA[o['kind']] for o in objects]
        parts.append('、'.join(names) + 'がいる')
    return '。'.join(parts)


def true_atoms(facts):
    scenes = {s for group in SCENE_GROUPS for s in group if judge({'op': 'scene', 'name': s}, facts) == 'yes'}
    objects = {}
    for kind in KINDS:
        if judge({'op': 'exists', 'objects': [{'kind': kind}]}, facts) != 'yes': continue
        objects[kind] = [p for p in POSITIONS if judge({'op': 'exists', 'objects': [{'kind': kind, 'position': p}]}, facts) == 'yes']
    return scenes, objects


def draw(rng, scenes, objects, target):
    """One combination of `target` constraints that the image satisfies, or None."""
    for _ in range(20):
        kinds = rng.sample(sorted(objects), min(len(objects), rng.choice([1, 2])))
        # Two objects get positions together or not at all: in 「画面左に歩行者、自転車」 the reader
        # cannot tell whether 画面左 also applies to the bicycle.
        positioned = all(objects[k] for k in kinds) and rng.random() < 0.6
        chosen = [{'kind': k, **({'position': rng.choice(objects[k])} if positioned else {})} for k in kinds]
        groups = [g for g in SCENE_GROUPS if scenes & set(g)]
        rng.shuffle(groups)
        picked = []
        for g in groups:
            if size(picked, chosen) >= target: break
            picked.append(rng.choice(sorted(scenes & set(g))))
        if size(picked, chosen) == target:
            return sorted(picked, key=lambda s: [s in g for g in SCENE_GROUPS].index(True)), chosen
    return None


def run(args):
    images, records = load_pool()
    order = sorted(records, key=lambda i: stable_key(SEED, i))
    sample = [both_facts(records[i], images[i]) for i in order[:DECIDABILITY_SAMPLE]]
    catalog = {json.dumps(canonical_target(c['expression'])) for c in load_conditions()
               if c['expression']['op'] in ('all', 'scene', 'exists')}
    rng = random.Random(args.seed)
    seen, rows, rejected = set(), [], {'catalog': 0, 'rate': 0}
    counts = {3: 0, 4: 0}
    for image_id in order[DECIDABILITY_SAMPLE:]:
        if all(v >= PER_SIZE for v in counts.values()): break
        target = rng.choice([k for k, v in counts.items() if v < PER_SIZE])
        drawn = draw(rng, *true_atoms(both_facts(records[image_id], images[image_id])), target)
        if drawn is None: continue
        scenes, objects = drawn
        expr = expression(scenes, objects)
        key = json.dumps(canonical_target(expr))
        if key in seen: continue
        seen.add(key)
        if key in catalog: rejected['catalog'] += 1; continue
        yes = sum(judge(expr, f) == 'yes' for f in sample)
        if not MIN_RATE <= yes / len(sample) <= MAX_RATE: rejected['rate'] += 1; continue
        counts[target] += 1
        rows.append({'id': f'M{target}{counts[target]:02d}', 'query': sentence(scenes, objects), 'expression': expr,
                     'constraints': target, 'source_image_id': image_id, 'sample_positive_rate': yes / len(sample)})
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out/'conditions.jsonl', rows)
    write_json(args.out/'reports/conditions_stats.json', {
        'seed': args.seed, 'per_size': PER_SIZE, 'rate_bounds': [MIN_RATE, MAX_RATE], 'counts': counts,
        'rejected': rejected, 'distinct_tried': len(seen)})
    for r in rows: print(r['id'], r['query'], f"{r['sample_positive_rate']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, default=20260928)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
