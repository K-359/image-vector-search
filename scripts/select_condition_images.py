"""Allocate candidates into the fixed 5,000-image budget without inferring labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import heapq
import json
from pathlib import Path

try:
    from .condition_data import ROOT, digest, load_conditions, read_jsonl, stable_key, write_json, write_jsonl
except ImportError:
    from condition_data import ROOT, digest, load_conditions, read_jsonl, stable_key, write_json, write_jsonl

BUDGET = {'train':3500, 'val':500, 'test':1000}
RANK_BANDS = {'top': (1, 40), 'middle': (41, 200), 'tail': (201, 1000)}


def rank_band(rank):
    for band, (lo, hi) in RANK_BANDS.items():
        if lo <= rank <= hi:
            return band
    raise ValueError(f'unsupported retrieval rank: {rank}')


def allocate_retrieval(random_rows, pilot_rows, candidates, conditions, *, seed=20260924,
                       budget=None, band_budget=None):
    """Balance retrieval opportunities, not positive/negative label counts.

    A hit is only a sampling signal. Goals weight sampling priority; actual condition
    coverage remains unknown until annotation. No fabricated visual reviews are used.
    """
    budget = dict(BUDGET if budget is None else budget)
    goals = {(s, c['id']): int(c['allocation'][f'{s}_positive_target'])
             for c in conditions for s in budget}
    by_candidate = {r['image_id']: r for r in candidates}
    if len(by_candidate) != len(candidates):
        raise ValueError('duplicate candidate IDs')
    condition_ids = {c['id'] for c in conditions}
    hits_by_id = {}
    for row in candidates:
        hits = row['hits']
        if len({h['condition_id'] for h in hits}) != len(hits):
            raise ValueError('duplicate condition hit')
        for hit in hits:
            if hit['condition_id'] not in condition_ids:
                raise ValueError('unknown retrieval condition')
            rank_band(hit['rank'])
            if hit['image_id'] != row['image_id']:
                raise ValueError('retrieval hit belongs to another image')
        hits_by_id[row['image_id']] = hits

    if band_budget is None:
        random_counts = Counter(r['split'] for r in random_rows)
        band_budget = {}
        for split, total in budget.items():
            n = total - random_counts[split]
            band_budget[split] = {'top': n - 2*(n//4), 'middle': n//4, 'tail': n//4}
    selected, byid, used_groups = [], {}, set()
    split_counts, band_counts, signals = Counter(), Counter(), Counter()

    def eligible_hits(row, split):
        return [h for h in hits_by_id.get(row['image_id'], [])
                if goals.get((split, h['condition_id']), 0)]

    def add(row, split, hit=None, pilot=False):
        if row['image_id'] in byid or row['group_id'] in used_groups:
            raise ValueError('duplicate image or group')
        if split_counts[split] >= budget[split]:
            raise ValueError('split capacity exceeded')
        item = {**row, 'split': split, 'seed': seed}
        if pilot:
            item['pilot'] = True
        if row.get('selection_route') != 'random':
            if hit is None:
                hit = {'condition_id': row['target_condition'], 'rank': row['retrieval_rank'],
                       'score': row['retrieval_score']}
            band = rank_band(hit['rank'])
            item.update(selection_route='condition', target_condition=hit['condition_id'],
                        retrieval_rank=hit['rank'], retrieval_score=hit['score'],
                        selection_rank_band=band, selection_method='balanced_retrieval_v1',
                        selection_evidence='retrieval_candidate_only; not a relevance label')
            band_counts[(split, band)] += 1
        selected.append(item)
        byid[item['image_id']] = item
        used_groups.add(item['group_id'])
        split_counts[split] += 1
        for h in eligible_hits(item, split):
            signals[(split, h['condition_id'])] += 1

    for row in random_rows:
        add(row, row['split'])
    for row in pilot_rows:
        if row['split'] != 'train':
            raise ValueError('pilot must remain in train')
        if row['image_id'] in byid:
            existing = byid[row['image_id']]
            if any(existing[k] != row[k] for k in ('split', 'group_id', 'selection_route', 'sha256')):
                raise ValueError('pilot conflicts with reserved random image')
            existing['pilot'] = True
        else:
            add(row, 'train', pilot=True)

    ties = {r['image_id']: stable_key(seed, r['image_id']) for r in candidates}
    for split in ('test', 'val', 'train'):
        if split not in budget:
            continue
        remaining = budget[split] - sum(r['split'] == split for r in random_rows)
        if sum(band_budget[split].values()) != remaining:
            raise ValueError('rank band budget does not match split capacity')
        for band in RANK_BANDS:
            if band_counts[(split, band)] > band_budget[split][band]:
                raise ValueError('pilot exceeds rank band budget')
            eligible = {}
            for row in candidates:
                if row['image_id'] in byid or row['group_id'] in used_groups:
                    continue
                hits = eligible_hits(row, split)
                band_hits = [h for h in hits if rank_band(h['rank']) == band]
                if band_hits:
                    eligible[row['image_id']] = (row, hits, band_hits)

            def priority(image_id):
                row, hits, band_hits = eligible[image_id]
                missing = sum(signals[(split, h['condition_id'])] == 0 for h in hits)
                heldout_missing = sum(h['condition_id'].startswith('H') and
                                      signals[(split, h['condition_id'])] == 0 for h in hits)
                gain = sum(max(0, 1-signals[(split, h['condition_id'])]/goals[(split, h['condition_id'])])
                           / h['rank']**0.5 for h in hits)
                return (-heldout_missing, -missing, -gain,
                        min(h['rank'] for h in band_hits), ties[image_id])

            heap = [(priority(i), i) for i in eligible]
            heapq.heapify(heap)
            while heap and band_counts[(split, band)] < band_budget[split][band]:
                old, image_id = heapq.heappop(heap)
                row, hits, band_hits = eligible[image_id]
                if image_id in byid or row['group_id'] in used_groups:
                    continue
                new = priority(image_id)
                if heap and new > heap[0][0]:
                    heapq.heappush(heap, (new, image_id))
                    continue
                primary = min(band_hits, key=lambda h: (
                    signals[(split, h['condition_id'])]/goals[(split, h['condition_id'])],
                    h['rank'], h['condition_id']))
                add(row, split, primary)

    opportunities = [{'split': s, 'condition_id': c['id'],
                      'sampling_weight': goals[(s, c['id'])],
                      'selected_retrieval_candidates': signals[(s, c['id'])],
                      'actual_positive_count': None, 'actual_negative_count': None}
                     for s in budget for c in conditions if goals[(s, c['id'])]]
    report = {
        'method': 'balanced_retrieval_v1', 'seed': seed,
        'complete_image_budget': all(split_counts[s] == n for s, n in budget.items()),
        'split_counts': dict(split_counts), 'target_counts': budget,
        'unfilled_image_slots': {s: n-split_counts[s] for s, n in budget.items()},
        'selection_route_counts': dict(Counter(r['selection_route'] for r in selected)),
        'rank_band_targets': band_budget,
        'rank_band_counts': {s: {b: band_counts[(s, b)] for b in RANK_BANDS} for s in budget},
        'coverage_source': 'retrieval sampling opportunities only; actual coverage pending annotation',
        'condition_sampling': opportunities,
        'conditions_without_retrieval_candidates': [r for r in opportunities if not r['selected_retrieval_candidates']],
        'sequence_independence': 'unverified; groups represent exact content duplicates',
        'candidate_visual_review': 'not required for this selection; pilot review retained',
        'quality_policy': 'record existing pilot issues; no additional remediation or repeated pilot required',
    }
    return selected, report


def allocate_reviewed(random_rows, pilot_rows, candidates, reviews, conditions, *, seed=20260924, budget=None):
    """Lazy greedy coverage allocation. Review judgments are provisional, never gold."""
    budget = dict(BUDGET if budget is None else budget)
    goals = {}
    for c in conditions:
        for split in budget:
            for value, label in [('yes', 'positive'), ('no', 'negative')]:
                goals[(split, c['id'], value)] = int(c['allocation'][f'{split}_{label}_target'])
    condition_ids = {c['id'] for c in conditions}
    if len(reviews) != len({r['image_id'] for r in reviews}):
        raise ValueError('duplicate reviews')
    reviewed = {r['image_id']:r for r in reviews}
    for review in reviews:
        if not review.get('reviewer') or not review.get('notes'):
            raise ValueError('review provenance is required')
        if review.get('decision') not in ('accept','reject'):
            raise ValueError('invalid review decision')
        if not set(review.get('judgments',{})) <= condition_ids:
            raise ValueError('unknown condition in review')
        if not all(v in ('yes','no','unknown') for v in review.get('judgments',{}).values()):
            raise ValueError('invalid review judgment')
    selected = []
    byid = {}
    used_groups = set()
    split_counts = Counter()
    counts = Counter()

    def judgments(row):
        return reviewed.get(row['image_id'],{}).get('judgments',{})

    def add(row, split):
        if row['image_id'] in byid or row['group_id'] in used_groups:
            raise ValueError('duplicate image or group')
        if split_counts[split] >= budget[split]:
            raise ValueError('split capacity exceeded')
        selected_row = {**row, 'split':split}
        selected.append(selected_row)
        byid[row['image_id']] = selected_row
        used_groups.add(row['group_id'])
        split_counts[split] += 1
        for cid, value in judgments(row).items():
            if goals.get((split,cid,value),0):
                counts[(split,cid,value)] += 1

    for row in random_rows:
        add(row,row['split'])
    for row in pilot_rows:
        if row['split'] != 'train':
            raise ValueError('pilot must remain in train')
        if row['image_id'] in byid:
            existing = byid[row['image_id']]
            if existing['split'] != row['split'] or existing['group_id'] != row['group_id']:
                raise ValueError('pilot conflicts with reserved random split')
            existing['pilot'] = True
            continue
        add(row,'train')

    if len(candidates) != len({r['image_id'] for r in candidates}):
        raise ValueError('duplicate candidate IDs')
    eligible = [r for r in candidates if reviewed.get(r['image_id'],{}).get('decision')=='accept']

    def priority(row, split):
        missing = 0
        gain = 0.0
        heldout_missing = 0
        heldout_gain = 0.0
        for cid, value in judgments(row).items():
            target = goals.get((split,cid,value),0)
            current = counts[(split,cid,value)]
            if not target or current >= target:
                continue
            missing += current == 0
            gain += (target-current)/target
            if cid.startswith('H'):
                heldout_missing += current == 0
                heldout_gain += (target-current)/target
        # Every component decreases as coverage accumulates, enabling lazy heap updates.
        return (-heldout_missing,-heldout_gain,-missing,-gain,stable_key(seed,row['image_id']))

    for split in ('test','val','train'):
        if split not in budget:
            continue
        heap = [(priority(r,split),r['image_id'],r) for r in eligible if r['image_id'] not in byid and r['group_id'] not in used_groups]
        heapq.heapify(heap)
        while heap and split_counts[split] < budget[split]:
            old, image_id, row = heapq.heappop(heap)
            if image_id in byid or row['group_id'] in used_groups:
                continue
            new = priority(row,split)
            if heap and new > heap[0][0]:
                heapq.heappush(heap,(new,image_id,row))
                continue
            add({**row,'selection_route':'condition','selection_review':reviewed[image_id]},split)

    deficits = [
        {'split':split,'condition_id':cid,'judgment':value,'target':target,'provisionally_observed':counts[key],'shortfall':max(0,target-counts[key])}
        for key,target in goals.items() if target
        for split,cid,value in [key]
    ]
    report = {
        'complete_image_budget': all(split_counts[s]==n for s,n in budget.items()),
        'split_counts':dict(split_counts), 'target_counts':budget,
        'unfilled_image_slots':{s:n-split_counts[s] for s,n in budget.items()},
        'selection_route_counts':dict(Counter(r['selection_route'] for r in selected)),
        'coverage_source':'provisional visual candidate reviews; not final teacher or evaluation labels',
        'unreviewed_random_images':sum(r['image_id'] not in reviewed for r in random_rows),
        'deficits':deficits,
    }
    return selected, report


def run(args):
    randoms=read_jsonl(args.out/'manifests/random_1000.jsonl')
    pilot=read_jsonl(args.out/'manifests/pilot_50.jsonl')
    if Counter(r['split'] for r in randoms)!={'train':700,'val':100,'test':200}:
        raise ValueError('expected reserved random 700/100/200')
    if len(pilot)!=50 or Counter(r['selection_route'] for r in pilot)!={'condition':40,'random':10}:
        raise ValueError('expected fixed pilot of 40 condition + 10 random images')
    conditions=load_conditions()
    inputs=['manifests/random_1000.jsonl', 'manifests/pilot_50.jsonl',
            'selection/candidates.jsonl', 'selection/retrieval.json']
    provenance={p:digest(args.out/p) for p in inputs}
    provenance['catalog_sha256']=digest(ROOT/'docs/search-condition-catalog.md')
    provenance['allocation_sha256']=digest(ROOT/'docs/search-condition-allocation.csv')
    retrieval=json.loads((args.out/'selection/retrieval.json').read_text())
    if retrieval['catalog_sha256'] != provenance['catalog_sha256']:
        raise ValueError('retrieval catalog has changed')
    candidates=read_jsonl(args.out/'selection/candidates.jsonl')
    if args.method=='retrieval':
        selected,report=allocate_retrieval(randoms,pilot,candidates,conditions,seed=args.seed)
    else:
        provenance['selection/reviews.jsonl']=digest(args.out/'selection/reviews.jsonl')
        selected,report=allocate_reviewed(randoms,pilot,candidates,
            read_jsonl(args.out/'selection/reviews.jsonl'),conditions,seed=args.seed)
    report['inputs_sha256']=provenance
    report['selector_sha256']=digest(Path(__file__))
    final=args.out/'manifests/selected_5000.jsonl'
    if final.exists() and read_jsonl(final)!=selected:
        raise ValueError('final manifest already fixed')
    if not report['complete_image_budget']:
        write_json(args.out/'selection/allocation_report.json',report)
        write_jsonl(args.out/'selection/allocation_draft.jsonl',selected)
        print('Incomplete; final manifest not written:',report['unfilled_image_slots'])
        return
    # Recheck the actual selected files before fixing the manifest.
    def verify(row):
        if digest(ROOT/row['image_path']) != row['sha256']:
            raise ValueError(f"source image changed: {row['image_id']}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify,selected))
    if len({r['sha256'] for r in selected}) != 5000:
        raise ValueError('duplicate image content in final selection')
    report['source_hashes_verified']=len(selected)
    write_jsonl(final,selected)
    report['manifest_sha256']=digest(final)
    write_json(args.out/'selection/allocation_report.json',report)
    print('5,000-image manifest fixed. No teacher inference or training started.')
    print('Split counts:',report['split_counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'datasets/dashcam_reranker_v3_conditions')
    parser.add_argument('--seed',type=int,default=20260924)
    parser.add_argument('--method',choices=['reviewed','retrieval'],default='reviewed',
                        help='retrieval balances candidate hits without inventing relevance labels')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__': main()
