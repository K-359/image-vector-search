"""Export condition-balanced reranker pairs from completed, saved judgments.

No model calls, label regeneration, or semantic consistency filtering.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
from pathlib import Path

try:
    from .condition_data import ROOT, contains_heldout, digest, load_conditions, read_jsonl, stable_key, write_json, write_jsonl
except ImportError:
    from condition_data import ROOT, contains_heldout, digest, load_conditions, read_jsonl, stable_key, write_json, write_jsonl

SPLITS=('train','val','test')
DEFAULT_OUT=ROOT/'datasets/dashcam_reranker_v3_conditions'


def excluded_conditions(conditions):
    heldout=[c['expression'] for c in conditions if c['id'].startswith('H')]
    return {(s,c['id']):c['expression']['op']=='deferred' or
            (s in ('train','val') and any(contains_heldout(c['expression'],h) for h in heldout))
            for s in SPLITS for c in conditions}


def load_judgments(path, manifest, conditions):
    """Verify file identity/completeness and split rules, not label correctness."""
    byid={r['image_id']:r for r in manifest}
    ids={c['id'] for c in conditions}
    excluded=excluded_conditions(conditions)
    seen=defaultdict(set); buckets=defaultdict(lambda:defaultdict(list))
    with path.open() as f:
        for line in f:
            r=json.loads(line); image_id=r['image_id']; cid=r['condition_id']; split=r['split']
            if image_id not in byid or cid not in ids or split!=byid[image_id]['split']:
                raise ValueError('judgment does not match manifest/catalog')
            if cid in seen[image_id]:
                raise ValueError('duplicate image-condition judgment')
            seen[image_id].add(cid)
            value=r['judgment']
            if value not in ('yes','no','unknown'):
                raise ValueError('invalid judgment state')
            expected='excluded' if excluded[(split,cid)] else 'unknown' if value=='unknown' else 'teacher_candidate'
            if r['usage']!=expected:
                raise ValueError('judgment usage does not match split/unknown policy')
            buckets[(split,cid)][value].append(image_id)
    if set(seen)!=set(byid) or any(v!=ids for v in seen.values()):
        raise ValueError('missing image-condition judgments')
    return buckets


def build_pairs(manifest, conditions, buckets, rankings, *, seed=20260924):
    """Targets are caps on sampled pairs; full available coverage stays in the report."""
    byid={r['image_id']:r for r in manifest}
    if len(byid)!=len(manifest) or len({r['group_id'] for r in manifest})!=len(manifest):
        raise ValueError('duplicate image or group in manifest')
    for row in manifest:
        if row['split'] not in SPLITS:
            raise ValueError('invalid image split')
    excluded=excluded_conditions(conditions)
    pair_rows={s:[] for s in SPLITS}; queries=[]; coverage=[]
    usage=Counter()
    ranks={r['condition_id']:{h['image_id']:h['rank'] for h in r['hits']} for r in rankings}
    # Spread limited slots over source images. Seeded tie-breaking is reproducible.
    def pick(pool,n,cid,split):
        return sorted(pool,key=lambda i:(usage[i],stable_key(seed,f'{split}:{cid}:{i}')))[:n]

    for split in SPLITS:
        active=[]
        # Allocate positive slots before negative slots so rare positives get first use.
        for c in sorted(conditions,key=lambda c:stable_key(seed,c['id'])):
            cid=c['id']; pool=buckets.get((split,cid),{})
            pos=list(pool.get('yes',[])); neg=list(pool.get('no',[])); unknown=pool.get('unknown',[])
            for value,images in [('yes',pos),('no',neg),('unknown',unknown)]:
                if len(images)!=len(set(images)) or any(i not in byid or byid[i]['split']!=split for i in images):
                    raise ValueError('invalid candidate pool')
            if set(pos)&set(neg) or set(pos)&set(unknown) or set(neg)&set(unknown):
                raise ValueError('overlapping judgment pools')
            pt=int(c['allocation'][f'{split}_positive_target'])
            nt=int(c['allocation'][f'{split}_negative_target'])
            status=('deferred_temporal' if c['expression']['op']=='deferred' else
                    'excluded_heldout' if excluded[(split,cid)] else
                    'no_positive' if not pos else 'no_negative' if not neg else
                    'targets_met' if len(pos)>=pt and len(neg)>=nt else 'below_target')
            stat={'split':split,'condition_id':cid,'query_text':c['query'],
                  'status':status,'available_positive':len(pos),'available_negative':len(neg),
                  'unknown':len(unknown),'positive_target':pt,'negative_target':nt,
                  'positive_pairs':0,'negative_pairs':0,'hard_negative_pairs':0,'random_negative_pairs':0}
            coverage.append(stat)
            if status not in ('targets_met','below_target') or not pt or not nt:
                continue
            selected_positive=pick(pos,pt,cid,split)
            for image_id in selected_positive:
                usage[image_id]+=1
            active.append((c,neg,selected_positive,stat))
        for c,neg,positive,stat in active:
            cid=c['id']; nt=stat['negative_target']; ranking=ranks.get(cid,{})
            # Relevance comes exclusively from the saved 'no' judgment. Retrieval
            # rank only selects difficult candidates within that negative pool.
            hard=sorted((i for i in neg if i in ranking),key=lambda i:(ranking[i],usage[i],stable_key(seed,i)))[:(nt+1)//2]
            rest=pick(set(neg)-set(hard),nt-len(hard),cid,split)
            qid=f'condition:{split}:{cid}'
            for image_id,label,kind in ([(i,1,'positive') for i in positive]+
                                       [(i,0,'hard_negative') for i in hard]+
                                       [(i,0,'random_negative') for i in rest]):
                row=byid[image_id]
                if label==0:
                    usage[image_id]+=1
                pair_rows[split].append({'pair_id':'pair:'+stable_key(seed,f'{qid}:{image_id}')[:24],
                    'query_id':qid,'condition_id':cid,'query_text':c['query'],
                    'image_id':image_id,'image_path':row['image_path'],'image_sha256':row['sha256'],
                    'group_id':row['group_id'],'label':label,'negative_type':kind,'split':split,
                    'caption':None,'difficulty':c['allocation']['category'],
                    'selection_route':row['selection_route'],'retrieval_rank':ranking.get(image_id),
                    'label_source':'annotations/condition_judgments.jsonl','review_status':'teacher_only'})
            stat.update(positive_pairs=len(positive),negative_pairs=len(hard)+len(rest),
                        hard_negative_pairs=len(hard),random_negative_pairs=len(rest))
            queries.append({'query_id':qid,'condition_id':cid,'query_text':c['query'],'split':split,
                            'expression':c['expression'],'difficulty':c['allocation']['category'],
                            'positive_pairs':len(positive),'negative_pairs':len(hard)+len(rest)})
    for split in SPLITS:
        pair_rows[split].sort(key=lambda r:(r['query_id'],r['pair_id']))
    queries.sort(key=lambda r:r['query_id'])
    coverage.sort(key=lambda r:(r['split'],r['condition_id']))
    return pair_rows,queries,coverage


def run(args):
    out=args.out
    report=json.loads((out/'annotations/report.json').read_text())
    if not report['complete'] or report['facts_count']!=5000:
        raise ValueError('all 5,000 annotations must be complete before pair export')
    sources=['manifests/selected_5000.jsonl','annotations/facts.jsonl','annotations/condition_judgments.jsonl',
             'annotations/annotation_config.json','annotations/report.json','selection/rankings.jsonl']
    fingerprints={p:digest(out/p) for p in sources}
    if fingerprints['annotations/facts.jsonl']!=report['facts_sha256'] or fingerprints['manifests/selected_5000.jsonl']!=report['manifest_sha256']:
        raise ValueError('annotation report no longer matches its inputs')
    config=json.loads((out/'annotations/annotation_config.json').read_text())
    catalog_sha=digest(ROOT/'docs/search-condition-catalog.md')
    allocation_sha=digest(ROOT/'docs/search-condition-allocation.csv')
    if config['catalog_sha256']!=catalog_sha or config['allocation_sha256']!=allocation_sha:
        raise ValueError('catalog or allocation changed after annotation')
    manifest=read_jsonl(out/'manifests/selected_5000.jsonl'); conditions=load_conditions()
    if len(manifest)!=5000 or Counter(r['split'] for r in manifest)!={'train':3500,'val':500,'test':1000}:
        raise ValueError('unexpected manifest size/splits')
    for row in manifest:
        if not (ROOT/row['image_path']).is_file():
            raise ValueError('missing source image')
    buckets=load_judgments(out/'annotations/condition_judgments.jsonl',manifest,conditions)
    pairs,queries,coverage=build_pairs(manifest,conditions,buckets,read_jsonl(out/'selection/rankings.jsonl'),seed=args.seed)
    stats={'schema_version':'condition-pairs-v1','seed':args.seed,'teacher_only':True,
           'pair_exported':True,'training_launched':False,
           'sources_sha256':fingerprints,'catalog_sha256':catalog_sha,'allocation_sha256':allocation_sha,
           'builder_sha256':digest(Path(__file__)),
           'caption_mode':'image only; captions are not generated',
           'positive_sampling':'prefer fewer prior uses, then seeded tie break',
           'negative_sampling':'up to half (rounded up) high-retrieval-ranked known negatives; remainder fewer prior uses then seeded tie break',
           'coverage_scope':'teacher-derived; absent positive/negative queries omitted from pairs but listed in coverage',
           'query_split_policy':'shared fixed basic/composite queries across disjoint images; H queries only in test',
           'sequence_independence':'unverified; exact content groups isolated',
           'splits':{}}
    used_ids={s:{r['image_id'] for r in pairs[s]} for s in SPLITS}
    used_groups={s:{r['group_id'] for r in pairs[s]} for s in SPLITS}
    isolation={'image_overlap':{},'group_overlap':{},'query_text_overlap':{},
               'heldout_training_queries':[], 'note':'Ordinary fixed query text overlaps are intentional; image/content groups must not overlap.'}
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            key=f'{a}_{b}'
            isolation['image_overlap'][key]=sorted(used_ids[a]&used_ids[b])
            isolation['group_overlap'][key]=sorted(used_groups[a]&used_groups[b])
            isolation['query_text_overlap'][key]=sorted({r['query_text'] for r in pairs[a]}&{r['query_text'] for r in pairs[b]})
            if isolation['image_overlap'][key] or isolation['group_overlap'][key]:
                raise ValueError('image/group leakage across splits')
    for split in SPLITS:
        rows=pairs[split]
        if not rows or len({r['pair_id'] for r in rows})!=len(rows):
            raise ValueError('empty split or duplicate pair IDs')
        stats['splits'][split]={'manifest_images':sum(r['split']==split for r in manifest),
            'paired_images':len(used_ids[split]),'unpaired_images':sum(r['split']==split for r in manifest)-len(used_ids[split]),
            'queries':sum(r['split']==split for r in queries),'pairs':len(rows),
            'positive_pairs':sum(r['label']==1 for r in rows),'negative_pairs':sum(r['label']==0 for r in rows),
            'negative_types':dict(Counter(r['negative_type'] for r in rows if not r['label'])),
            'condition_statuses':dict(Counter(r['status'] for r in coverage if r['split']==split))}
        # Refuse to silently replace a frozen dataset with a different sample.
        path=out/f'pairs.{split}.jsonl'
        if path.exists() and read_jsonl(path)!=rows:
            raise ValueError(f'{path} already exists with different pairs')
    for split in SPLITS:
        write_jsonl(out/f'pairs.{split}.jsonl',pairs[split])
    write_jsonl(out/'derived/queries.jsonl',queries)
    write_json(out/'reports/condition_pair_coverage.json',coverage)
    csv_path=out/'reports/condition_pair_coverage.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    stats['outputs_sha256']={f'pairs.{s}.jsonl':digest(out/f'pairs.{s}.jsonl') for s in SPLITS}
    stats['outputs_sha256']['derived/queries.jsonl']=digest(out/'derived/queries.jsonl')
    write_json(out/'reports/condition_pair_stats.json',stats)
    write_json(out/'reports/condition_pair_isolation.json',isolation)
    for split in SPLITS:
        print(split,json.dumps(stats['splits'][split],ensure_ascii=False),flush=True)
    print('Pair export complete. No LLM inference or training started.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--seed',type=int,default=20260924)
    args=parser.parse_args();args.out=args.out.resolve()
    with (args.out/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__':
    main()
