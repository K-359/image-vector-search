"""New condition-driven data pipeline. Stages are explicit; no training is launched.

Run with the existing conda image environment. Outputs are isolated from v1/v2.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

try:
    from .condition_data import (
        FACT_PROMPT, ROOT, VERSION, contains_heldout, digest, evaluate,
        fact_schema, load_conditions, quality_flags, read_jsonl, stable_key,
        validate_facts, write_json, write_jsonl,
    )
except ImportError:
    from condition_data import (
        FACT_PROMPT, ROOT, VERSION, contains_heldout, digest, evaluate,
        fact_schema, load_conditions, quality_flags, read_jsonl, stable_key,
        validate_facts, write_json, write_jsonl,
    )

DEFAULT_OUT = ROOT/'datasets/dashcam_reranker_v3_conditions'
MODEL = 'Qwen/Qwen3-VL-Embedding-2B'
REVISION = '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'
QUERY_PROMPT = "Retrieve dashcam images that visually show the road scene, traffic participant, dangerous behavior, collision, or near-miss event described in the user's query."
PILOT_CONDITIONS = ['OBJ05','OBJ06','OBJ07','COL03','COL05','POS13','POS18','LOC25','LOC26','ORI09','ROAD05','ROAD09','ROAD10','ROAD13','ROAD14','ENV02','CO10','CE04','CA01','CB13']


def now(): return datetime.now(timezone.utc).isoformat()


def append(path, row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())


def scan_image(item):
    from PIL import Image
    index_id,path=item
    try:
        raw=path.read_bytes()
        with Image.open(path) as im:
            im.load(); width,height=im.size
            tiny=im.convert('L').resize((9,8))
            pixels=list(tiny.get_flattened_data() if hasattr(tiny,'get_flattened_data') else tiny.getdata()); bits=0
            for y in range(8):
                for x in range(8): bits=(bits<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
        return {'index_id':index_id,'image_id':path.stem,'image_path':str(path.relative_to(ROOT)),
                'sha256':hashlib.sha256(raw).hexdigest(),'dhash':f'{bits:016x}','width':width,'height':height,'error':None}
    except Exception as e:
        return {'index_id':index_id,'image_id':path.stem,'image_path':str(path.relative_to(ROOT)),'error':str(e)}


def run_inventory(args):
    started=time.monotonic(); paths_path=ROOT/'data_100k/image_paths.json'
    paths=json.loads(paths_path.read_text()); names=[Path(p).name for p in paths]
    if len(names)!=len(set(names)): raise ValueError('index basenames are not unique')
    excluded={Path(p).name for p in json.loads((ROOT/'data_100k/test_paths.json').read_text())}
    path=args.out/'inventory/images.jsonl'
    config={'paths_sha256':digest(paths_path),'seed':args.seed,'excluded':sorted(excluded),'algorithm':'sha256+dhash64-v1'}
    config_path=args.out/'inventory/config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config: raise ValueError('inventory config changed')
    write_json(config_path,config)
    rows=read_jsonl(path); done={r['index_id'] for r in rows}
    pending=[(i,ROOT/'images_100k'/n) for i,n in enumerate(names) if n not in excluded and i not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with path.open('a') as f:
            for n,row in enumerate(pool.map(scan_image,pending),1):
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                rows.append(row)
                if n%5000==0: f.flush(); print(f'inventory {n}/{len(pending)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    valid=sorted((r for r in rows if not r['error']),key=lambda r:r['image_id'])
    groups={}
    for r in valid:
        r['group_id']=r['sha256']; groups.setdefault(r['group_id'],r)
    representatives=sorted(groups.values(),key=lambda r:stable_key(args.seed,r['image_id']))
    if len(representatives)<5000: raise ValueError('insufficient distinct images')
    write_jsonl(args.out/'inventory/representatives.jsonl',representatives)
    randoms=[]
    for n,r in enumerate(representatives[:1000]):
        split='train' if n<700 else 'val' if n<800 else 'test'
        randoms.append({**r,'split':split,'selection_route':'random','seed':args.seed})
    write_jsonl(args.out/'manifests/random_1000.jsonl',randoms)
    write_json(args.out/'inventory/report.json',{'created_at':now(),'source_count':len(paths),'eligible_count':len(rows),'readable_count':len(valid),'exact_duplicate_count':len(valid)-len(groups),'representatives':len(groups),'errors':[r for r in rows if r['error']],'random_split_counts':dict(Counter(r['split'] for r in randoms)),'sequence_independence':'unverified','near_duplicates':'dhash candidates require visual verification','elapsed_seconds':time.monotonic()-started,'inventory_sha256':digest(path)})
    print('inventory complete',len(rows),len(groups),flush=True)


def load_index():
    import faiss
    import numpy as np
    paths=json.loads((ROOT/'data_100k/image_paths.json').read_text())
    index=faiss.read_index(str(ROOT/'data_100k/images.faiss'))
    if index.ntotal!=len(paths) or index.d!=2048 or index.metric_type!=faiss.METRIC_INNER_PRODUCT:
        raise ValueError('unexpected index shape/metric')
    if not np.array_equal(faiss.vector_to_array(index.id_map),np.arange(len(paths))): raise ValueError('index ID mismatch')
    return index,paths


def run_retrieve(args):
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    started=time.monotonic(); conditions=load_conditions()
    conditions=[c for c in conditions if c['expression']['op']!='deferred']
    index,paths=load_index(); faiss.omp_set_num_threads(args.workers)
    source_hash=digest(ROOT/'data_100k/images.faiss')
    model=SentenceTransformer(MODEL,revision=REVISION,local_files_only=True,device=args.device,model_kwargs={'torch_dtype':torch.bfloat16})
    # Re-embed three images using the original (empty) image prompt. Do not trust copied metadata.
    probe_ids=[0,12345,67890]; base=faiss.downcast_index(index.index)
    new=model.encode([{'image':str(ROOT/'images_100k'/Path(paths[i]).name)} for i in probe_ids],prompt='',batch_size=1,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype('float32')
    old=np.stack([base.reconstruct(i) for i in probe_ids])
    similarity=(new*old).sum(1)/(np.linalg.norm(new,axis=1)*np.linalg.norm(old,axis=1))
    if np.any(similarity<.995): raise ValueError(f'index sample mismatch {similarity}')
    sample=base.reconstruct_n(0,min(1000,index.ntotal))
    norms=np.linalg.norm(sample,axis=1)
    if np.any(abs(norms-1)>.01): raise ValueError('index is not approximately normalized')
    vectors=model.encode([c['query'] for c in conditions],prompt=QUERY_PROMPT,batch_size=8,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype('float32')
    scores,ids=index.search(vectors,1000)
    rankings=[]
    for c,cs,ci in zip(conditions,scores,ids):
        rankings.append({'condition_id':c['id'],'query':c['query'],'hits':[{'index_id':int(i),'image_id':Path(paths[i]).stem,'rank':rank,'score':float(s)} for rank,(s,i) in enumerate(zip(cs,ci),1) if i>=0]})
    write_jsonl(args.out/'selection/rankings.jsonl',rankings)
    write_json(args.out/'selection/retrieval.json',{'created_at':now(),'model':MODEL,'revision':REVISION,'image_prompt':'','query_prompt':QUERY_PROMPT,'index_sha256':source_hash,'paths_sha256':digest(ROOT/'data_100k/image_paths.json'),'catalog_sha256':digest(ROOT/'docs/search-condition-catalog.md'),'conditions':len(conditions),'index_count':index.ntotal,'dimension':index.d,'probe_ids':probe_ids,'probe_cosines':similarity.tolist(),'norm_range':[float(norms.min()),float(norms.max())],'verification_scope':'sampled content compatibility, not complete historical provenance','elapsed_seconds':time.monotonic()-started})
    print('retrieval complete',len(conditions),flush=True)


def stratified_hits(hits, seed, condition_id):
    result=[]
    for lower,upper in [(1,40),(41,200),(201,1000)]:
        pool=[h for h in hits if lower<=h['rank']<=upper]
        if lower!=1: pool=sorted(pool,key=lambda h:stable_key(seed,f"{condition_id}:{h['image_id']}"))
        result.extend(pool[:40])
    return result


def contact_sheets(rows, folder, prefix):
    from PIL import Image,ImageDraw
    folder.mkdir(parents=True,exist_ok=True)
    for page,start in enumerate(range(0,len(rows),12),1):
        canvas=Image.new('RGB',(4*400,3*255),'white'); draw=ImageDraw.Draw(canvas)
        for offset,row in enumerate(rows[start:start+12]):
            x=(offset%4)*400; y=(offset//4)*255
            with Image.open(ROOT/row['image_path']) as im:
                im=im.convert('RGB'); im.thumbnail((396,222)); canvas.paste(im,(x,y+30))
            draw.text((x+3,y+2),f"{start+offset+1:02} {row.get('target_condition','random')} {row['image_id']}",fill='black')
        canvas.save(folder/f'{prefix}_{page:02}.jpg')


def run_prepare(args):
    if (args.out/'manifests/pilot_50.jsonl').exists():
        raise ValueError('pilot already fixed; use a new output directory to change selection')
    started=time.monotonic()
    retrieval=json.loads((args.out/'selection/retrieval.json').read_text())
    inventory=json.loads((args.out/'inventory/config.json').read_text())
    if retrieval['paths_sha256']!=inventory['paths_sha256'] or retrieval['catalog_sha256']!=digest(ROOT/'docs/search-condition-catalog.md'):
        raise ValueError('retrieval provenance differs from inventory/catalog')
    if inventory['seed']!=args.seed:
        raise ValueError('selection seed differs from reserved random sample')
    reps=read_jsonl(args.out/'inventory/representatives.jsonl'); randoms=read_jsonl(args.out/'manifests/random_1000.jsonl')
    byid={r['image_id']:r for r in reps}
    reserved={r['group_id'] for r in randoms}; randomids={r['image_id'] for r in randoms}
    rankings=read_jsonl(args.out/'selection/rankings.jsonl')
    if len(rankings)!=227 or len(randoms)!=1000: raise ValueError('run inventory/retrieve first')
    sampled={r['condition_id']:stratified_hits(r['hits'],args.seed,r['condition_id']) for r in rankings}
    candidates={}
    for depth in range(120):
        for c,hits in sampled.items():
            if depth>=len(hits): continue
            hit=hits[depth]; row=byid.get(hit['image_id'])
            if row is None or row['group_id'] in reserved: continue
            if row['image_id'] not in candidates:
                if len(candidates)>=20000: continue
                candidates[row['image_id']]={**row,'hits':[]}
            candidates[row['image_id']]['hits'].append({'condition_id':c,**hit})
    write_jsonl(args.out/'selection/candidates.jsonl',candidates.values())
    # Pilot only: two independently reviewed candidates for each of 20 diverse queries.
    proposed=[]; used=set(randomids)
    ranks={r['condition_id']:r['hits'] for r in rankings}
    for cid in PILOT_CONDITIONS:
        pool=[h for h in ranks[cid] if h['image_id'] in byid and h['image_id'] not in used]
        if len(pool)<4: raise ValueError(f'insufficient pilot alternatives: {cid}')
        for hit in pool[:2]:
            row=byid[hit['image_id']]; used.add(row['image_id'])
            proposed.append({**row,'split':'train','selection_route':'condition','target_condition':cid,'retrieval_rank':hit['rank'],'retrieval_score':hit['score'],'seed':args.seed,'pilot':True})
    proposed.extend({**r,'pilot':True} for r in [r for r in randoms if r['split']=='train'][:10])
    if len(proposed)!=50: raise ValueError('pilot must be 40 condition + 10 random')
    write_jsonl(args.out/'pilot/proposed.jsonl',proposed)
    contact_sheets(proposed,args.out/'pilot/contact_sheets','proposed')
    # Near matches are review candidates, not automatic duplicate deletions.
    comparisons=[]
    for i,a in enumerate(proposed):
        for b in proposed[:i]+randoms:
            if a['image_id']==b['image_id']: continue
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance<=4:
                comparisons.append({'a':a['image_id'],'b':b['image_id'],'distance':distance,'b_split':b.get('split')})
    write_json(args.out/'pilot/selection_report.json',{'created_at':now(),'candidates':len(candidates),'proposed':len(proposed),'route_counts':dict(Counter(r['selection_route'] for r in proposed)),'near_duplicate_candidates':comparisons,'sequence_independence':'unverified','visual_review_required':True,'elapsed_seconds':time.monotonic()-started})
    print('pilot proposals ready: review contact sheets and supply pilot/review.jsonl',flush=True)


def finalize_pilot(args):
    rows=read_jsonl(args.out/'pilot/proposed.jsonl')
    reviews=read_jsonl(args.out/'pilot/review.jsonl')
    if len(rows)!=50 or len(reviews)!=50 or len({r['image_id'] for r in reviews})!=50:
        raise ValueError('exactly 50 unique visual reviews required')
    byid={r['image_id']:r for r in reviews}
    if set(byid)!={r['image_id'] for r in rows}: raise ValueError('review/image mismatch')
    if any(r['split']!='train' for r in rows) or Counter(r['selection_route'] for r in rows)!={'condition':40,'random':10}:
        raise ValueError('pilot must contain 40 condition + 10 random images, all in train')
    reserved={r['image_id']:r for r in read_jsonl(args.out/'manifests/random_1000.jsonl')}
    for row in rows:
        original=reserved.get(row['image_id'])
        if row['selection_route']=='random':
            if original is None or original['split']!='train' or original['group_id']!=row['group_id']:
                raise ValueError('pilot random image is not reserved for train')
        elif original is not None:
            raise ValueError('condition image overlaps reserved random sample')
    groups=set()
    for row in rows:
        review=byid[row['image_id']]
        if review.get('decision')!='accept' or not review.get('reviewer') or not review.get('notes'):
            raise ValueError(f"unreviewed proposal: {row['image_id']}")
        if row['group_id'] in groups: raise ValueError('duplicate group in pilot')
        groups.add(row['group_id'])
        if digest(ROOT/row['image_path'])!=row['sha256']: raise ValueError('image content changed')
        row['selection_review']=review
    manifest=args.out/'manifests/pilot_50.jsonl'
    if manifest.exists() and read_jsonl(manifest)!=rows: raise ValueError('pilot manifest already fixed; use a new run for changes')
    write_jsonl(manifest,rows)
    return rows


def request_json(url,payload=None,timeout=300):
    req=urllib.request.Request(url,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def run_annotate(args):
    rows=finalize_pilot(args); conditions=load_conditions()
    tags=request_json(args.ollama_url+'/api/tags')
    matches=[m for m in tags['models'] if m['name'].lower()==args.teacher.lower()]
    if len(matches)!=1: raise ValueError('teacher model not installed or ambiguous')
    teacher=matches[0]
    config={'version':VERSION,'teacher_name':teacher['name'],'teacher_digest':teacher['digest'],'manifest_sha256':digest(args.out/'manifests/pilot_50.jsonl'),'prompt_sha256':hashlib.sha256(FACT_PROMPT.encode()).hexdigest(),'schema_sha256':hashlib.sha256(json.dumps(fact_schema(),sort_keys=True).encode()).hexdigest(),'catalog_sha256':digest(ROOT/'docs/search-condition-catalog.md'),'temperature':0,'think':False,'num_ctx':8192,'num_predict':4096}
    config_path=args.out/'pilot/annotation_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config: raise ValueError('annotation config changed; new run required')
    write_json(config_path,config)
    write_json(args.out/'schema/facts.schema.json',fact_schema())
    write_json(args.out/'schema/conditions.json',conditions)
    (args.out/'schema/prompt.txt').write_text(FACT_PROMPT)
    path=args.out/'pilot/facts.jsonl'; done_rows=read_jsonl(path)
    done={r['image_id'] for r in done_rows}
    if len(done)!=len(done_rows) or not done<={r['image_id'] for r in rows}: raise ValueError('invalid existing facts')
    for r in done_rows: validate_facts(r['facts'])
    started=time.monotonic(); errors=0
    for n,row in enumerate(rows,1):
        if row['image_id'] in done: continue
        if digest(ROOT/row['image_path'])!=row['sha256']: raise ValueError('image content changed')
        encoded=base64.b64encode((ROOT/row['image_path']).read_bytes()).decode()
        for attempt in range(1,3):
            start=time.monotonic()
            payload={'model':teacher['name'],'stream':False,'think':False,'format':fact_schema(),'options':{'temperature':0,'num_ctx':8192,'num_predict':4096},'messages':[{'role':'user','content':FACT_PROMPT,'images':[encoded]}]}
            if attempt>1: payload['messages'][0]['content']+='\n前回の出力に形式または整合性エラーがありました。再確認して完全なJSONを出力してください。'+str(error)
            try:
                response=request_json(args.ollama_url+'/api/chat',payload,args.timeout)
                append(args.out/'pilot/raw_responses.jsonl',{'image_id':row['image_id'],'attempt':attempt,'created_at':now(),'elapsed_seconds':time.monotonic()-start,'request_prompt':payload['messages'][0]['content'],'response':response})
                if response.get('done_reason')!='stop':
                    raise ValueError('teacher output did not finish normally')
                facts=validate_facts(json.loads(response['message']['content']))
            except Exception as exc:
                error=exc; errors+=1
                append(args.out/'pilot/errors.jsonl',{'image_id':row['image_id'],'attempt':attempt,'error':str(exc),'created_at':now(),'elapsed_seconds':time.monotonic()-start})
                print(f'[{n}/50] attempt {attempt} failed: {exc}',flush=True)
                if attempt==2: break
                continue
            record={'version':VERSION,'image_id':row['image_id'],'image_path':row['image_path'],'split':row['split'],'selection_route':row['selection_route'],'created_at':now(),'elapsed_seconds':time.monotonic()-start,'teacher_name':teacher['name'],'teacher_digest':teacher['digest'],'facts':facts,'review_status':'teacher_only'}
            append(path,record); done.add(row['image_id'])
            print(f'[{n}/50] {row["image_id"]} objects={len(facts["objects"])} seconds={record["elapsed_seconds"]:.1f}',flush=True)
            break
    append(args.out/'pilot/annotation_runs.jsonl',{'created_at':now(),'elapsed_seconds':time.monotonic()-started,'completed_total':len(done),'errors_this_run':errors})
    run_report(args)
    if len(done)!=50: raise RuntimeError(f'pilot incomplete: {len(done)}/50')


def run_report(args):
    conditions=load_conditions(); facts=read_jsonl(args.out/'pilot/facts.jsonl')
    labels=[]; counts={c['id']:Counter() for c in conditions}; flagged={}
    quality_reviews=read_jsonl(args.out/'pilot/quality_reviews.jsonl')
    fact_ids={r['image_id'] for r in facts}
    condition_ids={c['id'] for c in conditions}
    for review in quality_reviews:
        if review['image_id'] not in fact_ids or review['condition_id'] not in condition_ids:
            raise ValueError('quality review refers to an unknown image/condition')
        if not review.get('reviewer') or not review.get('notes') or review.get('verdict') not in ('consistent','needs_review'):
            raise ValueError('invalid quality review')
    manual_issues={r['image_id'] for r in quality_reviews if r['verdict']=='needs_review'}
    heldout=[c['expression'] for c in conditions if c['id'].startswith('H')]
    for row in facts:
        validate_facts(row['facts'])
        flags=quality_flags(row['facts'])
        if row['image_id'] in manual_issues: flags.append('visual_review_issue')
        if flags: flagged[row['image_id']]=flags
        for c in conditions:
            value=evaluate(c['expression'],row['facts'])
            counts[c['id']][value]+=1
            excluded=c['expression']['op']=='deferred' or (row['split'] in ('train','val') and any(contains_heldout(c['expression'],h) for h in heldout))
            labels.append({'image_id':row['image_id'],'condition_id':c['id'],'split':row['split'],'judgment':value,'usage':'excluded' if excluded else 'review_required' if flags else 'unknown' if value=='unknown' else 'teacher_candidate','review_status':'teacher_only','quality_flags':flags})
    write_jsonl(args.out/'pilot/condition_judgments.jsonl',labels)
    times=[r['elapsed_seconds'] for r in facts]
    report={'created_at':now(),'facts_count':len(facts),'expected':50,'complete':len(facts)==50,'schema_version':VERSION,'teacher_only':True,'pair_exported':False,'conditions_with_positive':sum(v['yes']>0 for k,v in counts.items() if not k.startswith('D')),'conditions_with_positive_and_negative':sum(v['yes']>0 and v['no']>0 for k,v in counts.items() if not k.startswith('D')),'judgments':dict(Counter(r['judgment'] for r in labels)),'condition_counts':{k:dict(v) for k,v in counts.items()},'successful_call_seconds':{'sum':sum(times),'median':statistics.median(times) if times else None,'mean':statistics.mean(times) if times else None,'max':max(times) if times else None},'errors':len(read_jsonl(args.out/'pilot/errors.jsonl')),'annotation_runs':read_jsonl(args.out/'pilot/annotation_runs.jsonl')}
    report['quality_flags_by_image']=flagged
    report['quality_reviews']=quality_reviews
    report['usage_counts']=dict(Counter(r['usage'] for r in labels))
    report['coverage_scope']='raw teacher-derived counts; flagged observations included, not verified coverage'
    report['report_implementation_sha256']={'condition_data.py':digest(ROOT/'scripts/condition_data.py'),'build_condition_dataset.py':digest(Path(__file__))}
    write_json(args.out/'pilot/report.json',report)
    print('report',report['facts_count'],report['conditions_with_positive'],'positive conditions',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['inventory','retrieve','prepare','annotate','report','annotate-full','report-full'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--seed',type=int,default=20260924)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--teacher',default='qwen3.8-27b-mtp:UD-Q3_K_XL')
    parser.add_argument('--ollama-url',default='http://localhost:11434')
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--check-only',action='store_true',help='read-only preflight for annotate-full; no inference')
    parser.add_argument('--no-reuse-pilot',action='store_true',help='generate all selected images without importing pilot labels')
    parser.add_argument('--num-predict',type=int,help='explicit output-token cap override for pending full annotations; saved per request')
    args=parser.parse_args(); args.out=args.out.resolve(); args.out.mkdir(parents=True,exist_ok=True)
    if args.check_only and args.stage!='annotate-full':
        parser.error('--check-only is only supported for annotate-full')
    if args.no_reuse_pilot and args.stage!='annotate-full':
        parser.error('--no-reuse-pilot is only supported for annotate-full')
    if args.num_predict is not None and (args.stage!='annotate-full' or args.num_predict<=0):
        parser.error('--num-predict must be positive and is only supported for annotate-full')
    # Kernel lock is released automatically on termination; no stale PID lock files.
    with (args.out/'.pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage in ('annotate-full','report-full'):
            try:
                from .annotate_condition_dataset import run_annotate_full, run_report_full
            except ImportError:
                from annotate_condition_dataset import run_annotate_full, run_report_full
            (run_annotate_full if args.stage=='annotate-full' else run_report_full)(args)
        else:
            globals()['run_'+args.stage](args)


if __name__=='__main__': main()
