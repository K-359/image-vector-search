"""Resumable annotation of the fixed 5,000-image manifest; no training side effects."""
from __future__ import annotations

import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error

try:
    from .condition_data import (ROOT, VERSION, FACT_PROMPT, digest, fact_schema,
        load_conditions, read_jsonl, validate_schema, write_json, write_jsonl,
        evaluate, contains_heldout)
    from .build_condition_dataset import append, now, request_json
except ImportError:
    from condition_data import (ROOT, VERSION, FACT_PROMPT, digest, fact_schema,
        load_conditions, read_jsonl, validate_schema, write_json, write_jsonl,
        evaluate, contains_heldout)
    from build_condition_dataset import append, now, request_json


def parse_facts(response):
    """Check readable output structure only; never reject semantic inconsistencies."""
    if response.get('done_reason')!='stop':
        raise ValueError('teacher output did not finish normally')
    facts=json.loads(response['message']['content'])
    validate_schema(facts,fact_schema())
    return facts


def read_log(path, *, repair=False):
    """Recover only an interrupted final write; never discard complete corrupt rows."""
    if not path.exists():
        return []
    rows=[]
    with path.open('rb') as f:
        while line:=f.readline():
            end=f.tell()
            if not line.endswith(b'\n'):
                if not repair:
                    raise ValueError(f'incomplete final write: {path}; resume annotation to recover')
                # Preserve even a syntactically valid fragment, as it was not committed.
                backup=path.with_name(path.name+f'.interrupted-{time.time_ns()}')
                backup.write_bytes(line)
                with path.open('r+b') as dest:
                    dest.truncate(end-len(line)); dest.flush(); os.fsync(dest.fileno())
                print(f'Recovered interrupted final write: {path.name}; saved {backup.name}',flush=True)
                break
            rows.append(json.loads(line))
    return rows


def load_manifest(out):
    path=out/'manifests/selected_5000.jsonl'
    rows=read_jsonl(path)
    if len(rows)!=5000 or len({r['image_id'] for r in rows})!=5000:
        raise ValueError('expected 5,000 unique selected images')
    if len({r['group_id'] for r in rows})!=5000 or len({r['sha256'] for r in rows})!=5000:
        raise ValueError('duplicate image content or group')
    expected={('train','condition'):2800,('train','random'):700,
              ('val','condition'):400,('val','random'):100,
              ('test','condition'):800,('test','random'):200}
    if Counter((r['split'],r['selection_route']) for r in rows)!=expected:
        raise ValueError('selected split/route allocation changed')
    allocation=json.loads((out/'selection/allocation_report.json').read_text())
    if allocation['manifest_sha256']!=digest(path):
        raise ValueError('selected manifest differs from allocation report')
    return rows


def validate_records(records, rows, config):
    byid={r['image_id']:r for r in rows}
    seen=set()
    for record in records:
        if config.get('reuse_pilot') is False and record.get('reused_from')=='pilot/facts.jsonl':
            raise ValueError('pilot labels are forbidden in this annotation run')
        image_id=record['image_id']
        if image_id in seen or image_id not in byid:
            raise ValueError('duplicate or unknown image in annotation facts')
        seen.add(image_id)
        row=byid[image_id]
        for key in ('split','image_path','selection_route'):
            if record[key]!=row[key]:
                raise ValueError(f'annotation/manifest mismatch: {image_id} {key}')
        for key in ('version','teacher_name','teacher_digest'):
            if record[key]!=config[key]:
                raise ValueError(f'annotation/config mismatch: {image_id} {key}')
        if record.get('image_sha256',row['sha256'])!=row['sha256']:
            raise ValueError('annotation source hash mismatch')
        validate_schema(record['facts'],fact_schema())
    return seen


def prepare_annotation(args):
    """Read-only preflight, including teacher identity and reusable pilot provenance."""
    rows=load_manifest(args.out)
    conditions=load_conditions()
    tags=request_json(args.ollama_url+'/api/tags',timeout=args.timeout)
    matches=[m for m in tags['models'] if m['name'].lower()==args.teacher.lower()]
    if len(matches)!=1:
        raise ValueError('teacher model not installed or ambiguous')
    teacher=matches[0]
    common={'version':VERSION,'teacher_name':teacher['name'],'teacher_digest':teacher['digest'],
            'prompt_sha256':hashlib.sha256(FACT_PROMPT.encode()).hexdigest(),
            'schema_sha256':hashlib.sha256(json.dumps(fact_schema(),sort_keys=True).encode()).hexdigest(),
            'catalog_sha256':digest(ROOT/'docs/search-condition-catalog.md'),
            'temperature':0,'think':False,'num_ctx':8192,'num_predict':4096}
    config_path=args.out/'annotations/annotation_config.json'
    previous=json.loads(config_path.read_text()) if config_path.exists() else None
    reuse_pilot=not getattr(args,'no_reuse_pilot',False)
    if previous is not None and previous.get('reuse_pilot') is False:
        reuse_pilot=False  # A resumed clean run must never import the old pilot.
    pilot=[]; pilot_provenance={}
    if reuse_pilot:
        pilot_config=json.loads((args.out/'pilot/annotation_config.json').read_text())
        if any(pilot_config.get(k)!=v for k,v in common.items()):
            raise ValueError('pilot teacher/prompt/schema changed; cannot reuse its 50 facts')
        if pilot_config['manifest_sha256']!=digest(args.out/'manifests/pilot_50.jsonl'):
            raise ValueError('pilot manifest changed')
        pilot_rows=read_jsonl(args.out/'manifests/pilot_50.jsonl')
        pilot=read_jsonl(args.out/'pilot/facts.jsonl')
        if len(pilot_rows)!=50 or len(pilot)!=50:
            raise ValueError('expected 50 completed pilot facts')
        validate_records(pilot,pilot_rows,common)
        validate_records(pilot,rows,common)
        pilot_provenance={'pilot_facts_sha256':digest(args.out/'pilot/facts.jsonl'),
                          'pilot_config_sha256':digest(args.out/'pilot/annotation_config.json')}
    else:
        pilot_provenance={'reuse_pilot':False}
    config={**common,'manifest_sha256':digest(args.out/'manifests/selected_5000.jsonl'),
            **pilot_provenance,
            'allocation_sha256':digest(ROOT/'docs/search-condition-allocation.csv'),
            'quality_policy':'diagnostic_only', 'max_attempts_per_image_per_run':2,
            'content_consistency_validation':'disabled'}
    if previous is not None:
        # Explicitly allow only removal of the old content gate, with generation
        # settings and every source fingerprint unchanged. Record migration below.
        legacy={k:v for k,v in config.items() if k!='content_consistency_validation'}
        if previous not in (config,legacy):
            raise ValueError('full annotation config changed; refusing to mix runs')
    facts_path=args.out/'annotations/facts.jsonl'
    if facts_path.exists() and not config_path.exists():
        raise ValueError('existing full facts without annotation config')
    def verify(row):
        if digest(ROOT/row['image_path'])!=row['sha256']:
            raise ValueError(f"source image changed: {row['image_id']}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(verify,rows))
    return rows,conditions,pilot,config


def run_annotate_full(args):
    rows,conditions,pilot,config=prepare_annotation(args)
    generation_options={k:config[k] for k in ('temperature','num_ctx','num_predict')}
    override=getattr(args,'num_predict',None)
    if override is not None:
        if type(override) is not int or override<=0:
            raise ValueError('output-token cap must be a positive integer')
        generation_options['num_predict']=override
    dest=args.out/'annotations'
    check_only=getattr(args,'check_only',False)
    done_rows=read_log(dest/'facts.jsonl',repair=not check_only)
    done=validate_records(done_rows,rows,config)
    reusable=[r for r in pilot if r['image_id'] not in done]
    print(f'Preflight OK: selected={len(rows)} saved={len(done)} reusable_pilot={len(reusable)} '
          f'pending={len(rows)-len(done)-len(reusable)} teacher={config["teacher_name"]} '
          f'output_token_cap={generation_options["num_predict"]}',flush=True)
    if check_only:
        print('Check only: no inference or annotation writes.',flush=True)
        return
    for name in ('raw_responses','errors','annotation_runs','config_changes'):
        read_log(dest/f'{name}.jsonl',repair=True)
    config_path=dest/'annotation_config.json'
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        if previous!=config:
            write_json(dest/'annotation_config.before_consistency_removal.json',previous)
            append(dest/'config_changes.jsonl',{'created_at':now(),
                   'reason':'User requested no content consistency checks; keep saved facts and accept prior readable responses.',
                   'previous':previous,'current':config})
    write_json(config_path,config)
    # Save source snapshots without treating code-only/report changes as new labels.
    sources=[ROOT/'scripts/annotate_condition_dataset.py',ROOT/'scripts/build_condition_dataset.py',ROOT/'scripts/condition_data.py']
    code_hashes={p.name:digest(p) for p in sources}
    source_dir=args.out/'provenance/full_annotation'/hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()[:16]
    source_dir.mkdir(parents=True,exist_ok=True)
    for p in sources:
        shutil.copyfile(p,source_dir/p.name)
    write_json(dest/'facts.schema.json',fact_schema())
    write_json(dest/'conditions.json',conditions)
    (dest/'prompt.txt').write_text(FACT_PROMPT)
    byid={r['image_id']:r for r in rows}
    for record in reusable:
        append(dest/'facts.jsonl',{**record,'image_sha256':byid[record['image_id']]['sha256'],
               'reused_from':'pilot/facts.jsonl'})
        done.add(record['image_id'])
    # Reuse responses formerly rejected only by semantic checks, including output
    # written just before interruption. Do not alter already accepted records.
    recovered=0
    for saved in read_log(dest/'raw_responses.jsonl'):
        image_id=saved['image_id']
        if image_id in done or image_id not in byid:
            continue
        try:
            facts=parse_facts(saved['response'])
        except (ValueError,KeyError):
            continue
        row=byid[image_id]
        append(dest/'facts.jsonl',{'version':VERSION,'image_id':image_id,
               'image_path':row['image_path'],'image_sha256':row['sha256'],
               'split':row['split'],'selection_route':row['selection_route'],
               'created_at':saved['created_at'],'elapsed_seconds':saved['elapsed_seconds'],
               'teacher_name':config['teacher_name'],'teacher_digest':config['teacher_digest'],
               'facts':facts,'review_status':'teacher_only','reused_from':'annotations/raw_responses.jsonl',
               'generation_options':saved.get('request_options',{k:config[k] for k in ('temperature','num_ctx','num_predict')}),
               'output_tokens':saved['response'].get('eval_count'),
               'recovered_at':now(),'recovered_attempt':saved['attempt']})
        done.add(image_id); recovered+=1
    print(f'Content consistency checks disabled. Recovered {recovered} saved responses; '
          f'{len(done)}/{len(rows)} complete, {len(rows)-len(done)} pending.',flush=True)
    started=time.monotonic(); errors=0; generated=0; status='interrupted'
    try:
        for row in rows:
            if row['image_id'] in done:
                continue
            raw=(ROOT/row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('source image changed after preflight')
            encoded=base64.b64encode(raw).decode()
            error=None
            for attempt in range(1,3):
                prompt=FACT_PROMPT
                if error is not None:
                    prompt+='\n前回の出力をJSONとして読み込めませんでした。指定のJSON形式で完全な出力を返してください。'+str(error)
                payload={'model':config['teacher_name'],'stream':False,'think':False,'format':fact_schema(),
                         'options':dict(generation_options),
                         'messages':[{'role':'user','content':prompt,'images':[encoded]}]}
                start=time.monotonic()
                try:
                    response=request_json(args.ollama_url+'/api/chat',payload,args.timeout)
                    append(dest/'raw_responses.jsonl',{'image_id':row['image_id'],'attempt':attempt,'created_at':now(),
                           'elapsed_seconds':time.monotonic()-start,'request_prompt':prompt,
                           'request_options':dict(generation_options),'response':response})
                    facts=parse_facts(response)
                except (ValueError,KeyError,urllib.error.URLError,TimeoutError) as exc:
                    error=exc; errors+=1
                    append(dest/'errors.jsonl',{'image_id':row['image_id'],'attempt':attempt,'error':str(exc),
                           'created_at':now(),'elapsed_seconds':time.monotonic()-start})
                    print(f'[{len(done)}/{len(rows)}] {row["image_id"]} attempt {attempt} failed: {exc}',flush=True)
                    if attempt==2 and isinstance(exc,(urllib.error.URLError,TimeoutError)):
                        raise RuntimeError('Ollama unavailable; restart with the same command after recovery') from exc
                    continue
                record={'version':VERSION,'image_id':row['image_id'],'image_path':row['image_path'],
                        'image_sha256':row['sha256'],'split':row['split'],'selection_route':row['selection_route'],
                        'created_at':now(),'elapsed_seconds':time.monotonic()-start,
                        'teacher_name':config['teacher_name'],'teacher_digest':config['teacher_digest'],
                        'facts':facts,'review_status':'teacher_only',
                        'generation_options':dict(generation_options),'output_tokens':response.get('eval_count')}
                append(dest/'facts.jsonl',record)
                done.add(row['image_id']); generated+=1
                elapsed=time.monotonic()-started
                eta=elapsed/generated*(len(rows)-len(done))/3600
                print(f'[{len(done)}/{len(rows)}] {row["image_id"]} seconds={record["elapsed_seconds"]:.1f} '
                      f'elapsed={elapsed/3600:.2f}h ETA={eta:.2f}h errors={errors}',flush=True)
                break
        status='complete' if len(done)==len(rows) else 'incomplete'
    finally:
        append(dest/'annotation_runs.jsonl',{'created_at':now(),'status':status,
               'elapsed_seconds':time.monotonic()-started,'completed_total':len(done),
               'generated_this_run':generated,'pilot_reused_this_run':len(reusable),
               'saved_responses_recovered_this_run':recovered,
               'generation_options':dict(generation_options),
               'errors_this_run':errors,'implementation_sha256':code_hashes})
    run_report_full(args)
    if status!='complete':
        raise RuntimeError(f'annotation incomplete: {len(done)}/{len(rows)}; rerun the same command for missing images')


def run_report_full(args):
    rows=load_manifest(args.out); dest=args.out/'annotations'
    config=json.loads((dest/'annotation_config.json').read_text())
    if config['manifest_sha256']!=digest(args.out/'manifests/selected_5000.jsonl'):
        raise ValueError('full annotation manifest changed')
    if config['catalog_sha256']!=digest(ROOT/'docs/search-condition-catalog.md') or config['allocation_sha256']!=digest(ROOT/'docs/search-condition-allocation.csv'):
        raise ValueError('condition catalog/allocation changed')
    conditions=load_conditions(); facts=read_log(dest/'facts.jsonl')
    validate_records(facts,rows,config)
    heldout=[c['expression'] for c in conditions if c['id'].startswith('H')]
    excluded={c['id']:any(contains_heldout(c['expression'],h) for h in heldout) for c in conditions}
    manual_issues=set(); existing_flags={}
    if config.get('reuse_pilot') is not False:
        manual_issues={r['image_id'] for r in read_jsonl(args.out/'pilot/quality_reviews.jsonl') if r['verdict']=='needs_review'}
        pilot_report=args.out/'pilot/report.json'
        existing_flags=json.loads(pilot_report.read_text()).get('quality_flags_by_image',{}) if pilot_report.exists() else {}
    counts={(s,c['id']):Counter() for s in ('train','val','test') for c in conditions}
    usage=Counter(); flagged={}
    def judgments():
        for row in facts:
            flags=list(existing_flags.get(row['image_id'],[]))
            if row['image_id'] in manual_issues and 'visual_review_issue' not in flags:
                flags.append('visual_review_issue')
            if flags:
                flagged[row['image_id']]=flags
            for c in conditions:
                value=evaluate(c['expression'],row['facts'])
                counts[(row['split'],c['id'])][value]+=1
                omit=c['expression']['op']=='deferred' or (row['split'] in ('train','val') and excluded[c['id']])
                use='excluded' if omit else 'unknown' if value=='unknown' else 'teacher_candidate'
                usage[use]+=1
                yield {'image_id':row['image_id'],'condition_id':c['id'],'split':row['split'],
                       'judgment':value,'usage':use,'review_status':'teacher_only','quality_flags':flags}
    # Stream 1,165,000 judgments instead of keeping them all in memory.
    write_jsonl(dest/'condition_judgments.jsonl',judgments())
    coverage=[]
    for split in ('train','val','test'):
        for c in conditions:
            count=counts[(split,c['id'])]
            omit=c['expression']['op']=='deferred' or (split in ('train','val') and excluded[c['id']])
            pt=int(c['allocation'][f'{split}_positive_target']); nt=int(c['allocation'][f'{split}_negative_target'])
            coverage.append({'split':split,'condition_id':c['id'],'positive':count['yes'],
                             'negative':count['no'],'unknown':count['unknown'],'excluded':omit,
                             'positive_target':pt,'negative_target':nt,
                             'positive_shortfall':None if omit else max(0,pt-count['yes']),
                             'negative_shortfall':None if omit else max(0,nt-count['no'])})
    report={'created_at':now(),'facts_count':len(facts),'expected':len(rows),'complete':len(facts)==len(rows),
            'output_token_cap_counts':dict(Counter(str(r.get('generation_options',{}).get('num_predict',config['num_predict'])) for r in facts)),
            'remaining_images':len(rows)-len(facts),'split_counts':dict(Counter(r['split'] for r in facts)),
            'teacher_only':True,'pair_exported':False,'quality_policy':'diagnostic_only',
            'content_consistency_validation':'disabled',
            'quality_flag_source':'none' if config.get('reuse_pilot') is False else 'previous pilot records only; no new content checks',
            'quality_flags_by_image':flagged,'usage_counts':dict(usage),'condition_counts':coverage,
            'coverage_scope':'teacher-derived counts, including flagged observations; not independently verified',
            'facts_sha256':digest(dest/'facts.jsonl'),'manifest_sha256':config['manifest_sha256'],
            'annotation_runs':read_log(dest/'annotation_runs.jsonl'),
            'report_implementation_sha256':{'annotate_condition_dataset.py':digest(Path(__file__)),
                                          'condition_data.py':digest(ROOT/'scripts/condition_data.py')}}
    write_json(dest/'report.json',report)
    print(f'Report saved: {len(facts)}/{len(rows)} images; {sum(usage.values())} condition judgments. '
          'No training started.',flush=True)
