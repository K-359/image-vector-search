import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import annotate_condition_dataset as full
from scripts.condition_data import ROOT, SCENES, INVENTORY_KINDS, digest, read_jsonl, write_json, write_jsonl


def facts():
    return {'scene':{s:'unknown' for s in SCENES},'scene_evidence':'不明','objects':[],
            'inventory':{k:{'presence':'unknown','complete':False} for k in INVENTORY_KINDS},
            'visibility_notes':'不明'}


class FullAnnotationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.out=self.root/'dataset'
        for name in ['scripts/annotate_condition_dataset.py','scripts/build_condition_dataset.py',
                     'scripts/condition_data.py','docs/search-condition-catalog.md','docs/search-condition-allocation.csv']:
            dest=self.root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        self.rows=[]
        for i in range(52):
            p=self.root/f'{i}.jpg'; p.write_bytes(str(i).encode())
            self.rows.append({'image_id':str(i),'image_path':p.name,'sha256':digest(p),
                              'split':'train','selection_route':'condition'})
        self.pilot=[{**r,'version':full.VERSION,'teacher_name':'fixture','teacher_digest':'fixed',
                     'facts':facts(),'elapsed_seconds':1,'review_status':'teacher_only'} for r in self.rows[:50]]
        write_jsonl(self.out/'manifests/pilot_50.jsonl',self.rows[:50])
        write_jsonl(self.out/'manifests/selected_5000.jsonl',self.rows)
        write_jsonl(self.out/'pilot/facts.jsonl',self.pilot)
        write_json(self.out/'pilot/report.json',{'quality_flags_by_image':{'0':['existing_pilot_issue']}})
        config={'version':full.VERSION,'teacher_name':'fixture','teacher_digest':'fixed',
                'manifest_sha256':digest(self.out/'manifests/pilot_50.jsonl'),
                'prompt_sha256':full.hashlib.sha256(full.FACT_PROMPT.encode()).hexdigest(),
                'schema_sha256':full.hashlib.sha256(json.dumps(full.fact_schema(),sort_keys=True).encode()).hexdigest(),
                'catalog_sha256':digest(self.root/'docs/search-condition-catalog.md'),
                'temperature':0,'think':False,'num_ctx':8192,'num_predict':4096}
        write_json(self.out/'pilot/annotation_config.json',config)
        self.args=SimpleNamespace(out=self.out,teacher='fixture',ollama_url='http://unused',
                                  timeout=1,workers=2,check_only=False)
        self.addCleanup(patch.stopall)
        patch.object(full,'ROOT',self.root).start()
        patch.object(full,'load_manifest',return_value=self.rows).start()
        self.request=patch.object(full,'request_json',side_effect=self.response).start()
        self.stdout=io.StringIO()

    def response(self,url,*args,**kwargs):
        if url.endswith('/api/tags'):
            return {'models':[{'name':'fixture','digest':'fixed'}]}
        value=facts(); value['scene'].update(day='yes',night='yes')
        # Deliberately contradict the empty object list. This must never trigger
        # rejection, regeneration, or failure during resume/report generation.
        value['inventory']['car']={'presence':'yes','complete':True}
        return {'done_reason':'stop','message':{'content':json.dumps(value)}}

    def test_check_only_performs_no_generation_or_annotation_writes(self):
        self.args.check_only=True
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(self.request.call_count,1)
        self.assertTrue(self.request.call_args.args[0].endswith('/api/tags'))
        self.assertFalse((self.out/'annotations').exists())
        self.assertIn('reusable_pilot=50 pending=2',self.stdout.getvalue())

    def test_clean_run_generates_every_image_and_never_imports_pilot_on_resume(self):
        self.args.no_reuse_pilot=True
        # A clean run does not even need the old annotation directory.
        shutil.rmtree(self.out/'pilot')
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(sum(c.args[0].endswith('/api/chat') for c in self.request.call_args_list),52)
        saved=read_jsonl(self.out/'annotations/facts.jsonl')
        self.assertEqual(len(saved),52)
        self.assertFalse(any('reused_from' in r for r in saved))
        self.assertFalse(json.loads((self.out/'annotations/annotation_config.json').read_text())['reuse_pilot'])
        self.assertEqual(json.loads((self.out/'annotations/report.json').read_text())['quality_flags_by_image'],{})
        # Even the original command without a flag inherits the saved clean policy.
        self.args.no_reuse_pilot=False
        self.request.reset_mock()
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(self.request.call_count,1)
        self.assertEqual(read_jsonl(self.out/'annotations/facts.jsonl'),saved)

    def test_clean_mode_cannot_append_to_existing_mixed_run(self):
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.args.no_reuse_pilot=True
        with self.assertRaisesRegex(ValueError,'config changed'):
            full.run_annotate_full(self.args)

    def test_output_cap_override_only_affects_pending_requests_and_is_recorded(self):
        self.args.num_predict=8192
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        chats=[c for c in self.request.call_args_list if c.args[0].endswith('/api/chat')]
        self.assertEqual(len(chats),2)
        for c in chats:
            self.assertEqual(c.args[1]['options'],{'temperature':0,'num_ctx':8192,'num_predict':8192})
            self.assertEqual(c.args[1]['messages'][0]['content'],full.FACT_PROMPT)
        saved=read_jsonl(self.out/'annotations/facts.jsonl')
        self.assertTrue(all(r['generation_options']['num_predict']==8192 for r in saved[50:]))
        self.assertEqual(json.loads((self.out/'annotations/annotation_config.json').read_text())['num_predict'],4096)
        self.assertEqual(read_jsonl(self.out/'annotations/annotation_runs.jsonl')[-1]['generation_options']['num_predict'],8192)
        self.assertEqual(json.loads((self.out/'annotations/report.json').read_text())['output_token_cap_counts'],{'4096':50,'8192':2})

    def test_reuses_pilot_resumes_after_interrupt_and_keeps_flags_diagnostic(self):
        pilot_hash=digest(self.out/'pilot/facts.jsonl')
        calls=0
        def interrupted(url,*args,**kwargs):
            nonlocal calls
            if url.endswith('/api/chat'):
                calls+=1
                if calls==2:
                    raise KeyboardInterrupt()
            return self.response(url,*args,**kwargs)
        self.request.side_effect=interrupted
        with contextlib.redirect_stdout(self.stdout),self.assertRaises(KeyboardInterrupt):
            full.run_annotate_full(self.args)
        self.assertEqual(len(read_jsonl(self.out/'annotations/facts.jsonl')),51)
        self.assertEqual(read_jsonl(self.out/'annotations/annotation_runs.jsonl')[-1]['status'],'interrupted')
        self.request.reset_mock(); self.request.side_effect=self.response
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(sum(c.args[0].endswith('/api/chat') for c in self.request.call_args_list),1)
        self.assertEqual(digest(self.out/'pilot/facts.jsonl'),pilot_hash)
        saved=read_jsonl(self.out/'annotations/facts.jsonl')
        self.assertEqual(len(saved),52)
        self.assertEqual(sum(r.get('reused_from')=='pilot/facts.jsonl' for r in saved),50)
        judgments=read_jsonl(self.out/'annotations/condition_judgments.jsonl')
        self.assertEqual(len(judgments),52*233)
        flagged=[r for r in judgments if r['image_id']=='51']
        self.assertTrue(all(not r['quality_flags'] for r in flagged))
        self.assertTrue(any(r['image_id']=='0' and r['quality_flags']==['existing_pilot_issue'] for r in judgments))
        self.assertFalse(any(r['usage']=='review_required' for r in judgments))
        self.assertTrue(all(r['usage']=='excluded' for r in flagged if r['condition_id'].startswith(('H','D'))))
        self.assertFalse((self.out/'pairs.train.jsonl').exists())
        self.request.reset_mock()
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(self.request.call_count,1)
        self.assertTrue(self.request.call_args.args[0].endswith('/api/tags'))

    def test_changed_configuration_or_source_rejected_before_chat(self):
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        p=self.out/'annotations/annotation_config.json'; config=json.loads(p.read_text())
        config['teacher_digest']='changed'; write_json(p,config)
        self.request.reset_mock()
        with self.assertRaisesRegex(ValueError,'config changed'):
            full.run_annotate_full(self.args)
        self.assertEqual(self.request.call_count,1)
        config['teacher_digest']='fixed'; write_json(p,config)
        (self.root/'51.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source image changed'):
            full.run_annotate_full(self.args)

    def test_failed_image_is_left_pending_and_retried_next_run(self):
        chats=0
        def invalid_first_image(url,*args,**kwargs):
            nonlocal chats
            if url.endswith('/api/chat'):
                chats+=1
                if chats<=2:
                    return {'done_reason':'length','message':{'content':'{}'}}
            return self.response(url,*args,**kwargs)
        self.request.side_effect=invalid_first_image
        with contextlib.redirect_stdout(self.stdout),self.assertRaisesRegex(RuntimeError,'incomplete'):
            full.run_annotate_full(self.args)
        self.assertEqual(len(read_jsonl(self.out/'annotations/facts.jsonl')),51)
        self.assertEqual(len(read_jsonl(self.out/'annotations/errors.jsonl')),2)
        self.request.side_effect=self.response; self.request.reset_mock()
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(sum(c.args[0].endswith('/api/chat') for c in self.request.call_args_list),1)

    def test_legacy_gate_migration_recovers_rejected_raw_without_inference(self):
        config=full.prepare_annotation(self.args)[3]
        config.pop('content_consistency_validation')
        write_json(self.out/'annotations/annotation_config.json',config)
        write_jsonl(self.out/'annotations/facts.jsonl',self.pilot)
        write_jsonl(self.out/'annotations/raw_responses.jsonl',[
            {'image_id':r['image_id'],'attempt':1,'created_at':'fixture','elapsed_seconds':1,
             'response':self.response('/api/chat')} for r in self.rows[50:]])
        self.request.reset_mock()
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(self.request.call_count,1)
        saved=read_jsonl(self.out/'annotations/facts.jsonl')
        self.assertEqual(len(saved),52)
        self.assertEqual(saved[:50],self.pilot)
        self.assertTrue(all(r['reused_from']=='annotations/raw_responses.jsonl' for r in saved[50:]))
        self.assertEqual(len(read_jsonl(self.out/'annotations/config_changes.jsonl')),1)
        self.assertEqual(json.loads((self.out/'annotations/annotation_config.json').read_text())['content_consistency_validation'],'disabled')
        self.assertEqual(json.loads((self.out/'annotations/report.json').read_text())['facts_count'],52)
        self.request.reset_mock()
        with contextlib.redirect_stdout(self.stdout):
            full.run_annotate_full(self.args)
        self.assertEqual(self.request.call_count,1)
        self.assertEqual(len(read_jsonl(self.out/'annotations/facts.jsonl')),52)

    def test_semantic_anomalies_are_saved_as_returned(self):
        value=facts()
        object={'id':'duplicate','kind':'car','bbox':[800,700,200,100],
                'color':'unknown','orientation':'unknown','lane':'unknown',
                'sidewalk':'unknown','on_crosswalk':'unknown','roadway':'unknown',
                'emergency_vehicle':'unknown','evidence':'fixture'}
        value['objects']=[object,dict(object)]
        value['inventory']['car']={'presence':'no','complete':False}
        response={'done_reason':'stop','message':{'content':json.dumps(value)}}
        self.assertEqual(full.parse_facts(response),value)


class LogRecoveryTest(unittest.TestCase):
    def test_only_incomplete_tail_is_recovered_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'facts.jsonl'; p.write_bytes(b'{"ok":1}\n{"partial":')
            with self.assertRaisesRegex(ValueError,'incomplete final write'):
                full.read_log(p)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(full.read_log(p,repair=True),[{'ok':1}])
            self.assertEqual(next(p.parent.glob('*.interrupted-*')).read_bytes(),b'{"partial":')
            self.assertEqual(p.read_bytes(),b'{"ok":1}\n')
            p.write_bytes(b'{"ok":1}\ninvalid\n')
            with self.assertRaises(ValueError):
                full.read_log(p,repair=True)
            self.assertEqual(p.read_bytes(),b'{"ok":1}\ninvalid\n')


if __name__=='__main__':
    unittest.main()
