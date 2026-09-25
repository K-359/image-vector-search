import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.build_condition_dataset import run_annotate
from scripts.condition_data import SCENES, INVENTORY_KINDS, digest, write_jsonl


class AnnotationResumeTest(unittest.TestCase):
    def test_completed_run_skips_teacher_images_and_changed_config_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            rows=[{'image_id':str(i),'split':'train'} for i in range(50)]
            write_jsonl(out/'manifests/pilot_50.jsonl',rows)
            empty={'scene':{s:'unknown' for s in SCENES},'scene_evidence':'不明','objects':[],
                   'inventory':{k:{'presence':'unknown','complete':False} for k in INVENTORY_KINDS},'visibility_notes':'不明'}
            records=[{**r,'facts':empty,'elapsed_seconds':1} for r in rows]
            path=out/'pilot/facts.jsonl'; write_jsonl(path,records)
            write_jsonl(out/'pilot/quality_reviews.jsonl',[{'image_id':'0','condition_id':'OBJ01','verdict':'needs_review','reviewer':'test_fixture','notes':'synthetic quality issue'}])
            before=digest(path)
            args=SimpleNamespace(out=out,ollama_url='http://unused',teacher='fixture',timeout=1)
            with patch('scripts.build_condition_dataset.finalize_pilot',return_value=rows), patch('scripts.build_condition_dataset.request_json',return_value={'models':[{'name':'fixture','digest':'fixed'}]}) as request, contextlib.redirect_stdout(io.StringIO()):
                run_annotate(args)
                self.assertEqual(request.call_count,1)
                self.assertTrue(request.call_args.args[0].endswith('/api/tags'))
                config=out/'pilot/annotation_config.json'; data=json.loads(config.read_text())
                data['prompt_sha256']='incompatible'; config.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'config changed'): run_annotate(args)
            self.assertEqual(digest(path),before)
            self.assertEqual(len((out/'pilot/condition_judgments.jsonl').read_text().splitlines()),50*233)
            self.assertFalse((out/'pairs.train.jsonl').exists())
            report=json.loads((out/'pilot/report.json').read_text())
            self.assertIn('visual_review_issue',report['quality_flags_by_image']['0'])


if __name__=='__main__': unittest.main()
