import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_condition_pairs import build_pairs, load_judgments
from scripts.condition_data import write_jsonl


def condition(cid,expr=None,positive=2,negative=2):
    return {'id':cid,'query':cid,'expression':expr or {'op':'exists','objects':[{'kind':'car'}]},
            'allocation':{'category':'basic',**{f'{s}_{label}_target':str(v)
                for s in ('train','val','test') for label,v in [('positive',positive),('negative',negative)]}}}


def image(i,split='train'):
    return {'image_id':i,'image_path':i+'.jpg','sha256':i,'group_id':i,
            'split':split,'selection_route':'random'}


class PairExportTest(unittest.TestCase):
    def test_sampling_uses_saved_labels_keeps_unknown_out_and_is_reproducible(self):
        rows=[image(i) for i in ['p1','p2','p3','n1','n2','n3','u']]
        c=condition('OBJ01'); pool={('train','OBJ01'):{'yes':['p1','p2','p3'],'no':['n1','n2','n3'],'unknown':['u']}}
        ranks=[{'condition_id':'OBJ01','hits':[{'image_id':'u','rank':1},{'image_id':'p1','rank':2},{'image_id':'n1','rank':3}]}]
        pairs,queries,coverage=build_pairs(rows,[c],pool,ranks)
        repeat=build_pairs(list(reversed(rows)),[c],{('train','OBJ01'):{k:list(reversed(v)) for k,v in pool[('train','OBJ01')].items()}},ranks)
        self.assertEqual((pairs,queries,coverage),repeat)
        selected=pairs['train']
        self.assertEqual(len(selected),4)
        self.assertEqual(sum(r['label'] for r in selected),2)
        self.assertNotIn('u',{r['image_id'] for r in selected})
        self.assertEqual([r['image_id'] for r in selected if r['negative_type']=='hard_negative'],['n1'])
        self.assertEqual(next(r for r in coverage if r['split']=='train')['available_positive'],3)
        self.assertTrue(all(r['caption'] is None and 'source_image_id' not in r for r in selected))

    def test_holdout_and_stronger_queries_excluded_from_train_and_val(self):
        h={'op':'exists','objects':[{'kind':'car','color':'red'}]}
        strong={'op':'exists','objects':[{'kind':'car','color':'red','position':'left'}]}
        conditions=[condition('H01',h),condition('CA01',strong),condition('D01',{'op':'deferred'})]
        rows=[image(f'{s}{i}',s) for s in ('train','val','test') for i in ('p','n')]
        pools={(s,c['id']):{'yes':[f'{s}p'],'no':[f'{s}n']} for s in ('train','val','test') for c in conditions}
        pairs,queries,coverage=build_pairs(rows,conditions,pools,[])
        self.assertFalse(pairs['train']);self.assertFalse(pairs['val'])
        self.assertEqual({r['condition_id'] for r in pairs['test']},{'H01','CA01'})
        self.assertNotIn('D01',{r['condition_id'] for r in queries})

    def test_missing_positive_is_reported_not_exported_as_negative_only(self):
        rows=[image('n'),image('u')]
        pools={('train','OBJ01'):{'no':['n'],'unknown':['u']}}
        pairs,queries,coverage=build_pairs(rows,[condition('OBJ01')],pools,[])
        self.assertFalse(pairs['train']);self.assertFalse(queries)
        stat=next(r for r in coverage if r['split']=='train')
        self.assertEqual(stat['status'],'no_positive')
        self.assertEqual(stat['available_negative'],1)

    def test_few_examples_are_retained_with_shortfall_and_no_duplicates(self):
        rows=[image('p'),image('n')]
        pools={('train','OBJ01'):{'yes':['p'],'no':['n']}}
        pairs,queries,coverage=build_pairs(rows,[condition('OBJ01',positive=50,negative=20)],pools,[])
        self.assertEqual(len(pairs['train']),2)
        self.assertEqual(next(r for r in coverage if r['split']=='train')['status'],'below_target')

    def test_cross_split_pool_and_duplicate_content_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'candidate pool'):
            build_pairs([image('p','test')],[condition('OBJ01')],{('train','OBJ01'):{'yes':['p']}},[])
        with self.assertRaisesRegex(ValueError,'duplicate image or group'):
            build_pairs([image('p'),{**image('q','test'),'group_id':'p'}],[],{},[])

    def test_incomplete_duplicate_or_misassigned_judgment_file_rejected(self):
        rows=[image('p')];conditions=[condition('OBJ01')]
        valid={'image_id':'p','split':'train','condition_id':'OBJ01','judgment':'unknown','usage':'unknown'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'judgments.jsonl'
            for invalid in [[],[valid,valid],[{**valid,'split':'test'}],[{**valid,'usage':'teacher_candidate'}]]:
                write_jsonl(path,invalid)
                with self.assertRaises(ValueError):load_judgments(path,rows,conditions)
            write_jsonl(path,[valid])
            self.assertEqual(load_judgments(path,rows,conditions)[('train','OBJ01')]['unknown'],['p'])


if __name__=='__main__':
    unittest.main()
