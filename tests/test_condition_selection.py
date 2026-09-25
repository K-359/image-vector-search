import unittest
from scripts.select_condition_images import allocate_reviewed, allocate_retrieval


def row(id,split=None):
    r={'image_id':id,'group_id':id,'selection_route':'condition'}
    if split: r.update(split=split,selection_route='random')
    return r


def condition(id,train=1,val=1,test=1):
    return {'id':id,'allocation':{f'{s}_{label}_target':str(v if label=='positive' else 0) for s,v in [('train',train),('val',val),('test',test)] for label in ['positive','negative']}}


def review(id,judgments):
    return {'image_id':id,'decision':'accept','reviewer':'test_fixture','notes':'synthetic fixture','judgments':judgments}


class AllocationTest(unittest.TestCase):
    def test_holdout_test_priority_and_fixed_random_and_pilot(self):
        conditions=[condition('OBJ01'),condition('H01',0,0,1)]
        randoms=[row('random','train')]
        pilot=[{**row('pilot'),'split':'train','pilot':True}]
        candidates=[row('heldout'),row('basic'),row('fill')]
        reviews=[review('heldout',{'H01':'yes'}),review('basic',{'OBJ01':'yes'}),review('fill',{})]
        selected,report=allocate_reviewed(randoms,pilot,candidates,reviews,conditions,budget={'train':3,'val':1,'test':1})
        self.assertTrue(report['complete_image_budget'])
        self.assertEqual({r['image_id']:r['split'] for r in selected},{'random':'train','pilot':'train','heldout':'test','basic':'val','fill':'train'})

    def test_unreviewed_candidates_cannot_fill_budget(self):
        selected,report=allocate_reviewed([],[],[row('unknown')],[],[condition('OBJ01')],budget={'train':1,'val':0,'test':0})
        self.assertFalse(report['complete_image_budget'])
        self.assertEqual(selected,[])

    def test_group_never_spans_splits_and_shortage_is_explicit(self):
        candidates=[{**row('a'),'group_id':'same'},{**row('b'),'group_id':'same'}]
        selected,report=allocate_reviewed([],[],candidates,[review('a',{}),review('b',{})],[condition('OBJ01')],budget={'train':1,'val':0,'test':1})
        self.assertEqual(len(selected),1)
        self.assertFalse(report['complete_image_budget'])
        self.assertEqual(report['unfilled_image_slots']['train'],1)

    def test_unknown_judgments_do_not_satisfy_quota(self):
        _,report=allocate_reviewed([],[],[row('a')],[review('a',{'OBJ01':'unknown'})],[condition('OBJ01')],budget={'train':1,'val':0,'test':0})
        d=next(x for x in report['deficits'] if x['split']=='train')
        self.assertEqual(d['provisionally_observed'],0)

    def test_conflicting_pilot_reservation_rejected(self):
        with self.assertRaises(ValueError):
            allocate_reviewed([row('same','test')],[row('same','train')],[],[],[],budget={'train':1,'val':0,'test':1})


def candidate(id, cid='OBJ01', rank=1, group=None):
    return {**row(id), 'group_id': group or id, 'sha256': group or id,
            'hits': [{'condition_id': cid, 'image_id': id, 'rank': rank, 'score': .5}]}


class RetrievalAllocationTest(unittest.TestCase):
    def test_fixed_images_rank_bands_and_reproducibility_without_labels(self):
        randoms=[{**row('random','train'),'sha256':'random'}]
        pilot=[{**candidate('pilot'),'split':'train','pilot':True,
                'target_condition':'OBJ01','retrieval_rank':1,'retrieval_score':.5},
               {**randoms[0],'pilot':True}]
        candidates=[candidate('pilot')]+[
            candidate(f'{rank}_{n}',rank=rank) for rank in (2,41,201) for n in range(4)]
        kwargs={'budget':{'train':5,'val':0,'test':0}}
        selected,report=allocate_retrieval(randoms,pilot,candidates,[condition('OBJ01')],**kwargs)
        repeated,_=allocate_retrieval(randoms,pilot,list(reversed(candidates)),[condition('OBJ01')],**kwargs)
        self.assertEqual(selected,repeated)
        self.assertTrue(report['complete_image_budget'])
        self.assertEqual(report['rank_band_counts']['train'],{'top':2,'middle':1,'tail':1})
        self.assertEqual(sum(r.get('pilot',False) for r in selected),2)
        self.assertEqual(report['selection_route_counts'],{'random':1,'condition':4})
        self.assertTrue(all('selection_review' not in r for r in selected))
        self.assertTrue(all(r['actual_positive_count'] is None and r['actual_negative_count'] is None
                            for r in report['condition_sampling']))

    def test_holdout_queries_only_drive_test_and_groups_are_unique(self):
        candidates=[candidate('h','H01'),candidate('basic'),candidate('duplicate',group='basic'),
                    candidate('fill')]
        selected,report=allocate_retrieval([],[],candidates,
            [condition('OBJ01'),condition('H01',0,0,1)],budget={'train':1,'val':1,'test':1})
        self.assertTrue(report['complete_image_budget'])
        self.assertEqual(next(r['split'] for r in selected if r['image_id']=='h'),'test')
        self.assertEqual(len({r['group_id'] for r in selected}),3)
        self.assertTrue(all(r['target_condition']!='H01' for r in selected if r['split']!='test'))

    def test_exhausted_band_is_reported_without_relaxing_allocation(self):
        selected,report=allocate_retrieval([],[],[candidate('top')],[condition('OBJ01')],
                                          budget={'train':4,'val':0,'test':0})
        self.assertFalse(report['complete_image_budget'])
        self.assertEqual(report['unfilled_image_slots']['train'],3)

    def test_random_pilot_split_conflict_rejected(self):
        r={**row('same','test'),'sha256':'same'}
        with self.assertRaises(ValueError):
            allocate_retrieval([r],[{**r,'split':'train'}],[],[],
                               budget={'train':1,'val':0,'test':1})


if __name__=='__main__': unittest.main()
