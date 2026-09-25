import copy
import unittest
from scripts.condition_data import (SCENES, INVENTORY_KINDS, load_conditions, evaluate, validate_facts, contains_heldout, position, quality_flags)
from scripts.build_condition_dataset import stratified_hits


def entity(id='o1',kind='car',**kwargs):
    return {'id':id,'kind':kind,'bbox':[10,20,110,300],'color':'unknown','orientation':'unknown','lane':'unknown','sidewalk':'unknown','on_crosswalk':'unknown','roadway':'unknown','emergency_vehicle':'no','evidence':'視認',**kwargs}


def facts(objects=(),complete=()):
    result={'scene':{s:'unknown' for s in SCENES},'scene_evidence':'判別不能','objects':list(objects),'inventory':{k:{'presence':'unknown','complete':False} for k in INVENTORY_KINDS},'visibility_notes':'全画面確認'}
    for k in INVENTORY_KINDS:
        known=[o for o in objects if o['kind']==k or (k=='emergency_vehicle' and o['emergency_vehicle']=='yes')]
        result['inventory'][k]={'presence':'yes' if known else 'no' if k in complete else 'unknown','complete':k in complete}
    return result


class ConditionEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.conditions={c['id']:c for c in load_conditions()}
    def score(self,id,f): return evaluate(self.conditions[id]['expression'],f)

    def test_unrecorded_is_unknown_but_confirmed_absence_is_negative(self):
        self.assertEqual(self.score('OBJ05',facts()),'unknown')
        self.assertEqual(self.score('OBJ05',facts(complete=['pedestrian'])),'no')
        self.assertEqual(self.score('X03',facts()),'unknown')
        self.assertEqual(self.score('X03',facts(complete=['pedestrian'])),'yes')

    def test_attributes_must_belong_to_same_object(self):
        f=facts([entity(color='white'),entity('o2','bus',color='red')],['car','bus'])
        self.assertEqual(self.score('COL03',f),'no')
        self.assertEqual(self.score('CB01',f),'no')
        self.assertEqual(self.score('CB02',f),'yes')

    def test_incomplete_enumeration_cannot_prove_attribute_absence(self):
        f=facts([entity(color='white')])
        self.assertEqual(self.score('COL03',f),'unknown')
        f['objects'][0]['color']='red'
        self.assertEqual(self.score('COL03',f),'yes')

    def test_unknown_type_can_be_an_extra_car(self):
        f=facts([entity(kind='unknown')],['car'])
        self.assertEqual(self.score('OBJ01',f),'unknown')

    def test_distinct_objects_cannot_share_one_witness(self):
        expr={'op':'exists','objects':[{'kind':'car'},{'kind':'car'}]}
        self.assertEqual(evaluate(expr,facts([entity()],['car'])),'no')
        self.assertEqual(evaluate(expr,facts([entity()])),'unknown')
        self.assertEqual(evaluate(expr,facts([entity(),entity('o2')],['car'])),'yes')

    def test_or_and_negation_respect_unknown(self):
        f=facts([entity(kind='bus')])
        self.assertEqual(self.score('X04',f),'yes')
        self.assertEqual(self.score('X04',facts(complete=['bus'])),'unknown')
        self.assertEqual(self.score('X04',facts(complete=['bus','truck'])),'no')
        f['scene']['night']='no'
        self.assertEqual(self.score('CE04',f),'no')

    def test_count_requires_complete_inventory_except_proven_excess(self):
        cars=[entity(),entity('o2')]
        self.assertEqual(self.score('X02',facts(cars)),'unknown')
        self.assertEqual(self.score('X02',facts(cars,['car'])),'yes')
        self.assertEqual(self.score('X02',facts(cars+[entity('o3')])),'no')

    def test_positions_use_boxes_not_lanes_or_other_objects(self):
        self.assertEqual(position(entity(bbox=[700,0,900,500])),'right')
        self.assertEqual(position(entity(bbox=[300,0,366,500])),'unknown')
        f=facts([entity(kind='bicycle',bbox=None,lane='right_adjacent')],['bicycle'])
        self.assertEqual(self.score('POS18',f),'unknown')

    def test_relation_uses_two_objects_and_unknown_coordinates(self):
        f=facts([entity(kind='pedestrian'),entity('o2','truck',bbox=[700,0,900,500])],['pedestrian','truck'])
        self.assertEqual(self.score('X01',f),'yes')
        f['objects'][0]['bbox']=None
        self.assertEqual(self.score('X01',f),'unknown')
        self.assertEqual(self.score('X01',facts(complete=['pedestrian'])),'no')

    def test_temporal_conditions_never_get_static_positive_or_negative(self):
        for id in ['D01','D02','D03','D04','D05','D06']: self.assertEqual(self.score(id,facts()),'unknown')

    def test_schema_rejects_invalid_and_contradictory_observations(self):
        f=facts([entity()],['car']); validate_facts(f)
        for mutate in [lambda x:x['objects'].append(copy.deepcopy(x['objects'][0])),lambda x:x['objects'][0].update(bbox=[100,0,50,90]),lambda x:x['inventory']['car'].update(presence='no'),lambda x:x['scene'].update(rain='probably')]:
            altered=copy.deepcopy(f); mutate(altered)
            with self.assertRaises(ValueError): validate_facts(altered)

    def test_catalog_and_holdout_semantic_exclusion(self):
        self.assertEqual(len(self.conditions),233)
        heldout=[c for c in self.conditions.values() if c['id'].startswith('H')]
        for c in self.conditions.values():
            if c['allocation']['use']=='train_val_test':
                self.assertFalse(any(contains_heldout(c['expression'],h['expression']) for h in heldout),c['id'])
        h=self.conditions['H08']['expression']
        superset={'op':'all','terms':[h,{'op':'scene','name':'night'}]}
        self.assertTrue(contains_heldout(superset,h))
        self.assertFalse(contains_heldout(self.conditions['COL04']['expression'],h))

    def test_candidate_sampling_is_deterministic_and_rank_stratified(self):
        hits=[{'rank':i,'image_id':str(i)} for i in range(1,1001)]
        sample=stratified_hits(hits,42,'OBJ01')
        self.assertEqual(sample,stratified_hits(hits,42,'OBJ01'))
        self.assertEqual(len(sample),120)
        self.assertEqual(sum(x['rank']<=40 for x in sample),40)
        self.assertEqual(sum(41<=x['rank']<=200 for x in sample),40)
        self.assertEqual(sum(x['rank']>200 for x in sample),40)

    def test_quality_flags_quarantine_repetition_and_conflicts_without_relabeling(self):
        f=facts([entity(str(i)) for i in range(24)])
        flags=quality_flags(f)
        self.assertIn('object_limit_reached',flags)
        self.assertIn('repeated_object_evidence',flags)
        self.assertIn('duplicate_object_box',flags)
        self.assertEqual(len(f['objects']),24)
        self.assertEqual(quality_flags(facts([entity()])),[])
        f=facts(); f['scene'].update(day='yes',night='yes')
        self.assertIn('conflicting_time_of_day',quality_flags(f))
        self.assertIn('possible_rider_as_pedestrian',quality_flags(facts([entity(kind='pedestrian',evidence='自転車に乗っている人物。')])))


if __name__=='__main__': unittest.main()
