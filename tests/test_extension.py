import unittest

from scripts.annotate_extension_facts import merge_facts
from scripts.condition_data import evaluate, load_extension_conditions
from scripts.generate_condition_queries import canonical_parse, canonical_target
from scripts.measure_catalog_coverage import gaps_of


def car(id, color='unknown', bbox=(100, 300, 300, 500), kind='car'):
    return {'id': id, 'kind': kind, 'bbox': list(bbox), 'color': color}


def facts(objects, complete=True):
    kinds = {'car', 'bus', 'truck', 'van', 'pedestrian', 'bicycle', 'motorcycle', 'train', 'animal', 'emergency_vehicle'}
    present = {o['kind'] for o in objects}
    return {'scene': {'night': 'yes'}, 'objects': [{**o, 'emergency_vehicle': 'no'} for o in objects],
            'inventory': {k: {'presence': 'yes' if k in present else 'no', 'complete': complete} for k in kinds}}


PARKED_CAR = {'op': 'exists', 'objects': [{'kind': 'car', 'parked': 'yes'}]}
YELLOW_CAR = {'op': 'exists', 'objects': [{'kind': 'car', 'color_ext': 'yellow'}]}


class ExtensionTest(unittest.TestCase):
    def test_merge_keeps_v3_colours_and_drops_small_boxes(self):
        merged = merge_facts(facts([car('o1', 'white'), car('o2'), car('o3', bbox=(0, 0, 10, 10))]), {
            'o1': {'parked': 'yes', 'color': 'yellow'}, 'o2': {'parked': 'no', 'color': 'yellow'},
            'o3': {'parked': 'yes', 'color': 'green'}})
        by_id = {o['id']: o for o in merged['objects']}
        self.assertEqual((by_id['o1']['color_ext'], by_id['o1']['parked']), ('other', 'yes'))
        self.assertEqual((by_id['o2']['color_ext'], by_id['o2']['parked']), ('yellow', 'no'))
        self.assertEqual((by_id['o3']['color_ext'], by_id['o3']['parked']), ('unknown', 'unknown'))
        self.assertEqual(merge_facts(facts([car('o1')]), None)['objects'][0]['parked'], 'unknown')

    def test_parked_and_yellow_conditions(self):
        both_moving = merge_facts(facts([car('o1', 'black'), car('o2', 'white')]),
                                  {'o1': {'parked': 'no', 'color': 'other'}, 'o2': {'parked': 'no', 'color': 'other'}})
        self.assertEqual(evaluate(PARKED_CAR, both_moving), 'no')
        self.assertEqual(evaluate(YELLOW_CAR, both_moving), 'no')
        one_parked = merge_facts(facts([car('o1'), car('o2')]),
                                 {'o1': {'parked': 'yes', 'color': 'yellow'}, 'o2': {'parked': 'unknown', 'color': 'unknown'}})
        self.assertEqual(evaluate(PARKED_CAR, one_parked), 'yes')
        self.assertEqual(evaluate(YELLOW_CAR, one_parked), 'yes')
        unsure = merge_facts(facts([car('o1')]), {'o1': {'parked': 'unknown', 'color': 'unknown'}})
        self.assertEqual(evaluate(PARKED_CAR, unsure), 'unknown')

    def test_extension_catalog_and_parser_use_the_same_keys(self):
        conditions = {c['id']: c for c in load_extension_conditions()}
        self.assertEqual(len(conditions), 25)
        parsed = {'scene': [], 'other': [], 'objects': [{'kind': 'car', 'color': 'yellow', 'position': 'none', 'lane': 'none',
                                                          'place': 'none', 'orientation': 'none', 'parked': 'yes'}]}
        self.assertEqual(canonical_parse(parsed), canonical_target(conditions['H21']['expression']))

    def test_extension_coverage_counts_mapped_words_only(self):
        parsed = {'scene': [], 'other': [], 'objects': [{'kind': 'car', 'color': 'yellow', 'parked': 'yes'}]}
        row = {'text': '路肩に停車している黄色い車', 'parsed': parsed}
        self.assertEqual(gaps_of(row, extension=True), set())
        self.assertEqual(gaps_of(row), {'停止・駐車', '一覧にない車種・色'})
        waiting = {'text': '信号待ちで停車している車', 'parsed': {'scene': [], 'other': [], 'objects': [{'kind': 'car', 'color': 'none'}]}}
        self.assertIn('停止・駐車', gaps_of(waiting, extension=True))


if __name__ == '__main__':
    unittest.main()
