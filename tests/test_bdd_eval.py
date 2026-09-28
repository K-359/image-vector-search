import unittest

from scripts.build_bdd_eval import both_facts, judge


def box(category, x1, y1, x2, y2, **attributes):
    return {'id': len(attributes) + int(x1), 'category': category, 'attributes': attributes,
            'box2d': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}}


def record(labels=(), scene='city street', weather='clear', timeofday='daytime'):
    return {'attributes': {'scene': scene, 'weather': weather, 'timeofday': timeofday}, 'labels': list(labels)}


IMAGE = {'width': 1280, 'height': 720}
CAR = {'op': 'exists', 'objects': [{'kind': 'car'}]}
LEFT_PEDESTRIAN = {'op': 'exists', 'objects': [{'kind': 'pedestrian', 'position': 'left'}]}


def label(expr, r):
    return judge(expr, both_facts(r, IMAGE))


class BddEvalTest(unittest.TestCase):
    def test_small_boxes_make_the_answer_unknown(self):
        self.assertEqual(label(CAR, record([box('car', 600, 300, 700, 360)])), 'yes')
        self.assertEqual(label(CAR, record([box('car', 600, 300, 610, 310)])), 'unknown')
        self.assertEqual(label(CAR, record()), 'no')
        no_pedestrian = {'op': 'not', 'term': {'op': 'exists', 'objects': [{'kind': 'pedestrian'}]}}
        self.assertEqual(label(no_pedestrian, record([box('person', 10, 10, 14, 18)])), 'unknown')

    def test_position_uses_box_centre_and_riders_are_not_pedestrians(self):
        self.assertEqual(label(LEFT_PEDESTRIAN, record([box('person', 100, 300, 140, 400)])), 'yes')
        self.assertEqual(label(LEFT_PEDESTRIAN, record([box('person', 1000, 300, 1040, 400)])), 'no')
        self.assertEqual(label(LEFT_PEDESTRIAN, record([box('rider', 100, 300, 140, 400)])), 'no')

    def test_exclusive_bdd_attributes_are_not_used_as_counter_evidence(self):
        tunnel = {'op': 'scene', 'name': 'tunnel'}
        self.assertEqual(label(tunnel, record(scene='tunnel')), 'yes')
        self.assertEqual(label(tunnel, record(scene='highway')), 'unknown')
        rain = {'op': 'scene', 'name': 'rain'}
        self.assertEqual(label(rain, record(weather='clear')), 'no')
        self.assertEqual(label(rain, record(weather='overcast')), 'unknown')
        self.assertEqual(label({'op': 'scene', 'name': 'snowfall'}, record(weather='snowy')), 'unknown')
        self.assertEqual(label({'op': 'scene', 'name': 'residential'}, record(scene='city street')), 'unknown')

    def test_attributes_bdd_does_not_annotate_stay_unknown(self):
        red_car = {'op': 'exists', 'objects': [{'kind': 'car', 'color': 'red'}]}
        self.assertEqual(label(red_car, record([box('car', 600, 300, 700, 360)])), 'unknown')
        self.assertEqual(label(red_car, record()), 'no')
        same_lane = {'op': 'exists', 'objects': [{'kind': 'car', 'lane': 'same'}]}
        self.assertEqual(label(same_lane, record([box('car', 600, 300, 700, 360)])), 'unknown')


if __name__ == '__main__':
    unittest.main()
