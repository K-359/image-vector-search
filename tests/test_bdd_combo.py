import random
import unittest

from scripts.build_combo_conditions import draw, expression, sentence, size


class ComboConditionTest(unittest.TestCase):
    def test_size_counts_scene_terms_kinds_and_positions(self):
        self.assertEqual(size(['night', 'urban'], [{'kind': 'bus'}]), 3)
        self.assertEqual(size([], [{'kind': 'bus', 'position': 'left'}, {'kind': 'pedestrian', 'position': 'right'}]), 4)

    def test_sentence_and_expression(self):
        objects = [{'kind': 'pedestrian', 'position': 'left'}]
        self.assertEqual(sentence(['night', 'urban'], objects), '夜・市街地。画面左に歩行者がいる')
        self.assertEqual(expression(['night'], objects),
                         {'op': 'all', 'terms': [{'op': 'scene', 'name': 'night'}, {'op': 'exists', 'objects': objects}]})
        self.assertEqual(expression(['night'], []), {'op': 'scene', 'name': 'night'})

    def test_draw_uses_only_true_atoms_and_positions_two_objects_together(self):
        scenes = {'night', 'clear', 'urban', 'crosswalk', 'red_signal'}
        objects = {'bus': ['left'], 'pedestrian': ['center', 'right'], 'bicycle': []}
        rng = random.Random(0)
        for _ in range(300):
            drawn = draw(rng, scenes, objects, rng.choice([3, 4]))
            if drawn is None: continue
            picked, chosen = drawn
            self.assertIn(size(picked, chosen), (3, 4))
            self.assertTrue(set(picked) <= scenes)
            for o in chosen:
                self.assertIn(o['kind'], objects)
                if 'position' in o: self.assertIn(o['position'], objects[o['kind']])
            if len(chosen) == 2: self.assertEqual(len({'position' in o for o in chosen}), 1)


if __name__ == '__main__':
    unittest.main()
