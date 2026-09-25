import unittest

from scripts.audit_search_coverage import classify, requirements


class CoverageEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.rules = {r['id']: r for r in requirements()}

    def test_missing_object_is_unconfirmed_not_negative(self):
        self.assertEqual(classify(self.rules['S06'], {'participants': []}), 'unconfirmed')

    def test_attributes_cannot_transfer_between_objects(self):
        card = {'participants': [{'type': 'car', 'color': 'white'}, {'type': 'bus', 'color': 'red'}]}
        self.assertNotEqual(classify(self.rules['S08'], card), 'supported_candidate')
        self.assertNotEqual(classify(self.rules['S26'], card), 'supported_candidate')

    def test_one_object_cannot_satisfy_two_required_objects(self):
        rule = {'mode': 'and', 'scene': {}, 'people': [{'type': 'car'}, {'type': 'car'}]}
        self.assertNotEqual(classify(rule, {'participants': [{'type': 'car'}]}), 'supported_candidate')

    def test_right_does_not_require_depth_weather_or_motion(self):
        for position in ['foreground_right', 'middle_right', 'background_right']:
            self.assertEqual(classify(self.rules['S10'], {'participants': [{'type': 'bicycle', 'position': position}]}), 'supported_candidate')

    def test_or_needs_only_one_alternative(self):
        self.assertEqual(classify(self.rules['X04'], {'participants': [{'type': 'truck'}]}), 'supported_candidate')

    def test_absence_and_count_and_actions_are_not_supported(self):
        for rid in ['X02', 'X03', 'D01', 'D06']:
            self.assertEqual(classify(self.rules[rid], {'participants': []}), 'unsupported')


if __name__ == '__main__':
    unittest.main()
