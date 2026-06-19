import unittest

from scripts.search import WRONG_WAY_BICYCLE_QUERY, build_reranker_query


class BuildRerankerQueryTest(unittest.TestCase):
    def test_normalizes_short_wrong_way_bicycle_query(self):
        self.assertEqual(
            build_reranker_query("自転車が逆走している"),
            WRONG_WAY_BICYCLE_QUERY,
        )

    def test_normalizes_whitespace_variant(self):
        self.assertEqual(
            build_reranker_query(" 逆走している 自転車 "),
            WRONG_WAY_BICYCLE_QUERY,
        )

    def test_keeps_unrelated_query(self):
        query = "自転車が横から飛び出している"
        self.assertEqual(build_reranker_query(query), query)

    def test_keeps_detailed_query_to_avoid_dropping_constraints(self):
        query = "赤い服の自転車が工事中の交差点で右側から逆走しているところを探して"
        self.assertEqual(build_reranker_query(query), query)


if __name__ == "__main__":
    unittest.main()
