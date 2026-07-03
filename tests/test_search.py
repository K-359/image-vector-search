import unittest
from unittest.mock import patch

from scripts.search import (
    WRONG_WAY_BICYCLE_QUERY,
    build_reranker_query,
    resolve_caption_index_embedding_device,
)


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


class ResolveCaptionIndexEmbeddingDeviceTest(unittest.TestCase):
    def test_keeps_explicit_device(self):
        self.assertEqual(
            resolve_caption_index_embedding_device(
                caption_index_embedding_device="cpu",
                embedding_device="cuda:1",
                reranker_device="cuda:0",
            ),
            "cpu",
        )

    def test_auto_uses_available_reranker_device(self):
        with patch("scripts.search.accelerator_device_is_available", return_value=True):
            self.assertEqual(
                resolve_caption_index_embedding_device(
                    caption_index_embedding_device="auto",
                    embedding_device="cpu",
                    reranker_device="cuda:0",
                ),
                "cuda:0",
            )

    def test_auto_falls_back_to_query_embedding_device(self):
        with patch("scripts.search.accelerator_device_is_available", return_value=False):
            self.assertEqual(
                resolve_caption_index_embedding_device(
                    caption_index_embedding_device="auto",
                    embedding_device="cpu",
                    reranker_device="cuda:0",
                ),
                "cpu",
            )


if __name__ == "__main__":
    unittest.main()
