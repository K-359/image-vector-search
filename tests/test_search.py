import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.search import (
    SearchResult,
    WRONG_WAY_BICYCLE_QUERY,
    build_reranker_candidate_records,
    build_reranker_query,
    resolve_index_image_paths,
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


class ResolveIndexImagePathsTest(unittest.TestCase):
    def test_keeps_stored_paths_without_override(self):
        paths = ["images/a.jpg", "images/b.jpg"]
        self.assertIs(resolve_index_image_paths(paths, None), paths)

    def test_remaps_old_paths_by_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            for name in ("a.jpg", "b.jpg"):
                (image_dir / name).touch()
            self.assertEqual(
                resolve_index_image_paths(["old/a.jpg", "old/b.jpg"], image_dir),
                [str((image_dir / "a.jpg").resolve()), str((image_dir / "b.jpg").resolve())],
            )

    def test_rejects_missing_remapped_image(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "1 件の画像が見つかりません"):
                resolve_index_image_paths(["old/missing.jpg"], Path(directory))


class BuildRerankerCandidateRecordsTest(unittest.TestCase):
    def test_records_initial_and_reranked_ranks(self):
        results = [
            SearchResult(0.9, 20, retrieval_score=0.2, reranker_score=0.9),
            SearchResult(0.8, 10, retrieval_score=0.4, reranker_score=0.8),
            SearchResult(0.7, 30, retrieval_score=0.1, reranker_score=0.7),
        ]
        records = build_reranker_candidate_records(
            results,
            [f"images/{index}.jpg" for index in range(31)],
            initial_ranks={10: 1, 20: 2},
            candidate_count=2,
        )
        self.assertEqual(
            [(record["image_id"], record["retrieval_rank"], record["reranker_rank"]) for record in records],
            [(20, 2, 1), (10, 1, 2)],
        )

if __name__ == "__main__":
    unittest.main()
