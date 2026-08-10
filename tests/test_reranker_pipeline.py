import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_reranker_dataset import (
    assign_splits,
    caption_coverage_for_cards,
    group_id_for,
    load_manifest_ids,
)
from scripts.evaluate_reranker import cluster_of
from scripts.reranker_common import PairRecord, resolve_adapter_path, validate_pair_captions
from scripts.train_reranker_qlora import (
    create_run_directory,
    metric_improved,
    promote_best_checkpoint,
    training_schedule,
)


def make_pair(*, query_id: str = "q1", source_image_id: str = "bdd100k:image-001", caption=None):
    return PairRecord.from_dict(
        {
            "pair_id": f"pair-{query_id}",
            "query_id": query_id,
            "query_text": "交差点を走る車",
            "image_id": "bdd100k:candidate-001",
            "image_path": "images_100k/candidate-001.jpg",
            "label": 1,
            "negative_type": "positive",
            "split": "test",
            "caption": caption,
            "source_image_id": source_image_id,
        }
    )


class SourceImageGroupingTest(unittest.TestCase):
    def test_keeps_full_hyphenated_bdd100k_id(self):
        image_id = "bdd100k:0000f77c-6257be58"
        self.assertEqual(group_id_for(image_id), image_id)

    def test_split_assigns_every_complete_image_id(self):
        cards = [
            {"image_id": "bdd100k:0000f77c-6257be58"},
            {"image_id": "bdd100k:0000f77c-cb820c98"},
            {"image_id": "bdd100k:another-video"},
        ]
        assignments = assign_splits(cards, (0.34, 0.33, 0.33), seed=42)
        self.assertEqual(set(assignments), {card["image_id"] for card in cards})
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})

    def test_bootstrap_clusters_queries_only_by_full_source_image(self):
        first = make_pair(query_id="q1", source_image_id="bdd100k:0000f77c-6257be58")
        second = make_pair(query_id="q2", source_image_id="bdd100k:0000f77c-cb820c98")
        self.assertNotEqual(cluster_of(first), cluster_of(second))
        self.assertEqual(cluster_of(first), "bdd100k:0000f77c-6257be58")


class ManifestValidationTest(unittest.TestCase):
    def test_missing_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"
            with self.assertRaisesRegex(RuntimeError, "マニフェストが見つかりません"):
                load_manifest_ids(path, ignore_manifest=False)

    def test_empty_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "マニフェストが空"):
                load_manifest_ids(path, ignore_manifest=False)

    def test_ignore_manifest_is_explicit_escape_hatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"
            self.assertIsNone(load_manifest_ids(path, ignore_manifest=True))

    def test_valid_manifest_returns_image_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                json.dumps({"image_id": "bdd100k:image-001"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_manifest_ids(path, ignore_manifest=False),
                {"bdd100k:image-001"},
            )


class CaptionCoverageTest(unittest.TestCase):
    def setUp(self):
        self.cards = [
            {"image_id": "bdd100k:image-001"},
            {"image_id": "bdd100k:image-002"},
        ]

    def test_external_caption_requires_full_image_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "coverage は 1/2"):
            caption_coverage_for_cards(
                self.cards,
                {"bdd100k:image-001": "道路"},
                path=Path("captions.jsonl"),
                allow_partial=False,
            )

    def test_external_caption_partial_coverage_requires_explicit_flag(self):
        coverage = caption_coverage_for_cards(
            self.cards,
            {"bdd100k:image-001": "道路"},
            path=Path("captions.jsonl"),
            allow_partial=True,
        )
        self.assertEqual(coverage["covered_images"], 1)
        self.assertTrue(coverage["partial_allowed"])

    def test_train_and_eval_pair_validation_requires_full_coverage(self):
        pairs = [make_pair(query_id="q1", caption="道路"), make_pair(query_id="q2")]
        with self.assertRaisesRegex(RuntimeError, "coverage は 1/2"):
            validate_pair_captions(pairs, context="test split")
        self.assertEqual(validate_pair_captions(pairs, allow_partial=True), (1, 2))


class CheckpointManagementTest(unittest.TestCase):
    def test_new_run_does_not_delete_legacy_best(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            legacy = output_dir / "adapter_config.json"
            legacy.write_text("{}", encoding="utf-8")
            run_dir = create_run_directory(output_dir)
            self.assertTrue(legacy.is_file())
            self.assertTrue(run_dir.is_dir())

    def test_promotes_run_best_without_removing_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            legacy = output_dir / "adapter_config.json"
            legacy.write_text("{}", encoding="utf-8")
            run_best = output_dir / "runs" / "run-001" / "best"
            run_best.mkdir(parents=True)
            (run_best / "adapter_config.json").write_text("{}", encoding="utf-8")
            (run_best / "adapter_model.safetensors").write_bytes(b"weights")

            promoted = promote_best_checkpoint(run_best, output_dir)

            self.assertTrue(promoted.is_symlink())
            self.assertEqual(resolve_adapter_path(output_dir), run_best.resolve())
            self.assertTrue(legacy.is_file())

    def test_promoting_new_run_atomically_repoints_best(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            first = output_dir / "runs" / "run-001" / "best"
            second = output_dir / "runs" / "run-002" / "best"
            for checkpoint in (first, second):
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

            promote_best_checkpoint(first, output_dir)
            promote_best_checkpoint(second, output_dir)

            self.assertEqual(resolve_adapter_path(output_dir), second.resolve())
            self.assertTrue((first / "adapter_config.json").is_file())

    def test_final_metric_can_update_best(self):
        self.assertTrue(metric_improved({"ndcg@5": 0.61}, 0.60))
        self.assertFalse(metric_improved({"ndcg@5": 0.60}, 0.60))


class TrainingScheduleTest(unittest.TestCase):
    def test_optimizer_steps_include_each_epoch_remainder(self):
        micro_steps, optimizer_steps = training_schedule(
            num_pairs=10,
            batch_size=3,
            epochs=2,
            gradient_accumulation_steps=3,
        )
        self.assertEqual(micro_steps, [4, 4])
        self.assertEqual(optimizer_steps, 4)

    def test_fractional_epoch_remainder_is_not_dropped(self):
        micro_steps, optimizer_steps = training_schedule(
            num_pairs=10,
            batch_size=3,
            epochs=1.5,
            gradient_accumulation_steps=3,
        )
        self.assertEqual(micro_steps, [4, 2])
        self.assertEqual(optimizer_steps, 3)


if __name__ == "__main__":
    unittest.main()
