import unittest

from qubolens.data import Dataset, make_demo
from qubolens.pipeline import optimize_dataset


class PipelineTests(unittest.TestCase):
    def test_end_to_end_result_is_exportable(self):
        rows = []
        target = []
        for index in range(72):
            first = (index % 11) / 10
            second = ((index * 7) % 13) / 12
            duplicate = first * 0.96 + second * 0.02
            noise = ((index * 17) % 19) / 18
            rows.append((first, second, duplicate, noise))
            target.append(float(first + second > 1.0))
        dataset = Dataset(
            name="Test set",
            feature_names=("first", "second", "duplicate", "noise"),
            rows=tuple(rows),
            target=tuple(target),
            target_name="label",
            task="classification",
        )
        result = optimize_dataset(dataset, k=2, quality="fast", seed=3)
        self.assertEqual(len(result["selection"]["names"]), 2)
        self.assertEqual(result["qubo"]["export"]["format"], "QUBO")
        self.assertEqual(len(result["qubo"]["matrix"]), 4)
        self.assertIn("finding", result["insight"])
        self.assertIn("score_explanation", result["insight"])
        self.assertIn("question", result["dataset"])
        self.assertEqual(result["runtime"]["dependencies"], 0)

    def test_interactive_validation_is_bounded_and_consistent(self):
        result = optimize_dataset(
            make_demo("edge-failure"),
            k=6,
            quality="fast",
            seed=42,
        )
        comparisons = (
            result["benchmark"]["qubo"],
            result["benchmark"]["greedy"],
            result["benchmark"]["all_features"],
        )
        self.assertTrue(
            all(metrics["validation_samples"] == 600 for metrics in comparisons)
        )
        self.assertTrue(all(metrics["cv_folds"] == 3 for metrics in comparisons))
        chosen_frontier = next(
            point
            for point in result["frontier"]
            if point["method"] == "QUBO" and point["k"] == 6
        )
        self.assertEqual(chosen_frontier["score"], result["benchmark"]["qubo"]["score"])


if __name__ == "__main__":
    unittest.main()
