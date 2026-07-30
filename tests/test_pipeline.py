import unittest

from qubolens.data import Dataset
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
        self.assertEqual(result["runtime"]["dependencies"], 0)


if __name__ == "__main__":
    unittest.main()
