import unittest

from qubolens.data import load_csv_dataset, make_demo


class DatasetTests(unittest.TestCase):
    def test_demo_is_deterministic(self):
        first = make_demo("edge-failure")
        second = make_demo("edge-failure")
        self.assertEqual(first.rows[:3], second.rows[:3])
        self.assertEqual(first.target[:20], second.target[:20])
        self.assertEqual(first.n_features, 18)

    def test_csv_loader_encodes_categories_and_imputes(self):
        lines = ["age,region,signal,outcome"]
        for index in range(36):
            age = "" if index == 3 else str(20 + index)
            region = ("north", "south", "west")[index % 3]
            signal = str(index % 7)
            outcome = "yes" if index % 2 else "no"
            lines.append(f"{age},{region},{signal},{outcome}")
        dataset = load_csv_dataset(
            "\n".join(lines),
            target_name="outcome",
            task="classification",
        )
        self.assertEqual(dataset.n_samples, 36)
        self.assertEqual(dataset.n_features, 3)
        self.assertEqual(set(dataset.target), {0.0, 1.0})
        self.assertTrue(any("Ordinal-encoded region" in note for note in dataset.notes))

    def test_csv_requires_enough_rows(self):
        with self.assertRaisesRegex(ValueError, "at least 30"):
            load_csv_dataset(
                "x,z,y\n1,2,0\n2,4,1\n",
                target_name="y",
                task="classification",
            )


if __name__ == "__main__":
    unittest.main()
