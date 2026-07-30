import itertools
import unittest

from qubolens.core import build_feature_qubo, solve_qubo


class QuboModelTests(unittest.TestCase):
    def setUp(self):
        self.relevance = [1.0, 0.92, 0.78, 0.2]
        self.redundancy = [
            [1.0, 0.96, 0.08, 0.02],
            [0.96, 1.0, 0.06, 0.04],
            [0.08, 0.06, 1.0, 0.15],
            [0.02, 0.04, 0.15, 1.0],
        ]
        self.model = build_feature_qubo(
            self.relevance, self.redundancy, k=2, redundancy_weight=0.9
        )

    def test_expanded_qubo_matches_direct_objective(self):
        for bits in itertools.product((0, 1), repeat=4):
            self.assertAlmostEqual(
                self.model.energy(bits),
                self.model.direct_objective(bits),
                places=9,
            )

    def test_solver_is_seeded_and_respects_cardinality(self):
        first = solve_qubo(self.model, reads=12, sweeps=100, seed=12)
        second = solve_qubo(self.model, reads=12, sweeps=100, seed=12)
        self.assertEqual(first.bits, second.bits)
        self.assertEqual(sum(first.bits), 2)
        self.assertAlmostEqual(first.energy, second.energy)

    def test_solver_avoids_redundant_high_relevance_pair(self):
        result = solve_qubo(self.model, reads=18, sweeps=130, seed=7)
        chosen = {index for index, bit in enumerate(result.bits) if bit}
        self.assertEqual(chosen, {0, 2})

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            build_feature_qubo(self.relevance, self.redundancy, k=0)


if __name__ == "__main__":
    unittest.main()
