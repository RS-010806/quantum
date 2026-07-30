import unittest

from qubolens import server


class ResultCacheTests(unittest.TestCase):
    def setUp(self):
        with server._RESULT_CACHE_LOCK:
            server._RESULT_CACHE.clear()

    def test_cached_results_are_safe_copies(self):
        key = server._cache_key(
            "demo:edge-failure",
            {
                "source": "demo",
                "dataset": "edge-failure",
                "k": 6,
                "quality": "fast",
            },
        )
        original = {
            "runtime": {"total_ms": 120.0, "cache_hit": False},
            "selection": {"names": ["signal"]},
        }
        server._store_result(key, original)

        first = server._cached_result(key)
        self.assertIsNotNone(first)
        self.assertTrue(first["runtime"]["cache_hit"])
        first["selection"]["names"].append("changed")

        second = server._cached_result(key)
        self.assertEqual(second["selection"]["names"], ["signal"])


if __name__ == "__main__":
    unittest.main()
