"""Cheap dependency-chain smoke test for local machines and Codespaces."""

import unittest


class V51ImportTest(unittest.TestCase):
    def test_bundled_model_chain_imports(self) -> None:
        import v51_combined_scoreline_model as v51

        self.assertTrue(callable(v51.build_from_zip))
        self.assertTrue(callable(v51.v29.select_top_scorelines_with_tail_risk))
        self.assertTrue(callable(v51.select_coverage_outlier))


if __name__ == "__main__":
    unittest.main()
