"""Regression tests for the Tar1 IPM matrix-coding plan."""

import unittest

import pandas as pd

from generate_embed_plan import generate_embedding_plan


class EmbeddingPlanTests(unittest.TestCase):
    def test_empty_candidate_table_has_no_plan(self):
        self.assertEqual(generate_embedding_plan(pd.DataFrame()), [])

    def test_flips_one_to_zero_when_required_by_matrix_syndrome(self):
        rows = []
        for index, (x, y, bit) in enumerate(
            ((0, 0, 1), (4, 0, 1), (0, 4, 1), (4, 4, 0))
        ):
            rows.append(
                {
                    "Frame": 1,
                    "CTU": 0,
                    "x": x,
                    "y": y,
                    "depth": 3,
                    "paritionSize": 3,
                    "coverIPM": 10 + index,
                    "coverBit": bit,
                    "D0": index + 1,
                    "D1": index + 2,
                    "IPM_0": 20 + index,
                    "IPM_1": 10 + index,
                }
            )

        plan = generate_embedding_plan(pd.DataFrame(rows), payload=1.0, embed_bits=3)

        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[2]["stegIPM"], 22)

    def test_flips_zero_to_one_when_required_by_matrix_syndrome(self):
        rows = []
        for index, (x, y) in enumerate(((0, 0), (4, 0), (0, 4), (4, 4))):
            rows.append(
                {
                    "Frame": 1,
                    "CTU": 0,
                    "x": x,
                    "y": y,
                    "depth": 3,
                    "paritionSize": 3,
                    "coverIPM": 10 + index,
                    "coverBit": 0,
                    "D0": index + 1,
                    "D1": index + 2,
                    "IPM_0": 10 + index,
                    "IPM_1": 20 + index,
                }
            )

        plan = generate_embedding_plan(pd.DataFrame(rows), payload=1.0, embed_bits=3)

        self.assertEqual(plan[2]["stegIPM"], 22)


if __name__ == "__main__":
    unittest.main()
