from __future__ import annotations

import unittest

from fg_pipeline.eval.reference_tables import PAPER_TABLES, paper_base_value, paper_reference_value


class ReferenceTablesTests(unittest.TestCase):
    def test_required_mitigation_rows_exist(self) -> None:
        rows = PAPER_TABLES["mitigation"]["rows"]
        self.assertIn("LLaVA-1.5", rows)
        self.assertIn("HSA-DPO w/ LLaVA-1.5", rows)
        self.assertAlmostEqual(rows["LLaVA-1.5"]["chairs"], 46.3)
        self.assertAlmostEqual(rows["HSA-DPO w/ LLaVA-1.5"]["pope_adv_f1"], 84.9)

    def test_reference_lookup_for_hss_and_pope(self) -> None:
        row_name, value = paper_reference_value("hss", "avg_hss")
        self.assertEqual(row_name, "LLaVA w/ HSA-DPO")
        self.assertAlmostEqual(value, 0.602)

        row_name, value = paper_base_value("pope_adv", "f1")
        self.assertEqual(row_name, "LLaVA-1.5")
        self.assertAlmostEqual(value, 84.5)


if __name__ == "__main__":
    unittest.main()
