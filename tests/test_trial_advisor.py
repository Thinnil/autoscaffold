"""
tests/test_trial_advisor.py
Unit tests for TrialAdvisor biophysical reasoning, slack ratios, and recommendations.
"""

import os
import shutil
import tempfile
import unittest

from scripts.trial_advisor import (
    TrialAdvisor,
    parse_linker_range,
    compute_slack_ratio,
    classify_slack_ratio,
    generate_trial_recommendations,
)


class TestTrialAdvisor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_trial_advisor_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parse_linker_range(self):
        self.assertEqual(parse_linker_range("6-10"), (6, 10, 8.0))
        self.assertEqual(parse_linker_range("12"), (12, 12, 12.0))
        self.assertEqual(parse_linker_range("10-6"), (6, 10, 8.0))
        self.assertEqual(parse_linker_range(None), (6, 10, 8.0))

    def test_slack_ratio_calculation_and_classification(self):
        # 8 aa linker, gap = 10 A -> reach = 30.4 A, sigma = 3.04
        sigma_slack = compute_slack_ratio(8, 10.0)
        self.assertAlmostEqual(sigma_slack, 3.04, places=2)
        self.assertIn("MODERATE_SLACK", classify_slack_ratio(sigma_slack))

        # 8 aa linker, gap = 40 A -> reach = 30.4 A, sigma = 0.76
        sigma_strain = compute_slack_ratio(8, 40.0)
        self.assertAlmostEqual(sigma_strain, 0.76, places=2)
        self.assertIn("CRITICAL_STRAIN", classify_slack_ratio(sigma_strain))

    def test_recommendation_on_linker_strain(self):
        ctx = {
            "linker_n": "4-6",
            "linker_c": "6-10",
            "chimera_mode": "replace",
            "out_dir": self.test_dir,
            "junction_measurements": {
                "n_terminal_distance": 28.0,  # 6 aa max reach is 22.8 A -> shortfall!
                "c_terminal_distance": 15.0,  # 10 aa reach is 38.0 A -> feasible
            },
        }
        advisor = TrialAdvisor(ctx, status="SUCCESS")
        res = advisor.analyze()
        recs = res["recommendations"]

        rec_ids = [r["id"] for r in recs]
        self.assertIn("REC-01-LINKER-N-EXTEND", rec_ids)
        self.assertNotIn("REC-02-LINKER-C-EXTEND", rec_ids)

        cmd = res["suggested_cli_command"]
        self.assertIn("--linker_n", cmd)

    def test_recommendation_on_excessive_slack_and_clashes(self):
        ctx = {
            "linker_n": "15-20",
            "linker_c": "6-10",
            "chimera_mode": "replace",
            "out_dir": self.test_dir,
            "junction_measurements": {
                "n_terminal_distance": 12.0,  # 20 aa reach is 76 A -> sigma = 6.33 (excessive)
                "c_terminal_distance": 15.0,
            },
        }
        summary = {"steric_clashes_detected": 12}
        advisor = TrialAdvisor(ctx, summary_data=summary, status="SUCCESS")
        res = advisor.analyze()
        rec_ids = [r["id"] for r in res["recommendations"]]
        self.assertIn("REC-03-LINKER-N-SHORTEN", rec_ids)

    def test_recommendation_on_drift_rejections(self):
        ctx = {
            "linker_n": "6-10",
            "linker_c": "6-10",
            "chimera_mode": "replace",
            "out_dir": self.test_dir,
            "trial_history": [
                {"attempt": 1, "max_drift": 1.2, "status": "REJECTED_DRIFT"},
                {"attempt": 2, "max_drift": 1.4, "status": "REJECTED_DRIFT"},
                {"attempt": 3, "max_drift": 0.9, "status": "REJECTED_DRIFT"},
            ],
            "junction_measurements": {
                "n_terminal_distance": 14.0,
                "c_terminal_distance": 14.0,
            },
        }
        advisor = TrialAdvisor(ctx, status="FAILED")
        res = advisor.analyze()
        rec_ids = [r["id"] for r in res["recommendations"]]
        self.assertIn("REC-05-SHIFT-HOST-ANCHOR", rec_ids)

    def test_recommendation_on_asymmetric_reach(self):
        ctx = {
            "linker_n": "6-10",
            "linker_c": "6-10",
            "chimera_mode": "replace",
            "out_dir": self.test_dir,
            "junction_measurements": {
                "n_terminal_distance": 8.0,
                "c_terminal_distance": 32.0,  # d_C > 2.5 * d_N and > 25 A
            },
        }
        advisor = TrialAdvisor(ctx, status="SUCCESS")
        res = advisor.analyze()
        rec_ids = [r["id"] for r in res["recommendations"]]
        self.assertIn("REC-06-GRAFT-MODE-ASYM-C", rec_ids)

    def test_artifact_generation(self):
        ctx = {
            "linker_n": "4-6",
            "linker_c": "6-10",
            "chimera_mode": "replace",
            "out_dir": self.test_dir,
            "junction_measurements": {
                "n_terminal_distance": 30.0,
                "c_terminal_distance": 15.0,
            },
        }
        # Pre-create run_methodology.txt
        meth_file = os.path.join(self.test_dir, "run_methodology.txt")
        with open(meth_file, "w", encoding="utf-8") as f:
            f.write("Initial Methodology Content\n")

        advice = generate_trial_recommendations(ctx, output_dir=self.test_dir, status="SUCCESS")
        self.assertEqual(advice["status"], "SUCCESS")
        md_file = os.path.join(self.test_dir, "next_trials_recommendations.md")

        self.assertTrue(os.path.isfile(md_file))
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("AutoScaffold Trial Advisor", content)
        self.assertIn("Biophysical Performance", content)
        self.assertIn("Ready-to-Run Next Trial Command", content)

        # Check methodology file appended
        with open(meth_file, "r", encoding="utf-8") as f:
            meth_content = f.read()
        self.assertIn("AUTOSCAFFOLD TRIAL ADVISOR: POST-RUN RECOMMENDATIONS", meth_content)


if __name__ == "__main__":
    unittest.main()
