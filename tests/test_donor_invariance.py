"""
tests/test_donor_invariance.py
Unit tests for donor 3D structural invariance, unconditional Kabsch alignment, and drift gating.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from scripts.chimera_stages.stage2_clash import (
    audit_motif_cartesian_drift,
)
from scripts.chimera_stages.stage4_runners import (
    run_closed_loop_backbone_generation,
    get_last_closed_loop_history,
)


class TestDonorInvariance(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_donor_invariance_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_pdb(self, path: str, coords: list, chain: str = "A", resi_start: int = 1):
        lines = []
        for idx, (x, y, z) in enumerate(coords, start=1):
            resi = resi_start + idx - 1
            line = f"ATOM  {idx:>5}  CA  ALA {chain}{resi:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            lines.append(line)
        lines.append("TER\nEND\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def test_unconditional_kabsch_realignment_under_se3_offset(self):
        """
        Tests that an SE(3) rotated and translated candidate (< 5.0 A offset)
        is unconditionally Kabsch-aligned back to the crystallographic frame,
        resulting in ~0.0 A internal drift.
        """
        seed_pdb = os.path.join(self.test_dir, "seed.pdb")
        cand_pdb = os.path.join(self.test_dir, "candidate.pdb")

        # 4 non-collinear residues
        seed_coords = [
            [10.0, 10.0, 10.0],
            [13.8, 10.0, 10.0],
            [13.8, 13.8, 10.0],
            [10.0, 13.8, 13.8],
        ]
        self._create_sample_pdb(seed_pdb, seed_coords, chain="D", resi_start=10)

        # Apply pure rigid rotation (45 deg around Z) + 2.0 A translation
        theta = np.pi / 4.0
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,           0.0,           1.0],
        ])
        t = np.array([1.5, 2.0, -1.0])

        cand_coords = [np.dot(c, R) + t for c in seed_coords]
        self._create_sample_pdb(cand_pdb, cand_coords, chain="D", resi_start=10)

        seed_prov = {
            10: {"orig_resi": 10, "orig_chain": "D", "orig_resn": "ALA"},
            11: {"orig_resi": 11, "orig_chain": "D", "orig_resn": "ALA"},
            12: {"orig_resi": 12, "orig_chain": "D", "orig_resn": "ALA"},
            13: {"orig_resi": 13, "orig_chain": "D", "orig_resn": "ALA"},
        }

        max_drift, mean_drift, details = audit_motif_cartesian_drift(
            candidate_pdb=cand_pdb,
            reference_seed_pdb=seed_pdb,
            seed_provenance=seed_prov,
        )

        # Because geometry is 100% rigid, Kabsch realignment should report near-zero drift
        self.assertLess(max_drift, 0.05, f"Expected drift < 0.05 A under SE(3) rotation, got {max_drift}")
        self.assertLess(mean_drift, 0.05)

    def test_internal_deformation_rejection(self):
        """
        Tests that genuine non-rigid internal deformation (> 0.80 A)
        is strictly detected and rejected.
        """
        seed_pdb = os.path.join(self.test_dir, "seed_def.pdb")
        cand_pdb = os.path.join(self.test_dir, "cand_def.pdb")

        seed_coords = [
            [10.0, 10.0, 10.0],
            [13.8, 10.0, 10.0],
            [13.8, 13.8, 10.0],
            [10.0, 13.8, 13.8],
        ]
        self._create_sample_pdb(seed_pdb, seed_coords, chain="D", resi_start=1)

        # Deform the internal coordinates
        cand_coords = [
            [10.0, 10.0, 10.0],
            [15.5, 10.0, 10.0],  # +1.7 A internal stretch
            [13.8, 13.8, 10.0],
            [10.0, 13.8, 13.8],
        ]
        self._create_sample_pdb(cand_pdb, cand_coords, chain="D", resi_start=1)

        seed_prov = {
            1: {"orig_resi": 1, "orig_chain": "D", "orig_resn": "ALA"},
            2: {"orig_resi": 2, "orig_chain": "D", "orig_resn": "ALA"},
            3: {"orig_resi": 3, "orig_chain": "D", "orig_resn": "ALA"},
            4: {"orig_resi": 4, "orig_chain": "D", "orig_resn": "ALA"},
        }

        max_drift, mean_drift, _ = audit_motif_cartesian_drift(
            candidate_pdb=cand_pdb,
            reference_seed_pdb=seed_pdb,
            seed_provenance=seed_prov,
        )

        self.assertGreater(max_drift, 0.80, f"Expected max_drift > 0.80 A, got {max_drift}")

    def test_rejection_history_accumulation_in_stage4(self):
        """
        Tests that run_closed_loop_backbone_generation records full attempt
        history and rejections for the Trial Advisor.
        """
        seed_pdb = os.path.join(self.test_dir, "loop_seed.pdb")
        seed_coords = [
            [10.0, 10.0, 10.0],
            [13.8, 10.0, 10.0],
            [13.8, 13.8, 10.0],
        ]
        self._create_sample_pdb(seed_pdb, seed_coords, chain="D", resi_start=1)

        seed_prov = {
            1: {"orig_resi": 1, "orig_chain": "D", "orig_resn": "ALA"},
            2: {"orig_resi": 2, "orig_chain": "D", "orig_resn": "ALA"},
            3: {"orig_resi": 3, "orig_chain": "D", "orig_resn": "ALA"},
        }

        # Sampler that produces drifting backbones
        def drift_sampler(seed, contig, prefix, attempt):
            out_pdb = f"{prefix}_0.pdb"
            # Deform by attempt amount
            coords = [
                [10.0, 10.0, 10.0],
                [13.8 + attempt * 1.5, 10.0, 10.0],
                [13.8, 13.8, 10.0],
            ]
            self._create_sample_pdb(out_pdb, coords, chain="D", resi_start=1)
            return out_pdb

        history = []
        with self.assertRaises(RuntimeError) as cm:
            run_closed_loop_backbone_generation(
                seed_pdb=seed_pdb,
                contig_str="A1-10/0 3-3/A15-20",
                seed_provenance=seed_prov,
                target_passing_designs=1,
                max_attempts=3,
                max_drift_cutoff=0.8,
                mean_drift_cutoff=0.5,
                work_dir=os.path.join(self.test_dir, "work_rfdiff"),
                sampler_fn=drift_sampler,
                attempt_history_out=history,
            )

        self.assertEqual(len(history), 3)
        for rec in history:
            self.assertEqual(rec["status"], "REJECTED_DRIFT")
            self.assertGreater(rec["max_drift"], 0.8)

        # Check module-level query
        mod_hist = get_last_closed_loop_history()
        self.assertEqual(len(mod_hist), 3)

        # Check exception carries rejection_history
        self.assertTrue(hasattr(cm.exception, "rejection_history"))
        self.assertEqual(len(cm.exception.rejection_history), 3)


if __name__ == "__main__":
    unittest.main()
