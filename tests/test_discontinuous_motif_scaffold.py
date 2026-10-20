import unittest
import os
import sys
import shutil
import tempfile

# Add repository root to sys.path for standalone and test-runner execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.discontinuous_motif_scaffold import DiscontinuousMotifScaffolder  # noqa: E402
from tests.test_chimera_scaffold import create_sample_pdb  # noqa: E402


class TestDiscontinuousMotifScaffold(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.complex_pdb = os.path.join(self.test_dir, "complex.pdb")
        create_sample_pdb(self.complex_pdb, {
            "A": [(r, "LEU", [float(r), 0.0, 0.0]) for r in range(1, 51)],
            "B": [(r, "TYR", [0.0, float(r), 0.0]) for r in range(100, 151)]
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_motif_info(self):
        scaffolder = DiscontinuousMotifScaffolder(
            complex_pdb=self.complex_pdb,
            motif_chain="B",
            motif_residues="110-115,130-135",
            out_dir=self.test_dir,
            dry_run=True
        )
        info = scaffolder.extract_motif_info()
        self.assertEqual(info["num_residues"], 12)
        self.assertEqual(info["sequence"], "Y" * 12)

    def test_build_contigs_monomer(self):
        scaffolder = DiscontinuousMotifScaffolder(
            complex_pdb=self.complex_pdb,
            motif_chain="B",
            motif_residues="110-115,130-135",
            min_total_len=75,
            out_dir=self.test_dir,
            dry_run=True
        )
        contigs = scaffolder.build_contigs()
        self.assertIn("B110-115", contigs)
        self.assertIn("B130-135", contigs)

    def test_build_contigs_with_target(self):
        scaffolder = DiscontinuousMotifScaffolder(
            complex_pdb=self.complex_pdb,
            motif_chain="B",
            motif_residues="110-115,130-135",
            target_chain="A",
            min_total_len=75,
            out_dir=self.test_dir,
            dry_run=True
        )
        contigs = scaffolder.build_contigs()
        self.assertIn("A1-50/0", contigs)
        self.assertIn("B110-115", contigs)

    def test_run_pipeline(self):
        scaffolder = DiscontinuousMotifScaffolder(
            complex_pdb=self.complex_pdb,
            motif_chain="B",
            motif_residues="110-115",
            target_chain="A",
            out_dir=self.test_dir,
            dry_run=True
        )
        summary = scaffolder.run_pipeline()
        self.assertEqual(summary["status"], "SUCCESS")
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "discontinuous_summary.json")))


if __name__ == "__main__":
    unittest.main()
