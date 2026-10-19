import unittest
import os
import shutil
import tempfile
import numpy as np

from scripts.pse_pdb_extractor import parse_pdb_file, extract_pdb_chains
from scripts.extract_fixed_motif import parse_residue_spec
from scripts.kabsch_rmsd import kabsch_fit
from scripts.chimera_scaffold import ChimeraScaffolder


def create_sample_pdb(path: str, chains_spec: dict):
    """
    Utility to create a valid minimal PDB file for testing.
    chains_spec: {'A': [(1, 'ALA', [0.0, 0.0, 0.0]), ...], 'B': ...}
    """
    atom_idx = 1
    with open(path, "w", encoding="utf-8") as f:
        for chain, residues in chains_spec.items():
            for resi, resn, (x, y, z) in residues:
                line = (
                    f"ATOM  {atom_idx:>5}  CA  {resn:>3} {chain}{resi:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
                )
                f.write(line)
                atom_idx += 1
            f.write("TER\n")
        f.write("END\n")


class TestChimeraScaffoldingPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Build sample donor PDB (Chain A, residues 20-50)
        self.donor_pdb = os.path.join(self.test_dir, "donor.pdb")
        donor_res = [(r, "GLY", [float(r), 0.0, 0.0]) for r in range(20, 51)]
        create_sample_pdb(self.donor_pdb, {"A": donor_res})

        # Build sample vessel PDB (Chain B, residues 1-100)
        self.vessel_pdb = os.path.join(self.test_dir, "vessel.pdb")
        vessel_res = [(r, "ALA", [0.0, float(r), 0.0]) for r in range(1, 101)]
        create_sample_pdb(self.vessel_pdb, {"B": vessel_res})

        # Build sample multi-chain PDB (Chain A donor, Chain B vessel)
        self.multichain_pdb = os.path.join(self.test_dir, "complex.pdb")
        create_sample_pdb(self.multichain_pdb, {
            "A": [(r, "SER", [float(r), 1.0, 0.0]) for r in range(10, 31)],
            "B": [(r, "LEU", [0.0, float(r), 2.0]) for r in range(1, 81)]
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pdb_inspection_and_extraction(self):
        """Test multi-chain PDB inspection and chain separation."""
        info = parse_pdb_file(self.multichain_pdb)
        self.assertEqual(info["file_type"], "pdb")
        self.assertIn("A", info["objects"])
        self.assertIn("B", info["objects"])
        self.assertEqual(info["objects"]["A"]["length"], 21)
        self.assertEqual(info["objects"]["B"]["length"], 80)
        self.assertEqual(info["objects"]["A"]["range"], (10, 30))
        self.assertEqual(info["objects"]["B"]["range"], (1, 80))

        # Test extraction
        extract_dir = os.path.join(self.test_dir, "extracted")
        extracted_files = extract_pdb_chains(self.multichain_pdb, extract_dir)
        self.assertIn("A", extracted_files)
        self.assertIn("B", extracted_files)
        self.assertTrue(os.path.isfile(extracted_files["A"]))
        self.assertTrue(os.path.isfile(extracted_files["B"]))

    def test_motif_spec_parser(self):
        """Test residue interval parser across various syntax formats."""
        res1 = parse_residue_spec("30-35", default_chain="A")
        self.assertEqual(res1, [("A", 30), ("A", 31), ("A", 32), ("A", 33), ("A", 34), ("A", 35)])

        res2 = parse_residue_spec("A10-A12, B5", default_chain="X")
        self.assertEqual(res2, [("A", 10), ("A", 11), ("A", 12), ("B", 5)])

    def test_contig_generation_replace_mode(self):
        """Test contig string for Mode: Replace / Graft."""
        out_scaffold = os.path.join(self.test_dir, "out_replace")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            linker_n="8-12",
            linker_c="8-12",
            out_dir=out_scaffold,
            dry_run=True
        )
        scaffolder.prepare_input_structures()
        contig, donor_segs, vessel_segs = scaffolder.compute_contigs_and_segments()
        # Vessel is 1-100. Excising 40-60 leaves 1-39 and 61-100
        expected = "V1-39/8-12/D30-40/8-12/V61-100"
        self.assertEqual(contig, expected)
        self.assertEqual(donor_segs, [(30, 40)])
        self.assertEqual(vessel_segs, [(1, 39), (61, 100)])

    def test_contig_generation_ligate_nterm(self):
        """Test contig string for Mode: Ligate N-term."""
        out_scaffold = os.path.join(self.test_dir, "out_nterm")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="25-35",
            chimera_mode="ligate_nterm",
            linker_c="10-15",
            out_dir=out_scaffold,
            dry_run=True
        )
        scaffolder.prepare_input_structures()
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        expected = "D25-35/10-15/V1-100"
        self.assertEqual(contig, expected)

    def test_contig_generation_ligate_cterm(self):
        """Test contig string for Mode: Ligate C-term."""
        out_scaffold = os.path.join(self.test_dir, "out_cterm")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="25-35",
            chimera_mode="ligate_cterm",
            linker_n="10-15",
            out_dir=out_scaffold,
            dry_run=True
        )
        scaffolder.prepare_input_structures()
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        expected = "V1-100/10-15/D25-35"
        self.assertEqual(contig, expected)

    def test_contig_generation_insert_loop(self):
        """Test contig string for Mode: Insert Loop."""
        out_scaffold = os.path.join(self.test_dir, "out_loop")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-35",
            chimera_mode="insert_loop",
            vessel_split_after=50,
            linker_n="6-8",
            linker_c="6-8",
            out_dir=out_scaffold,
            dry_run=True
        )
        scaffolder.prepare_input_structures()
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        expected = "V1-50/6-8/D30-35/6-8/V51-100"
        self.assertEqual(contig, expected)

    def test_seed_pdb_creation_and_atom_merging(self):
        """Test creation of merged seed PDB containing donor and vessel atoms."""
        out_scaffold = os.path.join(self.test_dir, "out_seed")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            out_dir=out_scaffold,
            dry_run=True
        )
        scaffolder.prepare_input_structures()
        scaffolder.compute_contigs_and_segments()
        seed_path = scaffolder.create_merged_seed_pdb()

        self.assertTrue(os.path.isfile(seed_path))
        parsed_seed = parse_pdb_file(seed_path)
        # Should have chains V and D
        self.assertIn("V", parsed_seed["objects"])
        self.assertIn("D", parsed_seed["objects"])
        # Preserved donor residues 30-40 -> 11 residues
        self.assertEqual(parsed_seed["objects"]["D"]["length"], 11)
        # Preserved vessel residues (1-39 and 61-100) -> 39 + 40 = 79 residues
        self.assertEqual(parsed_seed["objects"]["V"]["length"], 79)

    def test_kabsch_rmsd_alignment(self):
        """Test Kabsch SVD algorithm with identical and translated coordinates."""
        # Identical sets
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.copy(P)
        R, t, rmsd = kabsch_fit(P, Q)
        self.assertAlmostEqual(rmsd, 0.0, places=5)

        # Translated and rotated set
        theta = np.pi / 4
        rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        translation = np.array([10.0, -5.0, 2.5])
        Q_transformed = np.dot(P, rot_z) + translation

        R_fit, t_fit, rmsd_fit = kabsch_fit(P, Q_transformed)
        self.assertAlmostEqual(rmsd_fit, 0.0, places=5)

    def test_full_chimera_pipeline_execution(self):
        """Test full dry_run execution producing summary, fixed positions, and metrics CSV."""
        out_run = os.path.join(self.test_dir, "out_run")
        scaffolder = ChimeraScaffolder(
            input_file=self.multichain_pdb,
            donor_id="A",
            vessel_id="B",
            motif_residues="15-25",
            chimera_mode="replace",
            vessel_replace_range="30-50",
            num_designs=2,
            out_dir=out_run,
            dry_run=True
        )
        res = scaffolder.run_pipeline()
        self.assertEqual(res["status"], "COMPLETED")
        self.assertTrue(os.path.isfile(res["metrics_summary_csv"]))
        self.assertTrue(os.path.isfile(os.path.join(out_run, "chimera_summary.json")))
        self.assertGreater(res["total_candidates"], 0)

    def test_buffered_extraction_and_writing(self):
        """Test single-pass extraction and buffered file writing."""
        from scripts.pse_pdb_extractor import write_atoms_to_pdb
        out_single_pass = os.path.join(self.test_dir, "single_pass_out")
        extracted = extract_pdb_chains(self.multichain_pdb, out_single_pass, buffer_size=65536)
        self.assertEqual(len(extracted), 2)
        for ch, p in extracted.items():
            self.assertTrue(os.path.isfile(p))

        # Test write_atoms_to_pdb with custom buffer
        test_atoms = [
            {"atom_name": "CA", "resn": "ALA", "resi": 1, "chain": "A", "x": 1.0, "y": 2.0, "z": 3.0}
        ]
        test_pdb = os.path.join(self.test_dir, "test_write.pdb")
        write_atoms_to_pdb(test_atoms, test_pdb, buffer_size=65536)
        self.assertTrue(os.path.isfile(test_pdb))

    def test_kabsch_fit_zero_copy_alignment(self):
        """Test Kabsch SVD alignment with return_aligned=True without redundant matrix allocations."""
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = P + np.array([2.0, 3.0, 4.0])
        R, t, rmsd, P_aligned, sq_dist = kabsch_fit(P, Q, return_aligned=True)
        self.assertAlmostEqual(rmsd, 0.0, places=5)
        self.assertEqual(P_aligned.shape, (4, 3))
        self.assertEqual(sq_dist.shape, (4,))
        np.testing.assert_allclose(P_aligned, Q, atol=1e-5)

    def test_mpnn_batch_size_configuration(self):
        """Test mpnn_batch_size parameter propagation and command generation."""
        out_batch = os.path.join(self.test_dir, "out_batch")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-35",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            num_designs=1,
            mpnn_batch_size=4,
            out_dir=out_batch,
            dry_run=True
        )
        res = scaffolder.run_pipeline()
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(scaffolder.mpnn_batch_size, 4)
        summary_json = os.path.join(out_batch, "chimera_summary.json")
        self.assertTrue(os.path.isfile(summary_json))

    def test_vessel_locking_and_rfdiffusion_noise_scale(self):
        """
        Verify:
        1. When lock_vessel=True, vessel framework residues are locked in fixed_positions_dict.jsonl.
        2. When lock_vessel=False, only donor motif residues are locked.
        3. denoiser.noise_scale_ca=0.0 and noise_scale_frame=0.0 are strictly present in rfdiffusion_command.
        """
        import json
        out_locked = os.path.join(self.test_dir, "out_locked")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-35",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            lock_vessel=True,
            num_designs=1,
            out_dir=out_locked,
            dry_run=True
        )
        res = scaffolder.run_pipeline()

        # Check noise scales
        cmd = res["rfdiffusion_command"]
        self.assertIn("denoiser.noise_scale_ca=0.0", cmd)
        self.assertIn("denoiser.noise_scale_frame=0.0", cmd)

        # Check fixed_positions_dict.jsonl
        jsonl_files = [
            os.path.join(out_locked, "2_mpnn", f)
            for f in os.listdir(os.path.join(out_locked, "2_mpnn"))
            if f.endswith("_fixed_positions.jsonl")
        ]
        self.assertTrue(len(jsonl_files) > 0)
        with open(jsonl_files[0], "r", encoding="utf-8") as f:
            fixed_data = json.loads(f.readline())

        # When vessel is locked (1-39 and 61-100 = 79 residues) + motif (30-35 = 6 residues)
        # Total locked positions must exceed motif-only residues
        chain_fixed = list(fixed_data.values())[0]["A"]
        self.assertGreater(len(chain_fixed), 6)

        # Now test with lock_vessel=False
        out_unlocked = os.path.join(self.test_dir, "out_unlocked")
        scaffolder_unlocked = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-35",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            lock_vessel=False,
            num_designs=1,
            out_dir=out_unlocked,
            dry_run=True
        )
        res_unlocked = scaffolder_unlocked.run_pipeline()
        self.assertEqual(res_unlocked["status"], "COMPLETED")
        jsonl_files_unlocked = [
            os.path.join(out_unlocked, "2_mpnn", f)
            for f in os.listdir(os.path.join(out_unlocked, "2_mpnn"))
            if f.endswith("_fixed_positions.jsonl")
        ]
        with open(jsonl_files_unlocked[0], "r", encoding="utf-8") as f:
            fixed_data_unlocked = json.loads(f.readline())
        chain_fixed_unlocked = list(fixed_data_unlocked.values())[0]["A"]
        # Only 6 motif residues locked
        self.assertEqual(len(chain_fixed_unlocked), 6)

    def test_methodology_report_generation(self):
        """Test automated methodology report generation in output directory."""
        out_rep = os.path.join(self.test_dir, "out_rep")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-35",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            out_dir=out_rep,
            dry_run=True
        )
        scaffolder.run_pipeline()

        rep1 = os.path.join(out_rep, "run_methodology.txt")
        rep2 = os.path.join(out_rep, "methodology.txt")

        self.assertTrue(os.path.isfile(rep1), "run_methodology.txt not found")
        self.assertTrue(os.path.isfile(rep2), "methodology.txt not found")

        with open(rep1, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("AUTOSCAFFOLD: COMPUTATIONAL METHODOLOGY REPORT", content)
        self.assertIn("Chimera Scaffolding", content)
        self.assertIn("EXECUTION ENVIRONMENT & HARDWARE ARCHITECTURE", content)
        self.assertIn("EXACT REPRODUCIBILITY CLI COMMAND", content)
        self.assertIn("denoiser.noise_scale_ca=0.0", content)

    def test_methodology_reporter_caching_and_buffering(self):
        """Test hardware caching, timeout bounds, and buffered report generation."""
        import scripts.methodology_reporter as mr

        # Test initial hardware info
        info1 = mr.get_system_hardware_info()
        self.assertIsInstance(info1, dict)
        self.assertIn("os", info1)
        self.assertIn("cpu", info1)
        self.assertIsNotNone(mr._SYSTEM_HARDWARE_CACHE)

        # Mutate cache to ensure get_system_hardware_info returns cached copy instantly
        mr._SYSTEM_HARDWARE_CACHE["cpu"] = "Cached_Test_CPU_Architecture"
        info2 = mr.get_system_hardware_info()
        self.assertEqual(info2["cpu"], "Cached_Test_CPU_Architecture")

        # Test buffered generation & symlink/copy
        out_buf = os.path.join(self.test_dir, "out_buffered_report")
        mr.generate_methodology_report(
            run_name="Cache_Buffer_Test",
            pipeline_mode="Chimera Scaffolding",
            output_dir=out_buf
        )
        f1 = os.path.join(out_buf, "run_methodology.txt")
        f2 = os.path.join(out_buf, "methodology.txt")
        self.assertTrue(os.path.isfile(f1))
        self.assertTrue(os.path.isfile(f2) or os.path.islink(f2))
        with open(f2, "r", encoding="utf-8") as f:
            self.assertIn("Cached_Test_CPU_Architecture", f.read())

        # Reset cache
        mr._SYSTEM_HARDWARE_CACHE = None


if __name__ == "__main__":
    unittest.main()
