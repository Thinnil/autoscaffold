import unittest
from unittest.mock import patch
import os
import sys
import json
import math
import shutil
import tempfile
import subprocess
import numpy as np

# Add repository root to sys.path for standalone and test-runner execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.pse_pdb_extractor import (  # noqa: E402
    parse_pdb_file,
    extract_pdb_chains,
    extract_and_merge_objects,
    extract_and_merge_donors,
    safe_pickle_loads,
    safe_join_path,
    validate_output_dir,
    validate_pse_safety,
    clear_pdb_parse_cache,
)
from scripts.extract_fixed_motif import parse_residue_spec  # noqa: E402
from scripts.kabsch_rmsd import kabsch_fit  # noqa: E402
from scripts.chimera_scaffold import (  # noqa: E402
    ChimeraScaffolder,
    get_residue_ca_coords,
    calculate_anchor_distance,
    batch_anchor_distances,
    compute_recommended_linker,
    compute_extension_minimum,
    clear_ca_coords_cache,
    clear_probe_cache,
    clear_environment_cache,
    find_environment_managers,
    get_all_environment_roots,
    _CA_COORDS_CACHE,
    _RESI_COORDS_CACHE,
    _PROBE_CACHE,
    resolve_python_interpreter,
    resolve_neural_python,
    resolve_execution_runner,
    RunnerSpec,
    build_ephemeral_runner_cmd,
    probe_python_environment,
    display_dependency_alert,
    display_cuda_warning,
)


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
        create_sample_pdb(
            self.multichain_pdb,
            {
                "A": [(r, "SER", [float(r), 1.0, 0.0]) for r in range(10, 31)],
                "B": [(r, "LEU", [0.0, float(r), 2.0]) for r in range(1, 81)],
            },
        )

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
            clamp_linkers=False,
            out_dir=out_scaffold,
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        contig, donor_segs, vessel_segs = scaffolder.compute_contigs_and_segments()
        # Vessel is 1-100. Excising 40-60 leaves 1-39 and 61-100.
        # Donor residues (30-40) overlap with vessel (1-100), offset to 1030-1040 in unified Chain A.
        expected = "A1-39/8-12/A1030-1040/8-12/A61-100"
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
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        expected = "A1025-1035/10-15/A1-100"
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
            clamp_linkers=False,
            out_dir=out_scaffold,
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        expected = "A1-100/10-15/A1025-1035"
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
            clamp_linkers=False,
            out_dir=out_scaffold,
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        expected = "A1-50/6-8/A1030-1035/6-8/A51-100"
        self.assertEqual(contig, expected)

    def test_seed_pdb_creation_and_atom_merging(self):
        """Test creation of merged seed PDB containing donor and vessel atoms in unified Chain A."""
        out_scaffold = os.path.join(self.test_dir, "out_seed")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            out_dir=out_scaffold,
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        scaffolder.compute_contigs_and_segments()
        seed_path = scaffolder.create_merged_seed_pdb()

        self.assertTrue(os.path.isfile(seed_path))
        parsed_seed = parse_pdb_file(seed_path)
        # In RFdiffusion, all segments in a single designed chain must belong to unified Chain A
        self.assertIn("A", parsed_seed["objects"])
        # Total residues in Chain A: 79 vessel residues + 11 donor residues = 90
        self.assertEqual(parsed_seed["objects"]["A"]["length"], 90)

    def test_non_overlapping_donor_and_vessel_preserves_numbers(self):
        """Test that non-overlapping donor (e.g. 1-16) and vessel (e.g. 285-427) use 0 offset in unified Chain A."""
        vessel_ahr = os.path.join(self.test_dir, "vessel_ahr.pdb")
        donor_hsp = os.path.join(self.test_dir, "donor_hsp.pdb")
        v_res = [(r, "ALA", [0.0, float(r), 0.0]) for r in range(285, 428)]
        d_res = [(r, "GLY", [0.0, float(r) + 500.0, 0.0]) for r in range(1, 17)]
        create_sample_pdb(vessel_ahr, {"A": v_res})
        create_sample_pdb(donor_hsp, {"D": d_res})

        scaffolder = ChimeraScaffolder(
            donor_pdb=donor_hsp,
            vessel_pdb=vessel_ahr,
            motif_residues="all",
            chimera_mode="ligate_cterm",
            linker_n="10",
            clamp_linkers=False,
            out_dir=os.path.join(self.test_dir, "out_non_overlap"),
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        contig, donor_segs, vessel_segs = scaffolder.compute_contigs_and_segments()
        # Zero offset because 1-16 and 285-427 do not intersect
        self.assertEqual(scaffolder.donor_seed_offset, 0)
        self.assertTrue(contig.startswith("A285-427/10/A1-16"))
        seed_path = scaffolder.create_merged_seed_pdb()
        parsed = parse_pdb_file(seed_path)
        self.assertIn("A", parsed["objects"])
        # All atoms in unified Chain A
        self.assertEqual(len(parsed["objects"]), 1)

    def test_kabsch_rmsd_alignment(self):
        """Test Kabsch SVD algorithm with identical and translated coordinates."""
        # Identical sets
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.copy(P)
        R, t, rmsd = kabsch_fit(P, Q)
        self.assertAlmostEqual(rmsd, 0.0, places=5)

        # Translated and rotated set
        theta = np.pi / 4
        rot_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
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
            dry_run=True,
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
        test_atoms = [{"atom_name": "CA", "resn": "ALA", "resi": 1, "chain": "A", "x": 1.0, "y": 2.0, "z": 3.0}]
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
            dry_run=True,
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
            dry_run=True,
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
            dry_run=True,
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
            dry_run=True,
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
            run_name="Cache_Buffer_Test", pipeline_mode="Chimera Scaffolding", output_dir=out_buf
        )
        f1 = os.path.join(out_buf, "run_methodology.txt")
        f2 = os.path.join(out_buf, "methodology.txt")
        self.assertTrue(os.path.isfile(f1))
        self.assertTrue(os.path.isfile(f2) or os.path.islink(f2))
        with open(f2, "r", encoding="utf-8") as f:
            self.assertIn("Cached_Test_CPU_Architecture", f.read())

        # Reset cache
        mr._SYSTEM_HARDWARE_CACHE = None

    def test_get_residue_ca_coords_and_geometric_center_fallback(self):
        """Test CA coordinate extraction and geometric center fallback for missing CA."""
        custom_pdb = os.path.join(self.test_dir, "fallback_test.pdb")
        with open(custom_pdb, "w", encoding="utf-8") as f:
            # Residue 1: Has standard CA atom
            f.write("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n")
            # Residue 2: Missing CA, only N and C atoms
            f.write("ATOM      2  N   GLY A   2       0.000   0.000   0.000  1.00 20.00           N\n")
            f.write("ATOM      3  C   GLY A   2       2.000   2.000   2.000  1.00 20.00           C\n")
            f.write("TER\nEND\n")

        # Test residue 1 CA
        coord1 = get_residue_ca_coords(custom_pdb, chain_id="A", resi=1)
        self.assertIsNotNone(coord1)
        np.testing.assert_allclose(coord1, [1.0, 2.0, 3.0], atol=1e-4)

        # Test residue 2 fallback to mean of N and C: [1.0, 1.0, 1.0]
        coord2 = get_residue_ca_coords(custom_pdb, chain_id="A", resi=2)
        self.assertIsNotNone(coord2)
        np.testing.assert_allclose(coord2, [1.0, 1.0, 1.0], atol=1e-4)

        # Test non-existent residue
        coord_none = get_residue_ca_coords(custom_pdb, chain_id="A", resi=99)
        self.assertIsNone(coord_none)

    def test_calculate_anchor_distance(self):
        """Test Euclidean distance calculation between 3D vectors."""
        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([3.0, 4.0, 0.0])
        dist = calculate_anchor_distance(c1, c2)
        self.assertAlmostEqual(dist, 5.0, places=4)

    def test_compute_recommended_linker(self):
        """Test recommended linker calculation and extension minimums."""
        # 8.0 A
        min_8, max_8 = compute_recommended_linker(8.0)
        self.assertEqual(min_8, 3)
        self.assertEqual(max_8, 9)

        # 15.0 A
        min_15, max_15 = compute_recommended_linker(15.0)
        self.assertEqual(min_15, 5)
        self.assertEqual(max_15, 11)

        # 40.0 A
        min_40, max_40 = compute_recommended_linker(40.0)
        self.assertEqual(min_40, 14)
        self.assertEqual(max_40, 20)

        # Extension minimum for 40 A (ceil(40 / 3.8) = 11)
        ext_40 = compute_extension_minimum(40.0)
        self.assertEqual(ext_40, 11)

        # Non-positive distance fallback
        min_0, max_0 = compute_recommended_linker(0.0)
        self.assertEqual((min_0, max_0), (3, 9))

    def test_biophysical_clamping_impossible_linkers(self):
        """
        Verify that:
        1. When user provides custom linkers with max_len < L_ext_min,
           pipeline auto-clamps to physically feasible bounds.
        2. When user specifies 'auto', pipeline populates recommended bounds.
        """
        out_clamp = os.path.join(self.test_dir, "out_clamping_test")
        # In self.donor_pdb (res 30) and self.vessel_pdb (left 39, right 61):
        # Distance at N-junction is 49.2 A (L_ext_min = 13, recommended 17-23).
        # Distance at C-junction is 73.0 A (L_ext_min = 20, recommended 25-32).
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            linker_n="4-6",
            linker_c="4-6",
            clamp_linkers=True,
            out_dir=out_clamp,
            dry_run=True,
        )
        contig, _, _ = scaffolder.compute_contigs_and_segments()
        # Should have auto-clamped both linkers
        self.assertEqual(scaffolder.linker_n, "17-23")
        self.assertEqual(scaffolder.linker_c, "25-32")
        self.assertEqual(contig, "A1-39/17-23/A1030-1040/25-32/A61-100")

        # Test with 'auto'
        scaffolder_auto = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            linker_n="auto",
            linker_c="auto",
            clamp_linkers=True,
            out_dir=os.path.join(self.test_dir, "out_auto"),
            dry_run=True,
        )
        contig_auto, _, _ = scaffolder_auto.compute_contigs_and_segments()
        self.assertEqual(scaffolder_auto.linker_n, "17-23")
        self.assertEqual(scaffolder_auto.linker_c, "25-32")
        self.assertEqual(contig_auto, "A1-39/17-23/A1030-1040/25-32/A61-100")

    def test_crystallographic_loop_gaps_and_multi_segment_vessel(self):
        """
        Verify that unmodeled crystallographic gaps in the vessel (e.g. residues 31-60 missing)
        are properly detected and segmented rather than treated as a continuous void.
        """
        vessel_gap_pdb = os.path.join(self.test_dir, "vessel_gap.pdb")
        # Vessel has residues 1-30 and 61-100 (gap of 30 missing residues)
        v_res = [(r, "ALA", [0.0, float(r), 0.0]) for r in range(1, 31)]
        v_res.extend([(r, "ALA", [0.0, float(r), 0.0]) for r in range(61, 101)])
        create_sample_pdb(vessel_gap_pdb, {"B": v_res})

        parsed = parse_pdb_file(vessel_gap_pdb)
        self.assertEqual(parsed["objects"]["B"]["intervals"], ["1-30", "61-100"])

        out_gap = os.path.join(self.test_dir, "out_gap")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=vessel_gap_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="15-20",
            linker_n="8-12",
            linker_c="8-12",
            clamp_linkers=False,
            out_dir=out_gap,
            dry_run=True,
        )
        contig, donor_segs, vessel_segs = scaffolder.compute_contigs_and_segments()

        # Left segment is 1-14.
        # Right segments are 21-30 and 61-100.
        self.assertEqual(donor_segs, [(30, 40)])
        self.assertEqual(vessel_segs, [(1, 14), (21, 30), (61, 100)])
        # Expected contig preserves the 30aa crystallographic void as 30-30:
        expected = "A1-14/8-12/A1030-1040/8-12/A21-30/30-30/A61-100"
        self.assertEqual(contig, expected)

    def test_measure_junctions_method_and_cli(self):
        """Test measure_junctions method and CLI helper mode JSON output."""
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            clamp_linkers=False,
            out_dir=os.path.join(self.test_dir, "out_meas"),
            dry_run=True,
        )
        data = scaffolder.measure_junctions()
        self.assertEqual(data["mode"], "replace")
        self.assertIn("n_term", data["junctions"])
        self.assertIn("c_term", data["junctions"])
        self.assertAlmostEqual(data["junctions"]["n_term"]["distance_angstrom"], 49.2, places=1)
        self.assertEqual(data["junctions"]["n_term"]["ext_min"], 13)
        self.assertEqual(data["junctions"]["n_term"]["recommended_range"], "17-23")

        # Test CLI execution
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "chimera_scaffold.py")
        cmd = [
            sys.executable,
            script_path,
            "measure_junctions",
            "--donor_pdb",
            self.donor_pdb,
            "--vessel_pdb",
            self.vessel_pdb,
            "--motif_residues",
            "30-40",
            "--chimera_mode",
            "replace",
            "--vessel_replace_range",
            "40-60",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        cli_data = json.loads(res.stdout)
        self.assertEqual(cli_data["mode"], "replace")
        self.assertIn("junctions", cli_data)
        self.assertEqual(cli_data["junctions"]["n_term"]["ext_min"], 13)

    def test_launcher_path_resolution(self):
        """Test canonical path resolution in root autoscaffold launcher."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        launcher = os.path.join(repo_root, "autoscaffold")
        self.assertTrue(os.path.isfile(launcher), f"Launcher script not found: {launcher}")

        # Run launcher help via bash from repository root and from an external working directory
        res = subprocess.run(["bash", launcher, "help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("AutoScaffold Master Suite CLI", res.stdout)

        # Run from an external temporary directory to verify cwd-independence
        res_external = subprocess.run(["bash", launcher, "help"], cwd=self.test_dir, capture_output=True, text=True)
        self.assertEqual(res_external.returncode, 0)
        self.assertIn("AutoScaffold Master Suite CLI", res_external.stdout)

    def test_ca_coords_session_caching_and_invalidation(self):
        """Test in-memory session caching and invalidation for PDB CA coordinates."""
        clear_ca_coords_cache()
        self.assertEqual(len(_CA_COORDS_CACHE), 0)

        # First query: should parse and populate cache
        coord_first = get_residue_ca_coords(self.donor_pdb, chain_id="A", resi=25)
        self.assertIsNotNone(coord_first)
        self.assertEqual(len(_CA_COORDS_CACHE), 1)

        # Second query: should hit cache directly
        coord_second = get_residue_ca_coords(self.donor_pdb, chain_id="A", resi=30)
        self.assertIsNotNone(coord_second)
        self.assertEqual(len(_CA_COORDS_CACHE), 1)

        # Test clear cache
        clear_ca_coords_cache()
        self.assertEqual(len(_CA_COORDS_CACHE), 0)

    def test_measure_junction_distances_alias(self):
        """Test measure_junction_distances method alias on ChimeraScaffolder."""
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            clamp_linkers=False,
            out_dir=os.path.join(self.test_dir, "out_alias"),
            dry_run=True,
        )
        data1 = scaffolder.measure_junctions()
        data2 = scaffolder.measure_junction_distances()
        self.assertEqual(data1, data2)
        self.assertEqual(data2["mode"], "replace")
        self.assertIn("n_term", data2["junctions"])

    def test_contig_plan_file_generation(self):
        """Test buffered writing and presence of contig_plan.txt in output."""
        out_plan = os.path.join(self.test_dir, "out_plan")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            out_dir=out_plan,
            dry_run=True,
        )
        res = scaffolder.run_pipeline()
        self.assertIn("contig_plan", res)
        plan_file = res["contig_plan"]
        self.assertTrue(os.path.isfile(plan_file))
        with open(plan_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Contig Formula:", content)
        self.assertIn("Chimera Mode: replace", content)

    def test_anchor_distance_fast_scalar_math(self):
        """Test scalar fast path in calculate_anchor_distance."""
        c1 = [1.0, 2.0, 3.0]
        c2 = [4.0, 6.0, 3.0]
        # (4-1)^2 + (6-2)^2 + (3-3)^2 = 9 + 16 = 25 -> 5.0
        d = calculate_anchor_distance(c1, c2)
        self.assertAlmostEqual(d, 5.0, places=5)
        # Array input
        d_arr = calculate_anchor_distance(np.array(c1), np.array(c2))
        self.assertAlmostEqual(d_arr, 5.0, places=5)

    def test_multi_object_donor_merge(self):
        """Test multi-object extraction and coordinate-preserving merge into merged_donor.pdb."""
        multi_pdb = os.path.join(self.test_dir, "multi_donor_source.pdb")
        chain_a_res = [(r, "ALA", [float(r), 10.0, 20.0]) for r in range(1, 11)]
        chain_b_res = [(r, "GLY", [float(r) + 50.0, 30.0, 40.0]) for r in range(1, 11)]
        create_sample_pdb(multi_pdb, {"A": chain_a_res, "B": chain_b_res})

        merge_out = os.path.join(self.test_dir, "merged_donor_out")
        res = extract_and_merge_objects(multi_pdb, "A,B", merge_out, prefix="merged_donor")

        self.assertEqual(res["status"], "SUCCESS")
        merged_pdb = res["merged_pdb"]
        self.assertTrue(os.path.isfile(merged_pdb))
        manifest_path = res["manifest_path"]
        self.assertTrue(os.path.isfile(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["total_amino_acids"], 20)
        self.assertEqual(len(manifest["segments"]), 2)
        self.assertEqual(manifest["segments"][0]["residue_range"], [1, 10])
        self.assertEqual(manifest["segments"][0]["new_residue_range"], [1, 10])
        self.assertEqual(manifest["segments"][1]["residue_range"], [1, 10])
        self.assertEqual(manifest["segments"][1]["new_residue_range"], [11, 20])

        # Verify exact Cartesian coordinates preservation in merged PDB
        parsed_merged = parse_pdb_file(merged_pdb)
        self.assertIn("D", parsed_merged["objects"])
        self.assertEqual(parsed_merged["objects"]["D"]["length"], 20)

        # Check coordinates of first 10 residues (from chain A)
        for r in range(1, 11):
            coord = get_residue_ca_coords(merged_pdb, "D", r)
            self.assertIsNotNone(coord)
            np.testing.assert_allclose(coord, [float(r), 10.0, 20.0], atol=1e-3)

        # Check coordinates of next 10 residues (from chain B, reindexed to 11-20)
        for r in range(1, 11):
            coord = get_residue_ca_coords(merged_pdb, "D", r + 10)
            self.assertIsNotNone(coord)
            np.testing.assert_allclose(coord, [float(r) + 50.0, 30.0, 40.0], atol=1e-3)

    def test_extract_and_merge_donors_wrapper(self):
        """Test extract_and_merge_donors wrapper function with custom output path."""
        multi_pdb = os.path.join(self.test_dir, "wrapper_source.pdb")
        chain_a_res = [(r, "ALA", [float(r), 10.0, 20.0]) for r in range(1, 6)]
        create_sample_pdb(multi_pdb, {"A": chain_a_res})
        out_merged = os.path.join(self.test_dir, "wrapper_out", "custom_donor.pdb")
        res = extract_and_merge_donors(multi_pdb, "A", out_merged)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(os.path.isfile(out_merged))
        self.assertEqual(res["total_amino_acids"], 5)

    def test_discontinuous_motif_auto_segmentation(self):
        """Test auto-segmentation on discontinuous donor with motif_residues='all'."""
        disc_pdb = os.path.join(self.test_dir, "discontinuous_donor.pdb")
        seg1 = [(r, "ALA", [float(r), 0.0, 0.0]) for r in range(10, 16)]
        # Distance between residue 15 (15.0, 0, 0) and residue 80 (40.0, 0, 0) is 25.0 Å (> 4.2 Å)
        seg2 = [(r, "GLY", [40.0 + float(r - 80), 0.0, 0.0]) for r in range(80, 86)]
        create_sample_pdb(disc_pdb, {"A": seg1 + seg2})

        scaffolder = ChimeraScaffolder(
            donor_pdb=disc_pdb,
            vessel_pdb=self.vessel_pdb,
            donor_id="A",
            vessel_id="B",
            motif_residues="all",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            out_dir=os.path.join(self.test_dir, "out_disc_seg"),
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        segments = scaffolder.parse_motif_ranges()
        self.assertEqual(segments, [(10, 15), (80, 85)])

    def test_internal_linker_budgeting(self):
        """Test internal linker budgeting for a measured 3D gap of 21.0 Å."""
        l_ext = compute_extension_minimum(21.0)
        l_glob = max(3, int(math.ceil(21.0 / 3.0)))
        l_max = l_glob + 6
        self.assertEqual(l_ext, 6)
        self.assertEqual(l_glob, 7)
        self.assertEqual(f"{l_glob}-{l_max}", "7-13")

        disc_pdb = os.path.join(self.test_dir, "gap21_donor.pdb")
        seg1 = [(r, "ALA", [0.0, 0.0, 0.0]) for r in range(1, 6)]
        # Seg1 end at (0,0,0). Seg2 start at (21.0, 0, 0) -> distance = 21.0 Å
        seg2 = [(r, "GLY", [21.0 + float(r - 10), 0.0, 0.0]) for r in range(10, 16)]
        create_sample_pdb(disc_pdb, {"D": seg1 + seg2})

        scaffolder = ChimeraScaffolder(
            donor_pdb=disc_pdb,
            vessel_pdb=self.vessel_pdb,
            donor_id="D",
            vessel_id="B",
            motif_residues="all",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            linker_n="8-12",
            linker_c="8-12",
            clamp_linkers=False,
            out_dir=os.path.join(self.test_dir, "out_gap21"),
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        contig, donor_segs, _ = scaffolder.compute_contigs_and_segments()
        self.assertEqual(donor_segs, [(1, 5), (10, 15)])
        self.assertIn("/7-13/", contig)
        self.assertEqual(contig, "A1-39/8-12/A1001-1005/7-13/A1010-1015/8-12/A61-100")

    def test_fixed_positions_masking_multi_segment(self):
        """Test ProteinMPNN fixed position masking across 3 donor segments,
        2 internal linkers, and 2 external linkers."""
        donor_pdb = os.path.join(self.test_dir, "mask_donor.pdb")
        s1 = [(r, "ALA", [float(r), 0.0, 0.0]) for r in range(1, 6)]
        s2 = [(r, "ALA", [float(r) + 20.0, 0.0, 0.0]) for r in range(10, 15)]
        s3 = [(r, "ALA", [float(r) + 40.0, 0.0, 0.0]) for r in range(20, 26)]
        create_sample_pdb(donor_pdb, {"D": s1 + s2 + s3})

        vessel_pdb = os.path.join(self.test_dir, "mask_vessel.pdb")
        v_res = [(r, "VAL", [0.0, float(r), 0.0]) for r in range(1, 21)]
        create_sample_pdb(vessel_pdb, {"V": v_res})

        # Mock chimeric backbone with:
        # V_left (1-10) -> fixed
        # Linker N (11-13) -> unfixed
        # S1 (14-18) -> fixed
        # Int Linker 1 (19-21) -> unfixed
        # S2 (22-26) -> fixed
        # Int Linker 2 (27-29) -> unfixed
        # S3 (30-35) -> fixed
        # Linker C (36-38) -> unfixed
        # V_right (39-48) -> fixed
        bb_pdb = os.path.join(self.test_dir, "mock_chimeric_bb.pdb")
        bb_res = []
        cur_idx = 1
        for r in range(1, 11):
            bb_res.append((cur_idx, "VAL", [0.0, float(r), 0.0]))
            cur_idx += 1
        for _ in range(3):
            bb_res.append((cur_idx, "GLY", [999.0, 999.0, 999.0]))
            cur_idx += 1
        for r in range(1, 6):
            bb_res.append((cur_idx, "ALA", [float(r), 0.0, 0.0]))
            cur_idx += 1
        for _ in range(3):
            bb_res.append((cur_idx, "GLY", [998.0, 998.0, 998.0]))
            cur_idx += 1
        for r in range(10, 15):
            bb_res.append((cur_idx, "ALA", [float(r) + 20.0, 0.0, 0.0]))
            cur_idx += 1
        for _ in range(3):
            bb_res.append((cur_idx, "GLY", [997.0, 997.0, 997.0]))
            cur_idx += 1
        for r in range(20, 26):
            bb_res.append((cur_idx, "ALA", [float(r) + 40.0, 0.0, 0.0]))
            cur_idx += 1
        for _ in range(3):
            bb_res.append((cur_idx, "GLY", [996.0, 996.0, 996.0]))
            cur_idx += 1
        for r in range(11, 21):
            bb_res.append((cur_idx, "VAL", [0.0, float(r), 0.0]))
            cur_idx += 1

        create_sample_pdb(bb_pdb, {"A": bb_res})

        scaffolder = ChimeraScaffolder(
            donor_pdb=donor_pdb,
            vessel_pdb=vessel_pdb,
            donor_id="D",
            vessel_id="V",
            motif_residues="all",
            chimera_mode="replace",
            vessel_replace_range="11-15",
            lock_vessel=True,
            out_dir=os.path.join(self.test_dir, "out_mask"),
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        scaffolder.parse_motif_ranges()
        scaffolder.preserved_vessel_segments = [(1, 10), (11, 20)]

        fixed_dict = scaffolder.map_fixed_positions_for_mpnn(bb_pdb)
        fixed_indices = set(fixed_dict["A"])

        expected_fixed = set(range(1, 11)).union(range(14, 19), range(22, 27), range(30, 36), range(39, 49))
        self.assertEqual(fixed_indices, expected_fixed)

        unfixed_indices = set(range(1, 49)) - fixed_indices
        expected_unfixed = {11, 12, 13, 19, 20, 21, 27, 28, 29, 36, 37, 38}
        self.assertEqual(unfixed_indices, expected_unfixed)

    def test_all_four_modes_with_discontinuous_donor(self):
        """Test master contig assembly for replace, ligate_nterm, ligate_cterm, insert_loop with discontinuous donor."""
        disc_pdb = os.path.join(self.test_dir, "modes_disc_donor.pdb")
        seg1 = [(r, "ALA", [0.0, 0.0, 0.0]) for r in range(1, 6)]
        seg2 = [(r, "GLY", [15.0 + float(r - 10), 0.0, 0.0]) for r in range(10, 16)]
        create_sample_pdb(disc_pdb, {"D": seg1 + seg2})

        modes = ["replace", "ligate_nterm", "ligate_cterm", "insert_loop"]
        for mode in modes:
            scaffolder = ChimeraScaffolder(
                donor_pdb=disc_pdb,
                vessel_pdb=self.vessel_pdb,
                donor_id="D",
                vessel_id="B",
                motif_residues="all",
                chimera_mode=mode,
                vessel_replace_range="40-60",
                vessel_split_after=50,
                linker_n="8-12",
                linker_c="8-12",
                clamp_linkers=False,
                out_dir=os.path.join(self.test_dir, f"out_mode_{mode}"),
                dry_run=True,
            )
            scaffolder.prepare_input_structures()
            contig, donor_segs, vessel_segs = scaffolder.compute_contigs_and_segments()

            # Verify no empty components or consecutive slashes
            self.assertNotIn("//", contig)
            self.assertFalse(contig.startswith("/"))
            self.assertFalse(contig.endswith("/"))

            # Verify donor segments present (unified Chain A with offset 1000 due to vessel 1-100)
            self.assertEqual(donor_segs, [(1, 5), (10, 15)])
            self.assertIn("A1001-1005", contig)
            self.assertIn("A1010-1015", contig)

            # Verify internal linker is embedded between A1001-1005 and A1010-1015
            # 15.0 Å void -> L_glob = ceil(15/3) = 5 -> linker is 5-11
            self.assertIn("A1001-1005/5-11/A1010-1015", contig)

            if mode == "replace":
                self.assertTrue(contig.startswith("A1-39/"))
                self.assertTrue(contig.endswith("/A61-100"))
            elif mode == "ligate_nterm":
                self.assertTrue(contig.startswith("A1001-1005/5-11/A1010-1015/"))
                self.assertTrue(contig.endswith("/A1-100"))
            elif mode == "ligate_cterm":
                self.assertTrue(contig.startswith("A1-100/"))
                self.assertTrue(contig.endswith("/A1001-1005/5-11/A1010-1015"))
            elif mode == "insert_loop":
                self.assertTrue(contig.startswith("A1-50/"))
                self.assertTrue(contig.endswith("/A51-100"))

    def test_resolve_python_interpreter_hierarchy(self):
        """Verify 5-tier fallback hierarchy in resolve_python_interpreter and resolve_neural_python."""
        # Tier 1: Explicit CLI binary takes precedence
        explicit_bin = sys.executable
        res_bin, valid = resolve_python_interpreter(
            explicit_binary=explicit_bin,
            env_var_keys=["NONEXISTENT_KEY_XYZ"],
            conda_env_names=["nonexistent_env"],
            required_modules=["sys", "os"],
        )
        self.assertEqual(res_bin, os.path.realpath(explicit_bin))
        self.assertTrue(valid)

        # Explicit non-existent path
        bogus_path = os.path.join(self.test_dir, "nonexistent_python_binary")
        res_bogus, bogus_valid = resolve_python_interpreter(explicit_binary=bogus_path, required_modules=[])
        self.assertEqual(res_bogus, os.path.realpath(bogus_path))
        self.assertFalse(bogus_valid)

        # Tier 2: Environment variable override
        test_var_name = "TEST_RFDIFF_PYTHON_VAR"
        os.environ[test_var_name] = sys.executable
        try:
            res_env, valid_env = resolve_python_interpreter(
                explicit_binary=None, env_var_keys=[test_var_name], conda_env_names=[], required_modules=["sys"]
            )
            self.assertEqual(res_env, os.path.realpath(sys.executable))
            self.assertTrue(valid_env)
        finally:
            os.environ.pop(test_var_name, None)

        # Tier 5: Host interpreter fallback when no CLI or env vars provided
        res_fallback, valid_fallback = resolve_neural_python(
            explicit_path=None,
            env_var_names=["NONEXISTENT_KEY_12345"],
            target_env_names=["definitely_nonexistent_conda_env_999"],
            required_modules=["sys"],
        )
        self.assertEqual(res_fallback, os.path.realpath(sys.executable))
        self.assertTrue(valid_fallback)

    def test_probe_python_environment(self):
        """Verify module presence detection and fail-fast exception handling in probe_python_environment."""
        # Standard library modules must succeed
        res = probe_python_environment(
            sys.executable, required_modules=["sys", "os", "json"], check_cuda=False, raise_on_error=False
        )
        self.assertEqual(res["missing_modules"], [])

        # Nonexistent module detection
        res_missing = probe_python_environment(
            sys.executable, required_modules=["nonexistent_module_xyz_123456"], check_cuda=False, raise_on_error=False
        )
        self.assertIn("nonexistent_module_xyz_123456", res_missing["missing_modules"])

        # raise_on_error=True must raise EnvironmentError with formatted alert
        with self.assertRaises(EnvironmentError) as ctx:
            probe_python_environment(
                sys.executable,
                required_modules=["nonexistent_module_xyz_123456"],
                check_cuda=False,
                raise_on_error=True,
                tool_name="RFdiffusion",
                suggested_env="SE3nv",
            )
        self.assertIn("nonexistent_module_xyz_123456", str(ctx.exception))

    def test_probe_python_environment_caching(self):
        """Verify probe_python_environment caches probe results in _PROBE_CACHE and avoids subprocess calls."""
        clear_probe_cache()
        real_bin = os.path.realpath(sys.executable)
        req_mods = ["os", "sys"]
        cache_key = (real_bin, tuple(sorted(req_mods)))

        self.assertNotIn(cache_key, _PROBE_CACHE)

        # First run populates cache
        res1 = probe_python_environment(real_bin, required_modules=req_mods, check_cuda=False)
        self.assertEqual(res1["missing_modules"], [])
        self.assertIn(cache_key, _PROBE_CACHE)

        # Second run hits cache without spawning a new subprocess
        from unittest.mock import patch

        with patch("subprocess.run") as mock_run:
            res2 = probe_python_environment(real_bin, required_modules=req_mods, check_cuda=False)
            mock_run.assert_not_called()
            self.assertEqual(res2["missing_modules"], [])

        # clear_probe_cache() empties _PROBE_CACHE
        clear_probe_cache()
        self.assertNotIn(cache_key, _PROBE_CACHE)

    def test_path_deduplication_validate_output_dir(self):
        """Verify lexical de-duplication of nested /outputs/outputs/ path segments."""
        # Relative repeated outputs
        p1 = validate_output_dir("outputs/outputs/chimera_scaffold")
        self.assertNotIn("outputs/outputs", p1.replace("\\", "/"))

        # Deep repeated outputs
        p2 = validate_output_dir("outputs/outputs/outputs/run1")
        self.assertNotIn("outputs/outputs", p2.replace("\\", "/"))

        # Explicit absolute path with duplicate outputs
        abs_repeated = os.path.join(self.test_dir, "outputs", "outputs", "design_1")
        p3 = validate_output_dir(abs_repeated)
        self.assertNotIn("outputs/outputs", p3.replace("\\", "/"))

        # ChimeraScaffolder constructor normalization
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            out_dir=os.path.join(self.test_dir, "outputs", "outputs", "scaffold_run"),
            dry_run=True,
        )
        self.assertNotIn("outputs/outputs", scaffolder.out_dir.replace("\\", "/"))

    def test_preflight_alert_formatting(self):
        """Verify alert box rendering contains structured diagnostic and remediation text."""
        alert = display_dependency_alert(
            tool_name="RFdiffusion",
            python_bin="/home/user/miniconda3/bin/python",
            missing_modules=["torch", "hydra"],
            suggested_env="SE3nv",
        )
        self.assertIn("[AUTOSCAFFOLD ERROR]", alert)
        self.assertIn("RFdiffusion", alert)
        self.assertIn("torch, hydra", alert)
        self.assertIn("conda activate SE3nv", alert)
        self.assertIn("--python_rfdiff", alert)
        self.assertIn("PYTHON_RFDIFF", alert)

        cuda_warn = display_cuda_warning(
            python_bin="/home/user/miniconda3/envs/SE3nv/bin/python", tool_name="RFdiffusion"
        )
        self.assertIn("NO GPU / CUDA ACCELERATION DETECTED", cuda_warn)
        self.assertIn("RFdiffusion", cuda_warn)

    def test_dry_run_with_custom_interpreters(self):
        """Verify full dry run pipeline succeeds when passing custom neural python interpreters."""
        out_scaffold = os.path.join(self.test_dir, "custom_interp_out")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motif_residues="30-40",
            chimera_mode="replace",
            vessel_replace_range="40-60",
            linker_n="8-12",
            linker_c="8-12",
            clamp_linkers=False,
            python_rfdiff=sys.executable,
            python_mpnn=sys.executable,
            out_dir=out_scaffold,
            dry_run=True,
        )
        self.assertEqual(scaffolder.python_rfdiff, os.path.realpath(sys.executable))
        self.assertEqual(scaffolder.python_mpnn, os.path.realpath(sys.executable))

        res = scaffolder.run_pipeline()
        self.assertIn("rfdiffusion_command", res)
        # Verify rfdiffusion_command starts with resolved python_rfdiff
        self.assertEqual(res["rfdiffusion_command"][0], os.path.realpath(sys.executable))
        self.assertTrue(len(res["candidates"]) > 0)
        # Verify mpnn_command starts with resolved python_mpnn
        self.assertEqual(res["candidates"][0]["mpnn_command"][0], os.path.realpath(sys.executable))


class TestSecurityHardening(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_safe_session_unpickler_blocks_rce(self):
        """Verify SafeSessionUnpickler rejects arbitrary class instantiation and RCE payloads."""
        import pickle
        import os
        import subprocess

        # Test built-in functions
        for dangerous_target in [eval, exec, __import__, os.system, subprocess.Popen]:
            payload = pickle.dumps(dangerous_target)
            with self.assertRaises(pickle.UnpicklingError):
                safe_pickle_loads(payload)

    def test_safe_session_unpickler_allows_safe_primitives_and_numpy(self):
        """Verify SafeSessionUnpickler permits safe scientific data structures, primitives, and numpy arrays."""
        import pickle

        safe_obj = {
            "name": "test_session",
            "values": [1, 2.5, "alpha", True, b"raw_bytes"],
            "coords": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
            "nested": {"a": (1, 2, 3)},
        }
        pickled = pickle.dumps(safe_obj)
        unpickled = safe_pickle_loads(pickled)
        self.assertEqual(unpickled["name"], "test_session")
        self.assertEqual(unpickled["values"], [1, 2.5, "alpha", True, b"raw_bytes"])
        np.testing.assert_allclose(unpickled["coords"], safe_obj["coords"])

    def test_try_extract_pse_prevalidation_blocks_malicious_session(self):
        """Verify try_extract_pse_with_pymol and validate_pse_safety block malicious .pse files."""
        import pickle
        import gzip
        from scripts.pse_pdb_extractor import try_extract_pse_with_pymol

        malicious_pse = os.path.join(self.test_dir, "malicious.pse")
        malicious_payload = gzip.compress(pickle.dumps(os.system))
        with open(malicious_pse, "wb") as f:
            f.write(malicious_payload)

        # validate_pse_safety must raise an exception
        with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
            validate_pse_safety(malicious_pse)

        # try_extract_pse_with_pymol must also reject it
        with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
            try_extract_pse_with_pymol(malicious_pse)

    def test_safe_join_path_prevents_directory_traversal(self):
        """Verify safe_join_path strictly bounds output paths within the base directory."""
        base = os.path.join(self.test_dir, "base_dir")
        os.makedirs(base, exist_ok=True)

        # Normal child path
        child = safe_join_path(base, "valid_file.pdb")
        self.assertTrue(os.path.commonpath([base, child]) == os.path.realpath(base))

        # Relative traversal within filename
        traversal = safe_join_path(base, "../../etc/passwd")
        # Basename sanitization strips ../../ and retains sanitized basename strictly inside base
        self.assertTrue(os.path.commonpath([base, traversal]) == os.path.realpath(base))
        self.assertTrue(traversal.startswith(os.path.realpath(base)))

    def test_validate_output_dir_rejects_dangerous_paths(self):
        """Verify validate_output_dir rejects root, empty, and system directories."""
        with self.assertRaises(ValueError):
            validate_output_dir("")

        with self.assertRaises(ValueError):
            validate_output_dir("   ")

        with self.assertRaises(ValueError):
            validate_output_dir("/")

        with self.assertRaises(ValueError):
            validate_output_dir("C:\\")

        with self.assertRaises(ValueError):
            validate_output_dir("C:")

        with self.assertRaises(ValueError):
            validate_output_dir("C:/")

        with self.assertRaises(ValueError):
            validate_output_dir("/etc")

        with self.assertRaises(ValueError):
            validate_output_dir("/usr/local")

        # Valid directory should succeed
        valid_dir = os.path.join(self.test_dir, "valid_output")
        res = validate_output_dir(valid_dir)
        self.assertEqual(res, os.path.realpath(os.path.abspath(valid_dir)))

    def test_probe_python_environment_rejects_invalid_binary(self):
        """Verify probe_python_environment safely rejects nonexistent or argument-injected binaries."""
        res = probe_python_environment(
            "nonexistent_python_binary_arg -c evil", required_modules=["torch"], raise_on_error=False
        )
        self.assertEqual(res["missing_modules"], ["torch"])

        with self.assertRaises(EnvironmentError):
            probe_python_environment(
                "invalid_python_binary_path --flag", required_modules=["torch"], raise_on_error=True
            )

    def test_chimera_scaffolder_rejects_path_traversal_ids(self):
        """Verify ChimeraScaffolder rejects donor_id and vessel_id with directory traversal characters."""
        with self.assertRaises(ValueError):
            ChimeraScaffolder(
                donor_pdb=self.donor_pdb if hasattr(self, "donor_pdb") else "donor.pdb",
                vessel_pdb=self.vessel_pdb if hasattr(self, "vessel_pdb") else "vessel.pdb",
                donor_id="../../etc",
                motif_residues="30-40",
            )

        with self.assertRaises(ValueError):
            ChimeraScaffolder(
                donor_pdb="donor.pdb", vessel_pdb="vessel.pdb", vessel_id="../bad_vessel", motif_residues="30-40"
            )

        with self.assertRaises(ValueError):
            ChimeraScaffolder(
                donor_pdb="donor.pdb", vessel_pdb="vessel.pdb", donor_id=["A", "../../evil"], motif_residues="30-40"
            )

    def test_extract_and_merge_objects_rejects_traversal_target_ids(self):
        """Verify extract_and_merge_objects rejects target_ids containing traversal strings."""
        sample_pdb = os.path.join(self.test_dir, "sample.pdb")
        create_sample_pdb(sample_pdb, {"A": [(1, "ALA", [0.0, 0.0, 0.0])]})
        out_sub = os.path.join(self.test_dir, "merge_out")

        with self.assertRaises(ValueError):
            extract_and_merge_objects(sample_pdb, "../../etc", out_sub)

        with self.assertRaises(ValueError):
            extract_and_merge_objects(sample_pdb, ["A", "../evil"], out_sub)

    def test_runner_spec_build_command_rejects_shell_operators(self):
        """Verify RunnerSpec.build_command rejects shell operators, pipes, and semicolons."""
        runner = RunnerSpec(runner_type="direct", command_prefix=["python; rm -rf /"])
        with self.assertRaises(ValueError):
            runner.build_command("script.py")

        runner_pipe = RunnerSpec(runner_type="custom", command_prefix=["python", "|", "cat"])
        with self.assertRaises(ValueError):
            runner_pipe.build_command("script.py")

        runner_safe = RunnerSpec(runner_type="direct", command_prefix=[sys.executable])
        with self.assertRaises(ValueError):
            runner_safe.build_command("script.py; rm -rf /")

        with self.assertRaises(ValueError):
            runner_safe.build_command("script.py", "--flag", "val; evil")

    def test_probe_python_environment_rejects_shell_operators(self):
        """Verify probe_python_environment rejects shell operators, pipes, and semicolons."""
        with self.assertRaises(EnvironmentError):
            probe_python_environment("python; rm -rf /", raise_on_error=True)

        with self.assertRaises(EnvironmentError):
            probe_python_environment(["python", "|", "sh"], raise_on_error=True)

        res = probe_python_environment("python; echo pwned", required_modules=["torch"], raise_on_error=False)
        self.assertIn("torch", res["missing_modules"])
        probe_evil = probe_python_environment("python; evil", required_modules=["torch"], raise_on_error=False)
        self.assertIn("torch", probe_evil["missing_modules"])

    def test_get_all_environment_roots_excludes_root_directories(self):
        """Verify get_all_environment_roots canonicalizes paths and excludes root directory escapes."""
        from scripts.chimera_scaffold import get_all_environment_roots
        roots = get_all_environment_roots()
        for r in roots:
            self.assertNotIn(r, ["/", "\\", "C:\\", "C:/", "\\\\"])
            p_drive, p_rest = os.path.splitdrive(r)
            is_root = (
                (not p_drive and r.rstrip("/\\") == "")
                or (p_drive and p_rest in ("/", "\\", ""))
                or r in ("/", "\\", "//", "\\\\")
            )
            self.assertFalse(is_root, f"Root directory found in environment roots: {r}")

    def test_resolve_execution_runner_rejects_traversal_and_shell_injection(self):
        """Verify resolve_execution_runner rejects path traversal in target_env_names and shell injection in targets."""
        runner = resolve_execution_runner(
            explicit_target="python; evil",
            required_modules=["torch"],
        )
        self.assertFalse(runner.is_valid)

        runner_env = resolve_execution_runner(
            explicit_target=None,
            target_env_names=["../../etc", "../sensitive"],
            required_modules=["torch"],
        )
        self.assertFalse(runner_env.is_valid)


class TestPerformanceOptimizations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_kdtree_coordinate_matching_large_scale(self):
        """
        Verify KDTree-accelerated coordinate matching for large multi-domain complexes
        (1200+ residues) executes in milliseconds without O(N^2) bottlenecks.
        """
        import time

        # Generate a large 1200-residue backbone
        bb_pdb = os.path.join(self.test_dir, "large_bb_1200.pdb")
        bb_res = [(r, "ALA", [float(r % 100), float((r // 100) * 3), 0.0]) for r in range(1, 1201)]
        create_sample_pdb(bb_pdb, {"A": bb_res})

        # Donor has 150 residues matching specific coordinates
        donor_pdb = os.path.join(self.test_dir, "large_donor_150.pdb")
        donor_res = [(r, "GLY", [float((r + 400) % 100), float(((r + 400) // 100) * 3), 0.0]) for r in range(1, 151)]
        create_sample_pdb(donor_pdb, {"D": donor_res})

        # Vessel has 900 residues matching coordinates
        vessel_pdb = os.path.join(self.test_dir, "large_vessel_900.pdb")
        vessel_res = [(r, "VAL", [float(r % 100), float((r // 100) * 3), 0.0]) for r in range(1, 901)]
        create_sample_pdb(vessel_pdb, {"V": vessel_res})

        scaffolder = ChimeraScaffolder(
            donor_pdb=donor_pdb,
            vessel_pdb=vessel_pdb,
            donor_id="D",
            vessel_id="V",
            motif_residues="all",
            chimera_mode="replace",
            lock_vessel=True,
            out_dir=os.path.join(self.test_dir, "out_perf"),
            dry_run=True,
        )
        scaffolder.prepare_input_structures()
        scaffolder.parse_motif_ranges()
        scaffolder.preserved_vessel_segments = [(1, 900)]

        # Time the coordinate matching
        t0 = time.perf_counter()
        fixed_dict = scaffolder.map_fixed_positions_for_mpnn(bb_pdb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Must execute in under 100 milliseconds (typically < 10 ms with cKDTree)
        self.assertLess(elapsed_ms, 100.0, f"KDTree coordinate matching took too long: {elapsed_ms:.2f} ms")
        self.assertIn("A", fixed_dict)
        self.assertGreater(len(fixed_dict["A"]), 800)

    def test_ca_coords_o1_indexing_and_cache(self):
        """Verify _RESI_COORDS_CACHE provides O(1) residue coordinate lookups without chain_id."""
        clear_ca_coords_cache()
        sample_pdb = os.path.join(self.test_dir, "sample_indexed.pdb")
        res_list = [(r, "ALA", [float(r), 0.0, 0.0]) for r in range(1, 201)]
        create_sample_pdb(sample_pdb, {"A": res_list})

        # Query with chain_id=None
        c1 = get_residue_ca_coords(sample_pdb, chain_id=None, resi=150)
        self.assertIsNotNone(c1)
        np.testing.assert_allclose(c1, [150.0, 0.0, 0.0])

        # Verify resi_map is cached
        real_p = os.path.realpath(sample_pdb)
        mtime = os.path.getmtime(real_p)
        self.assertIn((real_p, mtime), _RESI_COORDS_CACHE)
        self.assertIn(150, _RESI_COORDS_CACHE[(real_p, mtime)])

    def test_batch_anchor_distances_vectorization(self):
        """Verify batch_anchor_distances calculates accurate vectorized Euclidean distances."""
        c1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 0.0, 0.0]])
        c2 = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 10.0], [5.0, 0.0, 0.0]])
        dists = batch_anchor_distances(c1, c2)
        np.testing.assert_allclose(dists, [5.0, 9.0, 0.0], atol=1e-5)

    def test_reusing_parsed_entities_atoms(self):
        """Verify passing entities_atoms avoids disk re-reading and yields identical results."""
        multi_pdb = os.path.join(self.test_dir, "entities_test.pdb")
        chain_a_res = [(r, "ALA", [float(r), 1.0, 2.0]) for r in range(1, 11)]
        create_sample_pdb(multi_pdb, {"A": chain_a_res})

        out1 = os.path.join(self.test_dir, "out1")
        res1 = extract_and_merge_objects(multi_pdb, "A", out1, prefix="first")
        self.assertIn("entities_atoms", res1)

        # Second merge reusing parsed entities_atoms
        out2 = os.path.join(self.test_dir, "out2")
        res2 = extract_and_merge_objects(multi_pdb, "A", out2, prefix="second", entities_atoms=res1["entities_atoms"])
        self.assertEqual(res1["total_amino_acids"], res2["total_amino_acids"])
        self.assertEqual(res1["manifest"]["segments"], res2["manifest"]["segments"])

    def test_parse_pdb_file_caching(self):
        """Verify parse_pdb_file session caching and invalidation."""
        clear_pdb_parse_cache()
        sample_pdb = os.path.join(self.test_dir, "cache_sample.pdb")
        create_sample_pdb(sample_pdb, {"A": [(1, "ALA", [0.0, 0.0, 0.0])]})

        meta1 = parse_pdb_file(sample_pdb)
        meta2 = parse_pdb_file(sample_pdb)
        self.assertEqual(meta1["objects"]["A"]["length"], meta2["objects"]["A"]["length"])

        clear_pdb_parse_cache()

    def test_probe_python_environment_runner_list(self):
        """Verify runner command prefix lists and RunnerSpec instances execute in probe_python_environment."""
        # 1. Probing with a list command prefix [sys.executable]
        res_list = probe_python_environment([sys.executable], required_modules=["sys", "os"], check_cuda=False)
        self.assertEqual(res_list["missing_modules"], [])

        # 2. Probing with RunnerSpec instance
        spec = RunnerSpec(runner_type="direct", command_prefix=[sys.executable], python_binary=sys.executable)
        res_spec = probe_python_environment(spec, required_modules=["sys"], check_cuda=False)
        self.assertEqual(res_spec["missing_modules"], [])

        # 3. Missing module detection via list command prefix
        res_missing = probe_python_environment(
            [sys.executable], required_modules=["nonexistent_mod_xyz_8888"], check_cuda=False, raise_on_error=False
        )
        self.assertIn("nonexistent_mod_xyz_8888", res_missing["missing_modules"])

    def test_ephemeral_runner_construction_and_building(self):
        """Verify ephemeral runner command construction for micromamba and conda, and RunnerSpec command building."""
        # Micromamba named environment
        cmd_mm_name = build_ephemeral_runner_cmd("micromamba", "/usr/bin/micromamba", env_name="SE3nv")
        self.assertEqual(cmd_mm_name, ["/usr/bin/micromamba", "run", "-n", "SE3nv", "python"])

        # Micromamba prefix environment
        cmd_mm_pref = build_ephemeral_runner_cmd("micromamba", "/usr/bin/micromamba", env_prefix="/opt/envs/SE3nv")
        self.assertEqual(cmd_mm_pref, ["/usr/bin/micromamba", "run", "-p", "/opt/envs/SE3nv", "python"])

        # Conda named environment (includes --no-capture-output)
        cmd_conda_name = build_ephemeral_runner_cmd("conda", "/usr/bin/conda", env_name="SE3nv")
        self.assertEqual(cmd_conda_name, ["/usr/bin/conda", "run", "-n", "SE3nv", "--no-capture-output", "python"])

        # Conda prefix environment
        cmd_conda_pref = build_ephemeral_runner_cmd("conda", "/usr/bin/conda", env_prefix="/opt/envs/SE3nv")
        expected_pref = ["/usr/bin/conda", "run", "-p", "/opt/envs/SE3nv", "--no-capture-output", "python"]
        self.assertEqual(cmd_conda_pref, expected_pref)

        # RunnerSpec build_command
        runner = RunnerSpec(
            runner_type="micromamba", command_prefix=["micromamba", "run", "-n", "SE3nv", "python"], env_name="SE3nv"
        )
        cmd = runner.build_command("run_inference.py", "--flag", "value")
        self.assertEqual(cmd, ["micromamba", "run", "-n", "SE3nv", "python", "run_inference.py", "--flag", "value"])
        self.assertIn("SE3nv", runner.display_label())

    def test_multi_candidate_resilient_probing(self):
        """Verify resolver skips candidate 1 when lacking modules and successfully resolves candidate 2."""
        from unittest.mock import patch

        cand1_dir = os.path.join(self.test_dir, "root1", "SE3nv", "bin" if sys.platform != "win32" else "Scripts")
        cand2_dir = os.path.join(self.test_dir, "root2", "SE3nv", "bin" if sys.platform != "win32" else "Scripts")
        os.makedirs(cand1_dir, exist_ok=True)
        os.makedirs(cand2_dir, exist_ok=True)

        py_name = "python.exe" if sys.platform == "win32" else "python"
        cand1_bin = os.path.join(cand1_dir, py_name)
        cand2_bin = os.path.join(cand2_dir, py_name)

        # Create mock executable files
        with open(cand1_bin, "w") as f:
            f.write("#!/bin/sh\n")
        with open(cand2_bin, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(cand1_bin, 0o755)
        os.chmod(cand2_bin, 0o755)

        roots = [os.path.join(self.test_dir, "root1"), os.path.join(self.test_dir, "root2")]

        def mock_probe(runner_or_bin, required_modules=None, **kwargs):
            target = str(runner_or_bin)
            if hasattr(runner_or_bin, "command_prefix"):
                target = " ".join(runner_or_bin.command_prefix)
            elif isinstance(runner_or_bin, (list, tuple)):
                target = " ".join(str(x) for x in runner_or_bin)

            if "root1" in target:
                return {"missing_modules": ["torch", "hydra"], "cuda_available": False}
            elif "root2" in target:
                return {
                    "missing_modules": [],
                    "cuda_available": True,
                    "torch_version": "2.0.1",
                    "cuda_device_name": "NVIDIA RTX 4090",
                }
            return {"missing_modules": required_modules or [], "cuda_available": False}

        with patch("scripts.chimera_scaffold.get_all_environment_roots", return_value=roots), patch(
            "scripts.chimera_scaffold.probe_python_environment", side_effect=mock_probe
        ):
            clear_probe_cache()
            runner = resolve_execution_runner(
                explicit_target=None,
                env_var_keys=[],
                target_env_names=["SE3nv"],
                required_modules=["torch", "hydra"],
                tool_label="RFdiffusion",
            )
            self.assertTrue(runner.is_valid)
            self.assertTrue(runner.cuda_available)
            self.assertIn("root2", runner.display_label())

    def test_terminal_scrollback_preservation_configuration(self):
        """Verify shell scripts preserve terminal scrollback buffer and do not contain unconditional clear."""
        import re

        scripts_to_check = [
            os.path.join(REPO_ROOT, "autoscaffold_suite.sh"),
            os.path.join(REPO_ROOT, "install.sh"),
            os.path.join(REPO_ROOT, "autoscaffold.sh"),
            os.path.join(REPO_ROOT, "chimera_scaffold.sh"),
            os.path.join(REPO_ROOT, "autoscaffold"),
        ]

        # autoscaffold launcher must export AUTOSCAFFOLD_NO_CLEAR=1 by default
        with open(os.path.join(REPO_ROOT, "autoscaffold"), "r", encoding="utf-8") as f:
            launcher_content = f.read()
        self.assertIn("AUTOSCAFFOLD_NO_CLEAR", launcher_content)

        # Every occurrence of clear in any script must be guarded by AUTOSCAFFOLD_CLEAR or AUTOSCAFFOLD_NO_CLEAR
        for script_path in scripts_to_check:
            if not os.path.isfile(script_path):
                continue
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                # Ignore comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(r"\bclear\b", line):
                    # Check that the surrounding context checks AUTOSCAFFOLD_NO_CLEAR or AUTOSCAFFOLD_CLEAR
                    start_idx = max(0, idx - 3)
                    end_idx = idx + 1
                    context = "\n".join(lines[start_idx:end_idx])
                    self.assertTrue(
                        "AUTOSCAFFOLD_CLEAR" in context or "AUTOSCAFFOLD_NO_CLEAR" in context,
                        f"Unshielded clear command at {script_path}:{idx + 1}: {line}",
                    )

    def test_environment_managers_and_roots_memoization(self):
        """Verify find_environment_managers and get_all_environment_roots cache results."""
        from unittest.mock import patch
        import scripts.chimera_scaffold as cs

        clear_environment_cache()
        self.assertIsNone(cs._CACHED_MANAGERS)
        self.assertIsNone(cs._CACHED_ROOTS)

        # First call populates cache
        mgrs1 = find_environment_managers()
        roots1 = get_all_environment_roots()
        self.assertIsNotNone(cs._CACHED_MANAGERS)
        self.assertIsNotNone(cs._CACHED_ROOTS)

        # Subsequent calls hit cache without scanning
        with patch("shutil.which", side_effect=RuntimeError("Should not be called when cached")):
            mgrs2 = find_environment_managers()
            self.assertEqual(mgrs1, mgrs2)

        with patch("os.path.isdir", side_effect=RuntimeError("Should not be called when cached")):
            roots2 = get_all_environment_roots()
            self.assertEqual(roots1, roots2)

        # clear_environment_cache resets cache
        clear_environment_cache()
        self.assertIsNone(cs._CACHED_MANAGERS)
        self.assertIsNone(cs._CACHED_ROOTS)

    def test_probe_python_environment_dual_cache_indexing(self):
        """Verify probe_python_environment dual-indexes tuple(cmd_prefix) and real_bin in O(1)."""
        from unittest.mock import patch
        import scripts.chimera_scaffold as cs

        clear_probe_cache()
        host_exe = os.path.realpath(sys.executable)

        # Probe with string path
        res_str = probe_python_environment(host_exe, required_modules=["json"], check_cuda=False)
        self.assertEqual(res_str["missing_modules"], [])

        # Verify dual keys exist in cache
        sorted_reqs = ("json",)
        tuple_key = ((host_exe,), sorted_reqs)
        bin_key = (host_exe, sorted_reqs)
        self.assertIn(tuple_key, cs._PROBE_CACHE)
        self.assertIn(bin_key, cs._PROBE_CACHE)

        # Now probe via RunnerSpec and list - should hit cache without calling subprocess.run
        with patch("subprocess.run", side_effect=RuntimeError("Subprocess should not be invoked on cached entry")):
            spec = RunnerSpec(runner_type="direct", command_prefix=[host_exe], python_binary=host_exe)
            res_spec = probe_python_environment(spec, required_modules=["json"], check_cuda=False)
            self.assertEqual(res_spec["missing_modules"], [])

            res_list = probe_python_environment([host_exe], required_modules=["json"], check_cuda=False)
            self.assertEqual(res_list["missing_modules"], [])

    def test_resolve_execution_runner_early_exit_on_cuda(self):
        """Verify resolve_execution_runner short-circuits immediately when a candidate has CUDA available."""
        from unittest.mock import patch

        cand1_dir = os.path.join(self.test_dir, "early_root1", "SE3nv", "bin" if sys.platform != "win32" else "Scripts")
        cand2_dir = os.path.join(self.test_dir, "early_root2", "SE3nv", "bin" if sys.platform != "win32" else "Scripts")
        os.makedirs(cand1_dir, exist_ok=True)
        os.makedirs(cand2_dir, exist_ok=True)

        py_name = "python.exe" if sys.platform == "win32" else "python"
        cand1_bin = os.path.join(cand1_dir, py_name)
        cand2_bin = os.path.join(cand2_dir, py_name)

        with open(cand1_bin, "w") as f:
            f.write("#!/bin/sh\n")
        with open(cand2_bin, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(cand1_bin, 0o755)
        os.chmod(cand2_bin, 0o755)

        roots = [os.path.join(self.test_dir, "early_root1"), os.path.join(self.test_dir, "early_root2")]

        probed_candidates = []

        def tracking_probe(runner_or_bin, required_modules=None, **kwargs):
            target = str(runner_or_bin)
            probed_candidates.append(target)
            if "early_root1" in target:
                return {
                    "missing_modules": [],
                    "cuda_available": True,
                    "torch_version": "2.2.0",
                    "cuda_device_name": "NVIDIA RTX 4090",
                }
            elif "early_root2" in target:
                return {
                    "missing_modules": [],
                    "cuda_available": True,
                    "torch_version": "2.2.0",
                    "cuda_device_name": "NVIDIA RTX 4090",
                }
            return {"missing_modules": required_modules or [], "cuda_available": False}

        with patch("scripts.chimera_scaffold.get_all_environment_roots", return_value=roots), patch(
            "scripts.chimera_scaffold.probe_python_environment", side_effect=tracking_probe
        ):
            clear_probe_cache()
            runner = resolve_execution_runner(
                explicit_target=None,
                env_var_keys=[],
                target_env_names=["SE3nv"],
                required_modules=["torch"],
                tool_label="RFdiffusion",
            )
            self.assertTrue(runner.is_valid)
            self.assertTrue(runner.cuda_available)
            self.assertIn("early_root1", runner.display_label())

            # Crucial assertion: early_root2 was NEVER probed because early_root1 had cuda_available=True
            root2_probes = [p for p in probed_candidates if "early_root2" in p]
            self.assertEqual(len(root2_probes), 0, f"Remaining candidates probed despite CUDA hit: {root2_probes}")

    def test_chimera_scaffolder_mpnn_fallback_to_rfdiff_runner(self):
        """Verify ChimeraScaffolder automatically falls back to rfdiff_runner if mpnn environment lacks torch."""
        from scripts.chimera_scaffold import ChimeraScaffolder, RunnerSpec

        valid_rfdiff_runner = RunnerSpec(
            runner_type="direct",
            command_prefix=["/path/to/SE3nv/bin/python"],
            python_binary="/path/to/SE3nv/bin/python",
            is_valid=True,
            cuda_available=True,
            torch_version="2.2.0",
        )
        invalid_mpnn_runner = RunnerSpec(
            runner_type="direct",
            command_prefix=["/path/to/mlfold/bin/python"],
            python_binary="/path/to/mlfold/bin/python",
            is_valid=False,
            missing_modules=["torch"],
        )

        def mock_resolve(explicit_target=None, **kwargs):
            tool = kwargs.get("tool_label", "")
            if "RFdiffusion" in tool:
                return valid_rfdiff_runner
            return invalid_mpnn_runner

        dummy_pdb = os.path.join(self.test_dir, "fallback_test.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n")

        with patch("scripts.chimera_scaffold.resolve_execution_runner", side_effect=mock_resolve), patch(
            "scripts.chimera_scaffold.probe_python_environment",
            return_value={"missing_modules": [], "cuda_available": True, "torch_version": "2.2.0"},
        ):
            scaffolder = ChimeraScaffolder(
                input_file=dummy_pdb,
                donor_id="D",
                vessel_id="V",
                motif_residues="30-40",
                chimera_mode="replace",
                vessel_replace_range="40-60",
                python_mpnn="/path/to/mlfold/bin/python",
                python_rfdiff="/path/to/SE3nv/bin/python",
                dry_run=True,
                strict_python=False,
            )
            # Automatic fallback should have taken place
            self.assertTrue(scaffolder.mpnn_env_valid)
            self.assertEqual(scaffolder.mpnn_runner, valid_rfdiff_runner)
            self.assertEqual(scaffolder.python_mpnn, "/path/to/SE3nv/bin/python")

            # With strict_python=True, fallback should NOT occur
            scaffolder_strict = ChimeraScaffolder(
                input_file=dummy_pdb,
                donor_id="D",
                vessel_id="V",
                motif_residues="30-40",
                chimera_mode="replace",
                vessel_replace_range="40-60",
                python_mpnn="/path/to/mlfold/bin/python",
                python_rfdiff="/path/to/SE3nv/bin/python",
                dry_run=True,
                strict_python=True,
            )
            self.assertFalse(scaffolder_strict.mpnn_env_valid)
            self.assertEqual(scaffolder_strict.mpnn_runner, invalid_mpnn_runner)

    def test_extract_and_thread_mpnn_sequences(self):
        """Verify extract_mpnn_sequences and thread_sequence_onto_pdb generate complete 3D PDBs with designed AAs."""
        from scripts.chimera_scaffold import extract_mpnn_sequences, thread_sequence_onto_pdb

        # Create mock FASTA with native backbone + designed sequence
        mock_fa = os.path.join(self.test_dir, "test_design.fa")
        with open(mock_fa, "w", encoding="utf-8") as f:
            f.write(">test_design, score=1.5\nAAAA\n>test_design_seq1, score=0.8\nKWER\n")

        seqs = extract_mpnn_sequences(mock_fa)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(seqs[0][1], "KWER")

        # Create mock backbone PDB
        mock_bb = os.path.join(self.test_dir, "test_bb.pdb")
        with open(mock_bb, "w", encoding="utf-8") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n")
            f.write("ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00 20.00           C\n")
            f.write("ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00 20.00           C\n")
            f.write("ATOM      4  CA  ALA A   4      11.400   0.000   0.000  1.00 20.00           C\n")

        threaded_pdb = os.path.join(self.test_dir, "threaded.pdb")
        res = thread_sequence_onto_pdb(mock_bb, "KWER", threaded_pdb)
        self.assertTrue(res)
        self.assertTrue(os.path.isfile(threaded_pdb))

        with open(threaded_pdb, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.startswith("ATOM")]
            self.assertIn("LYS", lines[0])
            self.assertIn("TRP", lines[1])
            self.assertIn("GLU", lines[2])
            self.assertIn("ARG", lines[3])

    def test_methodology_reporter_prevents_circular_symlinks(self):
        """Verify methodology_reporter produces valid regular files and never circular symlinks."""
        from scripts.methodology_reporter import generate_methodology_report

        out_rep = os.path.join(self.test_dir, "methodology_test")
        os.makedirs(out_rep, exist_ok=True)

        # Run report generation twice
        generate_methodology_report(
            output_dir=out_rep,
            run_name="TestRun1",
            pipeline_mode="Chimera Scaffolding",
        )
        generate_methodology_report(
            output_dir=out_rep,
            run_name="TestRun2",
            pipeline_mode="Chimera Scaffolding",
        )

        f1 = os.path.join(out_rep, "run_methodology.txt")
        f2 = os.path.join(out_rep, "methodology.txt")
        self.assertTrue(os.path.isfile(f1))
        self.assertTrue(os.path.isfile(f2))
        self.assertFalse(os.path.islink(f1))


if __name__ == "__main__":
    unittest.main()
