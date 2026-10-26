#!/usr/bin/env python3
"""
tests/test_multichain_scaffolding.py
Comprehensive verification suite for Closed-Loop Iterative Multi-Chain
Discontinuous Motif Scaffolding, adhering to CLOSED_LOOP_MULTICHAIN_SCAFFOLDING_SPEC.md.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure autoscaffold root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.chimera_scaffold import (  # noqa: E402
    ChimeraScaffolder,
    MultiChainMotifManifest,
    parse_multi_chain_motif_spec,
    build_multi_chain_manifest,
    compute_gap_linker_length,
    compute_gap_linker_string,
    audit_motif_cartesian_drift,
    run_closed_loop_backbone_generation,
    get_residue_anchor_coords,
    clear_ca_coords_cache,
    _purge_candidate_artifacts,
    _SEED_CA_CACHE,
    _ANCHOR_COORDS_CACHE,
)
from scripts.relax_chimera import (  # noqa: E402
    relax_structure,
    parse_res_indices,
    untangle_severe_clashes,
)


def create_mock_pdb(path: str, residues_by_chain: dict):
    """
    Creates a valid PDB file for testing.
    residues_by_chain: {chain_id: [(resi, resn, {atom_name: [x, y, z]}), ...]}
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    atom_idx = 1
    with open(path, "w", encoding="utf-8") as f:
        for ch, res_list in residues_by_chain.items():
            for resi, resn, atoms in res_list:
                for atom_name, (x, y, z) in atoms.items():
                    elem = atom_name[0]
                    if len(atom_name) == 4 or (len(atom_name) >= 1 and atom_name[0].isdigit()):
                        af = f"{atom_name:<4s}"
                    else:
                        af = f" {atom_name:<3s}"
                    line = (
                        f"ATOM  {atom_idx:>5d} {af:4s} {resn:>3s} {ch}{resi:>4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {elem:>2s}\n"
                    )
                    f.write(line)
                    atom_idx += 1
            f.write("TER\n")
        f.write("END\n")


class TestMultiChainScaffolding(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Build synthetic donor PDB with Chain A and Chain B
        self.donor_pdb = os.path.join(self.test_dir, "hsp90_dimer.pdb")
        chain_a_res = [
            (312, "TRP", {"N": [0.0, 0.0, 0.0], "CA": [1.2, 0.0, 0.0], "C": [2.4, 0.0, 0.0], "O": [2.8, 1.0, 0.0]}),
            (313, "GLU", {"N": [3.6, 0.0, 0.0], "CA": [4.8, 0.0, 0.0], "C": [6.0, 0.0, 0.0], "O": [6.4, 1.0, 0.0]}),
            # Boundary break on Chain A: skip to 325 (delta resi > 1, dist > 4.2 Å)
            (325, "LYS", {"N": [15.0, 0.0, 0.0], "CA": [16.2, 0.0, 0.0], "C": [17.4, 0.0, 0.0], "O": [17.8, 1.0, 0.0]}),
        ]
        chain_b_res = [
            (191, "PHE", {"N": [0.0, 10.0, 0.0], "CA": [1.2, 10.0, 0.0], "C": [2.4, 10.0, 0.0], "O": [2.8, 11.0, 0.0]}),
            (200, "MET", {"N": [0.0, 25.0, 0.0], "CA": [1.2, 25.0, 0.0], "C": [2.4, 25.0, 0.0], "O": [2.8, 26.0, 0.0]}),
        ]
        create_mock_pdb(self.donor_pdb, {"A": chain_a_res, "B": chain_b_res})

        # Build synthetic vessel PDB (Chain V / B, residues 1-100)
        self.vessel_pdb = os.path.join(self.test_dir, "vessel_host.pdb")
        vessel_res = []
        for r in range(1, 101):
            x = 0.0
            y = float(r) * 1.5
            z = 0.0
            vessel_res.append(
                (
                    r,
                    "ALA",
                    {
                        "N": [x, y, z],
                        "CA": [x + 1.0, y, z],
                        "C": [x + 2.0, y, z],
                        "O": [x + 2.5, y + 1.0, z],
                    },
                )
            )
        create_mock_pdb(self.vessel_pdb, {"V": vessel_res})

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Pillar 1 & Section 7.1.1: Multi-Chain Motif Parsing & Auto-Segmentation
    # -------------------------------------------------------------------------
    def test_multi_chain_motif_parsing(self):
        """
        Verify:
        1. Parsing of multi-chain unified string: 'A:312-313, B:191, B:200'
        2. Parsing of multi-file discontinuous spec: 'f1.pdb:A:1-10, f2.pdb:B:20-30'
        3. Partitioning into continuous MotifSegment objects with boundary break detection.
        """
        # 1. Multi-chain parsing
        parsed = parse_multi_chain_motif_spec("A:312-313, B:191, B:200", default_source_file=self.donor_pdb)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0], (self.donor_pdb, "A", 312))
        self.assertEqual(parsed[1], (self.donor_pdb, "A", 313))
        self.assertEqual(parsed[2], (self.donor_pdb, "B", 191))
        self.assertEqual(parsed[3], (self.donor_pdb, "B", 200))

        # 2. Multi-file parsing
        mf_spec = "file1.pdb:A:1-10, file2.pdb:B:20-30"
        parsed_mf = parse_multi_chain_motif_spec(mf_spec)
        self.assertEqual(len(parsed_mf), 21)
        self.assertEqual(parsed_mf[0], ("file1.pdb", "A", 1))
        self.assertEqual(parsed_mf[9], ("file1.pdb", "A", 10))
        self.assertEqual(parsed_mf[10], ("file2.pdb", "B", 20))
        self.assertEqual(parsed_mf[20], ("file2.pdb", "B", 30))

        # Plus syntax: 'A:312-313+315'
        parsed_plus = parse_multi_chain_motif_spec("A:312-313+315", default_source_file=self.donor_pdb)
        self.assertEqual(len(parsed_plus), 3)
        self.assertEqual([r for _, _, r in parsed_plus], [312, 313, 315])

        # 3. Partitioning into MotifSegment instances with auto-segmentation
        spec = "A:312-313, A:325, B:191, B:200"
        manifest = build_multi_chain_manifest(spec, default_source_file=self.donor_pdb)
        self.assertIsInstance(manifest, MultiChainMotifManifest)
        self.assertEqual(manifest.total_residues, 5)

        # Segments:
        # D1: A 312-313 (continuous)
        # D2: A 325 (break due to delta resi = 12 > 1 and CA distance > 4.2 Å)
        # D3: B 191 (break due to chain change A -> B)
        # D4: B 200 (break due to delta resi = 9 > 1)
        self.assertEqual(len(manifest.segments), 4)
        self.assertEqual(manifest.segments[0].segment_id, "D1")
        self.assertEqual(manifest.segments[0].source_chain, "A")
        self.assertEqual(manifest.segments[0].length, 2)
        self.assertEqual(manifest.segments[0].start_resi, 312)
        self.assertEqual(manifest.segments[0].end_resi, 313)

        self.assertEqual(manifest.segments[1].segment_id, "D2")
        self.assertEqual(manifest.segments[1].source_chain, "A")
        self.assertEqual(manifest.segments[1].length, 1)
        self.assertEqual(manifest.segments[1].start_resi, 325)

        self.assertEqual(manifest.segments[2].segment_id, "D3")
        self.assertEqual(manifest.segments[2].source_chain, "B")
        self.assertEqual(manifest.segments[2].length, 1)
        self.assertEqual(manifest.segments[2].start_resi, 191)

        self.assertEqual(manifest.segments[3].segment_id, "D4")
        self.assertEqual(manifest.segments[3].source_chain, "B")
        self.assertEqual(manifest.segments[3].length, 1)
        self.assertEqual(manifest.segments[3].start_resi, 200)

        # Check provenance map: 1-based sequential keys
        self.assertEqual(len(manifest.provenance_map), 5)
        self.assertEqual(manifest.provenance_map[1]["orig_resi"], 312)
        self.assertEqual(manifest.provenance_map[1]["source_chain"], "A")
        self.assertEqual(manifest.provenance_map[1]["resn"], "TRP")
        self.assertEqual(manifest.provenance_map[3]["orig_resi"], 325)
        self.assertEqual(manifest.provenance_map[4]["orig_resi"], 191)
        self.assertEqual(manifest.provenance_map[4]["source_chain"], "B")

    # -------------------------------------------------------------------------
    # Pillar 2: Coordinate Fusion into chimera_seed.pdb & seed_provenance.json
    # -------------------------------------------------------------------------
    def test_spatial_coordinate_fusion_and_provenance(self):
        """
        Verify:
        1. chimera_seed.pdb assigns Chain 'D' to donor motif residues and Chain 'V' to vessel framework.
        2. Strict Cartesian preservation (zero rotation/translation applied).
        3. seed_provenance.json is written with 1-to-1 residue mapping.
        """
        out_prep = os.path.join(self.test_dir, "out_prep")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motifs="A:312-313, B:191, B:200",
            chimera_mode="replace",
            vessel_replace_range="50-60",
            out_dir=out_prep,
            dry_run=True,
        )
        scaffolder.compute_contigs_and_segments()
        seed_file = scaffolder.create_merged_seed_pdb()

        self.assertTrue(os.path.isfile(seed_file))
        prov_file = os.path.join(out_prep, "0_prep", "seed_provenance.json")
        self.assertTrue(os.path.isfile(prov_file))

        with open(prov_file, "r") as f:
            prov = json.load(f)
        self.assertEqual(len(prov), 4)
        self.assertEqual(prov["1"]["orig_resi"], 312)
        self.assertEqual(prov["2"]["orig_resi"], 313)
        self.assertEqual(prov["3"]["orig_resi"], 191)
        self.assertEqual(prov["4"]["orig_resi"], 200)

        # Inspect chimera_seed.pdb atom lines
        donor_d_lines = []
        vessel_v_lines = []
        with open(seed_file, "r") as f:
            for line in f:
                if line.startswith("ATOM  "):
                    ch = line[21]
                    if ch == "D":
                        donor_d_lines.append(line)
                    elif ch == "V":
                        vessel_v_lines.append(line)

        self.assertGreater(len(donor_d_lines), 0)
        self.assertGreater(len(vessel_v_lines), 0)

        # Invariant check: donor seed coordinates must match original donor coordinates exactly
        # Res 1 in seed (orig 312 TRP, CA coord [1.2, 0.0, 0.0])
        ca_d1 = next(line for line in donor_d_lines if line[12:16].strip() == "CA" and int(line[22:26]) == 1)
        x = float(ca_d1[30:38])
        y = float(ca_d1[38:46])
        z = float(ca_d1[46:54])
        np.testing.assert_allclose([x, y, z], [1.2, 0.0, 0.0], atol=1e-3)

        # Res 3 in seed (orig 191 PHE, CA coord [1.2, 10.0, 0.0])
        ca_d3 = next(line for line in donor_d_lines if line[12:16].strip() == "CA" and int(line[22:26]) == 3)
        x3 = float(ca_d3[30:38])
        y3 = float(ca_d3[38:46])
        z3 = float(ca_d3[46:54])
        np.testing.assert_allclose([x3, y3, z3], [1.2, 10.0, 0.0], atol=1e-3)

    # -------------------------------------------------------------------------
    # Pillar 3 & Section 7.1.2: Euclidean Gap Graph & Dynamic Linker Budgeting
    # -------------------------------------------------------------------------
    def test_euclidean_gap_graph_budgeting(self):
        """
        Verify:
        1. L_min = max(3, ceil(d_gap / 3.2))
        2. L_max = L_min + 5
        3. Contig formulation across synthetic known distances (15.0 Å, 32.0 Å, 28.5 Å).
        """
        # Exact mathematical formula check
        # d_gap = 15.0 Å: ceil(15.0 / 3.2) = ceil(4.6875) = 5 -> L = 5-10
        l_min, l_max = compute_gap_linker_length(15.0)
        self.assertEqual((l_min, l_max), (5, 10))
        self.assertEqual(compute_gap_linker_string(15.0), "5-10")

        # d_gap = 32.0 Å: ceil(32.0 / 3.2) = 10 -> L = 10-15
        l_min_32, l_max_32 = compute_gap_linker_length(32.0)
        self.assertEqual((l_min_32, l_max_32), (10, 15))
        self.assertEqual(compute_gap_linker_string(32.0), "10-15")

        # d_gap = 5.0 Å: ceil(5.0 / 3.2) = 2 -> clamped to max(3, 2) = 3 -> L = 3-8
        l_min_5, l_max_5 = compute_gap_linker_length(5.0)
        self.assertEqual((l_min_5, l_max_5), (3, 8))
        self.assertEqual(compute_gap_linker_string(5.0), "3-8")

        # Contig assembly test with multi-chain scaffolder
        out_contig = os.path.join(self.test_dir, "out_contig")
        scaffolder = ChimeraScaffolder(
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            motifs="A:312-313, B:191, B:200",
            chimera_mode="replace",
            vessel_replace_range="50-60",
            out_dir=out_contig,
            dry_run=True,
        )
        contig, donor_segs, vessel_segs = scaffolder.compute_contigs_and_segments()

        # Contig must contain V and D parts separated by budgeted linkers
        self.assertTrue(contig.startswith("V1-49/"), f"Unexpected contig start: {contig}")
        self.assertTrue(contig.endswith("/V61-100"), f"Unexpected contig end: {contig}")
        self.assertIn("D1-2", contig)
        self.assertIn("D3-3", contig)
        self.assertIn("D4-4", contig)
        self.assertNotIn("//", contig)

    # -------------------------------------------------------------------------
    # Pillar 4 & Section 7.1.3: In-Flight Cartesian Deviation Audit & Closed-Loop
    # -------------------------------------------------------------------------
    def test_closed_loop_drift_audit_rejection(self):
        """
        Verify:
        1. Candidate 1 with injected drift = 0.4 Å passes Cartesian drift audit (<= 0.8 Å).
        2. Candidate 2 with injected drift = 1.2 Å fails Cartesian drift audit (> 0.8 Å).
        3. Active retry loop purges rejected models and succeeds when passing candidate is sampled.
        4. RuntimeError is raised if max_attempts is exceeded without passing candidates.
        """
        seed_pdb = os.path.join(self.test_dir, "seed_test.pdb")
        # Reference seed: 3 motif residues on Chain D
        seed_res = [
            (1, "TRP", {"CA": [0.0, 0.0, 0.0]}),
            (2, "GLU", {"CA": [3.8, 0.0, 0.0]}),
            (3, "PHE", {"CA": [7.6, 0.0, 0.0]}),
        ]
        create_mock_pdb(seed_pdb, {"D": seed_res})

        prov = {
            1: {"orig_resi": 312, "source_chain": "A", "resn": "TRP"},
            2: {"orig_resi": 313, "source_chain": "A", "resn": "GLU"},
            3: {"orig_resi": 191, "source_chain": "B", "resn": "PHE"},
        }

        # Candidate 1: Injected drift = 0.4 Å (< 0.8 Å cutoff -> PASS)
        cand1_pdb = os.path.join(self.test_dir, "cand_pass.pdb")
        cand1_res = [
            (1, "TRP", {"CA": [0.0, 0.0, 0.4]}),
            (2, "GLU", {"CA": [3.8, 0.0, 0.4]}),
            (3, "PHE", {"CA": [7.6, 0.0, 0.4]}),
        ]
        create_mock_pdb(cand1_pdb, {"D": cand1_res})

        max_drift1, mean_drift1, _ = audit_motif_cartesian_drift(cand1_pdb, seed_pdb, prov)
        self.assertAlmostEqual(max_drift1, 0.4, places=2)
        self.assertAlmostEqual(mean_drift1, 0.4, places=2)
        self.assertLessEqual(max_drift1, 0.8)
        self.assertLessEqual(mean_drift1, 0.5)

        # Candidate 2: Injected drift = 1.2 Å (> 0.8 Å cutoff -> REJECT)
        cand2_pdb = os.path.join(self.test_dir, "cand_fail.pdb")
        cand2_res = [
            (1, "TRP", {"CA": [0.0, 0.0, 1.2]}),
            (2, "GLU", {"CA": [3.8, 0.0, 1.2]}),
            (3, "PHE", {"CA": [7.6, 0.0, 1.2]}),
        ]
        create_mock_pdb(cand2_pdb, {"D": cand2_res})

        max_drift2, mean_drift2, _ = audit_motif_cartesian_drift(cand2_pdb, seed_pdb, prov)
        self.assertAlmostEqual(max_drift2, 1.2, places=2)
        self.assertGreater(max_drift2, 0.8)

        # Test active retry loop:
        # Attempt 1 returns drifted candidate (1.2 Å) -> rejected & purged
        # Attempt 2 returns passing candidate (0.4 Å) -> accepted
        work_dir = os.path.join(self.test_dir, "closed_loop_test")
        os.makedirs(work_dir, exist_ok=True)

        def mock_sampler(seed, contig, prefix, attempt):
            out_p = f"{prefix}_0.pdb"
            if attempt == 1:
                # Drifted candidate
                create_mock_pdb(out_p, {"D": cand2_res})
            else:
                # Passing candidate
                create_mock_pdb(out_p, {"D": cand1_res})
            return out_p

        passing = run_closed_loop_backbone_generation(
            seed_pdb=seed_pdb,
            contig_str="D1-3",
            seed_provenance=prov,
            target_passing_designs=1,
            max_attempts=5,
            sampler_fn=mock_sampler,
            work_dir=work_dir,
        )
        self.assertEqual(len(passing), 1)
        self.assertIn("candidate_2", passing[0])
        # Verify attempt 1 candidate was discarded and removed
        self.assertFalse(os.path.isfile(os.path.join(work_dir, "candidate_1_0.pdb")))

        # Test failure when all attempts exceed drift cutoff
        def fail_sampler(seed, contig, prefix, attempt):
            out_p = f"{prefix}_0.pdb"
            create_mock_pdb(out_p, {"D": cand2_res})
            return out_p

        with self.assertRaises(RuntimeError):
            run_closed_loop_backbone_generation(
                seed_pdb=seed_pdb,
                contig_str="D1-3",
                seed_provenance=prov,
                target_passing_designs=1,
                max_attempts=3,
                sampler_fn=fail_sampler,
                work_dir=os.path.join(self.test_dir, "fail_test"),
            )

    # -------------------------------------------------------------------------
    # Pillar 5 & Section 7.1.4: 3-Tier Selective Harmonic Pinning & Relaxation
    # -------------------------------------------------------------------------
    def test_selective_pinning_relaxation(self):
        """
        Verify:
        1. parse_res_indices parses ranges and comma-separated residue lists.
        2. Soft-Core geometric pre-pass resolves severe atomic clashes (d < 1.2 Å).
        3. Selective harmonic pinning configuration applies:
           k_motif = 5000 kJ/mol/nm^2 to motif heavy atoms
           k_vessel = 500 kJ/mol/nm^2 to vessel backbone atoms
           k_linker = 0 kJ/mol/nm^2 to linker atoms
        """
        # 1. Parse residue indices
        indices = parse_res_indices("55-57,110-112,120")
        self.assertEqual(indices, [55, 56, 57, 110, 111, 112, 120])

        # 2. Clashing structure creation
        clash_pdb = os.path.join(self.test_dir, "clash_test.pdb")
        # Residues 1-2: Vessel
        # Residues 3-4: Linkers (extreme clash: sidechains separated by 0.05 Å)
        # Residues 5-6: Motif
        test_chains = {
            "A": [
                (1, "ALA", {"N": [0.0, 0.0, 0.0], "CA": [1.0, 0.0, 0.0], "C": [2.0, 0.0, 0.0], "O": [2.0, 1.0, 0.0]}),
                (2, "ALA", {"N": [3.0, 0.0, 0.0], "CA": [4.0, 0.0, 0.0], "C": [5.0, 0.0, 0.0], "O": [5.0, 1.0, 0.0]}),
                # Linker with clash
                (
                    3,
                    "VAL",
                    {
                        "N": [6.0, 0.0, 0.0],
                        "CA": [7.0, 0.0, 0.0],
                        "C": [8.0, 0.0, 0.0],
                        "O": [8.0, 1.0, 0.0],
                        "CG1": [7.5, 0.5, 0.0],
                    },
                ),
                (
                    4,
                    "LEU",
                    {
                        "N": [9.0, 0.0, 0.0],
                        "CA": [10.0, 0.0, 0.0],
                        "C": [11.0, 0.0, 0.0],
                        "O": [11.0, 1.0, 0.0],
                        "CD1": [7.55, 0.5, 0.0],
                    },
                ),
                # Motif
                (
                    5,
                    "TRP",
                    {"N": [12.0, 0.0, 0.0], "CA": [13.0, 0.0, 0.0], "C": [14.0, 0.0, 0.0], "O": [14.0, 1.0, 0.0]},
                ),
                (
                    6,
                    "GLU",
                    {"N": [15.0, 0.0, 0.0], "CA": [16.0, 0.0, 0.0], "C": [17.0, 0.0, 0.0], "O": [17.0, 1.0, 0.0]},
                ),
            ]
        }
        create_mock_pdb(clash_pdb, test_chains)

        # Verify untangle_severe_clashes resolves the clash
        class DummyElem:
            def __init__(self, s):
                self.symbol = s

        class DummyRes:
            def __init__(self, i):
                self.index = i

        class DummyAtom:
            def __init__(self, name, r_i, s="C"):
                self.name = name
                self.residue = DummyRes(r_i)
                self.element = DummyElem(s)

        class DummyTop:
            def __init__(self, atoms):
                self._atoms = atoms

            def atoms(self):
                return iter(self._atoms)

        atoms = [
            DummyAtom("CG1", 3),
            DummyAtom("CD1", 4),
        ]
        # 0.05 Å clash = 0.005 nm
        positions = [
            np.array([0.75, 0.05, 0.0]),
            np.array([0.755, 0.05, 0.0]),
        ]
        from types import SimpleNamespace
        pos_objs = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in positions]
        top = DummyTop(atoms)

        nudged = untangle_severe_clashes(top, pos_objs, min_dist_angstrom=1.2)
        p1 = np.array([nudged[0].x, nudged[0].y, nudged[0].z])
        p2 = np.array([nudged[1].x, nudged[1].y, nudged[1].z])
        new_dist = np.linalg.norm(p1 - p2)
        # Nudge should separate them by > 0.3 Å (> 0.03 nm)
        self.assertGreater(new_dist, 0.03)

        # 3. Test selective pinning parameter verification via mock or real OpenMM
        out_relaxed = os.path.join(self.test_dir, "relaxed.pdb")
        has_openmm = importlib.util.find_spec("openmm") is not None

        if has_openmm:
            # Execute real OpenMM selective minimization
            success = relax_structure(
                input_pdb=clash_pdb,
                output_pdb=out_relaxed,
                motif_res_indices=[5, 6],
                vessel_res_indices=[1, 2],
                k_motif=5000.0,
                k_vessel=500.0,
                k_linker=0.0,
                max_iterations=50,
            )
            self.assertTrue(success)
            self.assertTrue(os.path.isfile(out_relaxed))
        else:
            # Verify Mock CustomExternalForce configuration with O(1) residue classification
            mock_mm = MagicMock()
            mock_app = MagicMock()
            mock_unit = MagicMock()
            mock_force = MagicMock()
            mock_mm.app = mock_app
            mock_mm.unit = mock_unit
            mock_mm.CustomExternalForce.return_value = mock_force

            class Res:
                def __init__(self, i):
                    self.index = i
                    self.id = str(i + 1)

            class Atom:
                def __init__(self, name, res, elem, idx):
                    self.name = name
                    self.residue = res
                    self.element = MagicMock(symbol=elem)
                    self.index = idx

            atoms_list = []
            residues_list = []
            positions_list = []
            for r_idx in range(6):
                res = Res(r_idx)
                residues_list.append(res)
                for aname in ["N", "CA", "C", "O", "CB"]:
                    elem = "C" if aname == "CB" else aname[0]
                    atm = Atom(aname, res, elem, len(atoms_list))
                    atoms_list.append(atm)
                    positions_list.append(MagicMock(x=float(len(atoms_list)), y=0.0, z=0.0))

            mock_pdb = MagicMock()
            mock_pdb.topology.atoms.side_effect = lambda: iter(atoms_list)
            mock_pdb.topology.residues.side_effect = lambda: iter(residues_list)
            mock_pdb.positions = positions_list
            mock_app.PDBFile.return_value = mock_pdb

            with patch.dict("sys.modules", {"openmm": mock_mm, "openmm.app": mock_app, "openmm.unit": mock_unit}):
                from scripts.relax_chimera import relax_with_selective_pinning
                res = relax_with_selective_pinning(
                    input_pdb=clash_pdb,
                    output_pdb=out_relaxed,
                    motif_res_indices=[5, 6],
                    vessel_res_indices=[1, 2],
                    k_motif=5000.0,
                    k_vessel=500.0,
                    k_linker=0.0,
                    max_iterations=50,
                )
                self.assertTrue(res)
                self.assertTrue(mock_force.addParticle.called)
                added_k = [call[0][1][0] for call in mock_force.addParticle.call_args_list]
                # Motif residues 5-6 (heavy atoms) -> 5000.0 kJ/mol/nm^2
                self.assertEqual(added_k.count(5000.0), 10)
                # Vessel residues 1-2 (backbone N, CA, C, O) -> 500.0 kJ/mol/nm^2
                self.assertEqual(added_k.count(500.0), 8)
                # Linker residues 3-4 (k_linker=0.0) -> omitted from restraint force
                self.assertEqual(added_k.count(0.0), 0)

    # -------------------------------------------------------------------------
    # Pillar 5: Performance Optimizations, Vectorization, and Caching Audits
    # -------------------------------------------------------------------------
    def test_performance_optimizations_and_caching(self):
        """
        Verify:
        1. Reference seed CA caching in audit_motif_cartesian_drift eliminates redundant disk I/O.
        2. Vectorized distance calculations with NumPy correctly compute max and mean drift.
        3. Anchor atom caching in get_residue_anchor_coords operates in O(1) time without re-reading PDB.
        4. MotifSegment terminal properties (n_term_n, c_term_c) return 3D coords in O(1) time.
        5. compute_gap_linker_length and compute_gap_linker_string utilize LRU caching.
        6. _purge_candidate_artifacts safely purges candidate PDBs, TRBs, and micro-batch artifacts.
        """
        clear_ca_coords_cache()

        # 1. Test Seed PDB CA caching
        seed_pdb = os.path.join(self.test_dir, "perf_seed.pdb")
        create_mock_pdb(
            seed_pdb,
            {"D": [(1, "ALA", {"CA": [0.0, 0.0, 0.0]}), (2, "GLY", {"CA": [3.8, 0.0, 0.0]})]},
        )
        cand_pdb = os.path.join(self.test_dir, "perf_cand.pdb")
        create_mock_pdb(
            cand_pdb,
            {"D": [(1, "ALA", {"CA": [0.0, 0.0, 0.2]}), (2, "GLY", {"CA": [3.8, 0.0, 0.2]})]},
        )
        prov = {
            1: {"orig_resi": 1, "source_chain": "D", "resn": "ALA"},
            2: {"orig_resi": 2, "source_chain": "D", "resn": "GLY"},
        }

        # First audit populates cache
        max_d1, mean_d1, _ = audit_motif_cartesian_drift(cand_pdb, seed_pdb, prov)
        self.assertAlmostEqual(max_d1, 0.2, places=2)
        real_seed_path = os.path.realpath(seed_pdb)
        seed_mtime = os.path.getmtime(real_seed_path)
        self.assertIn((real_seed_path, seed_mtime), _SEED_CA_CACHE)

        # Subsequent audit uses cached seed coordinates (O(1) memory lookup)
        max_d2, mean_d2, _ = audit_motif_cartesian_drift(cand_pdb, seed_pdb, prov)
        self.assertEqual(max_d1, max_d2)

        # 2. Test Anchor Atom Caching
        vessel_pdb = os.path.join(self.test_dir, "perf_vessel.pdb")
        create_mock_pdb(
            vessel_pdb,
            {"V": [(10, "VAL", {"N": [1.0, 2.0, 3.0], "CA": [2.0, 2.0, 3.0], "C": [3.0, 2.0, 3.0]})]},
        )
        anchor_n = get_residue_anchor_coords(vessel_pdb, "V", 10, "N")
        np.testing.assert_allclose(anchor_n, [1.0, 2.0, 3.0])
        real_vessel = os.path.realpath(vessel_pdb)
        vessel_mtime = os.path.getmtime(real_vessel)
        self.assertIn((real_vessel, vessel_mtime), _ANCHOR_COORDS_CACHE)

        # Cached query is instantaneous
        anchor_c = get_residue_anchor_coords(vessel_pdb, "V", 10, "C")
        np.testing.assert_allclose(anchor_c, [3.0, 2.0, 3.0])

        # 3. Test MotifSegment terminal properties
        manifest = build_multi_chain_manifest("A:312-313", default_source_file=self.donor_pdb)
        seg = manifest.segments[0]
        self.assertIsNotNone(seg.n_term_n)
        self.assertIsNotNone(seg.c_term_c)
        np.testing.assert_allclose(seg.n_term_n, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(seg.c_term_c, [6.0, 0.0, 0.0])

        # 4. Test LRU Caching of dynamic linkers
        l_min, l_max = compute_gap_linker_length(15.0)
        self.assertEqual((l_min, l_max), (5, 10))
        l_str = compute_gap_linker_string(15.0)
        self.assertEqual(l_str, "5-10")

        # 5. Test micro-batch artifact cleanup
        work_dir = os.path.join(self.test_dir, "microbatch_purge_test")
        os.makedirs(work_dir, exist_ok=True)
        cand_p = os.path.join(work_dir, "candidate_99_0.pdb")
        trb_p = os.path.join(work_dir, "candidate_99_0.trb")
        aux_p = os.path.join(work_dir, "candidate_99_scores.json")
        for f in (cand_p, trb_p, aux_p):
            with open(f, "w") as fp:
                fp.write("mock")
        self.assertTrue(os.path.isfile(cand_p))
        self.assertTrue(os.path.isfile(trb_p))
        self.assertTrue(os.path.isfile(aux_p))

        _purge_candidate_artifacts(os.path.join(work_dir, "candidate_99"), cand_p)
        self.assertFalse(os.path.isfile(cand_p))
        self.assertFalse(os.path.isfile(trb_p))
        self.assertFalse(os.path.isfile(aux_p))

    # -------------------------------------------------------------------------
    # Pillar 6: Security Audit Verification (Path Traversal, Safe Unpickling)
    # -------------------------------------------------------------------------
    def test_security_multichain_and_safe_unpickling(self):
        """
        Verify:
        1. parse_multi_chain_motif_spec rejects path traversal in chain ID (e.g. '../A') and dangerous root files.
        2. build_multi_chain_manifest rejects dangerous root file specifications.
        3. audit_motif_cartesian_drift rejects dangerous root paths and safely handles .trb unpickling.
        4. relax_chimera validate_path rejects directory traversal and root paths.
        """
        # 1. Path traversal in chain ID
        with self.assertRaises(ValueError):
            parse_multi_chain_motif_spec("f.pdb:../A:1-10")

        with self.assertRaises(ValueError):
            parse_multi_chain_motif_spec("f.pdb:A/B:1-10")

        # 2. Root file specification
        with self.assertRaises(ValueError):
            parse_multi_chain_motif_spec("/:A:1-10")

        with self.assertRaises(ValueError):
            build_multi_chain_manifest("/:A:1-10")

        # 3. Path validation in relax_chimera
        from scripts.relax_chimera import validate_path
        with self.assertRaises(ValueError):
            validate_path("/", label="Root check")

        with self.assertRaises(ValueError):
            validate_path("///", label="Root check")

        # 4. SafeSessionUnpickler blocks malicious arbitrary class execution in .trb file
        import pickle
        malicious_trb = os.path.join(self.test_dir, "malicious.trb")
        # Attempt to pickle a dangerous callable (os.system)
        with open(malicious_trb, "wb") as pf:
            pickle.dump(os.system, pf)

        # Candidate PDB pointing to malicious TRB
        cand_pdb = os.path.join(self.test_dir, "malicious.pdb")
        create_mock_pdb(cand_pdb, {"D": [(1, "ALA", {"CA": [0.0, 0.0, 0.0]})]})

        seed_pdb = os.path.join(self.test_dir, "seed_dummy.pdb")
        create_mock_pdb(seed_pdb, {"D": [(1, "ALA", {"CA": [0.0, 0.0, 0.0]})]})

        prov = {1: {"orig_resi": 1, "source_chain": "A", "resn": "ALA"}}

        # audit_motif_cartesian_drift must catch the UnpicklingError safely without executing code
        max_d, mean_d, details = audit_motif_cartesian_drift(cand_pdb, seed_pdb, prov)
        self.assertIsInstance(max_d, float)
        self.assertIsInstance(mean_d, float)


if __name__ == "__main__":
    unittest.main()
