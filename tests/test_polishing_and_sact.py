#!/usr/bin/env python3
"""
Unit tests for Sequence-Anchored Contig Tracking (SACT) and 2-Stage Soft-Core Polishing.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from scripts.chimera_scaffold import thread_sequence_onto_pdb
from scripts.relax_chimera import (
    untangle_severe_clashes,
    find_default_input_pdb,
)


class TestSACTAndPolishing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sact_preserves_donor_residues_ew(self):
        """
        Verify that Sequence-Anchored Contig Tracking (SACT) strictly enforces
        the authentic native donor identity (e.g. W312 and E313), overriding
        any mutated sequence produced by ProteinMPNN.
        """
        # Create a mock 4-residue backbone: Res 1, 2 = Linker (designed), Res 3, 4 = Donor (W, E)
        bb_pdb = os.path.join(self.test_dir, "backbone.pdb")
        with open(bb_pdb, "w") as f:
            f.write(
                "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N\n"
                "ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM      3  C   ALA A   1      12.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM      4  O   ALA A   1      12.500  11.000  10.000  1.00 20.00           O\n"
                "ATOM      5  N   ALA A   2      13.000  10.000  10.000  1.00 20.00           N\n"
                "ATOM      6  CA  ALA A   2      14.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM      7  C   ALA A   2      15.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM      8  O   ALA A   2      15.500  11.000  10.000  1.00 20.00           O\n"
                # Donor Res 3 (TRP)
                "ATOM      9  N   TRP A   3      16.000  10.000  10.000  1.00 20.00           N\n"
                "ATOM     10  CA  TRP A   3      17.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM     11  C   TRP A   3      18.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM     12  O   TRP A   3      18.500  11.000  10.000  1.00 20.00           O\n"
                # Donor Res 4 (GLU)
                "ATOM     13  N   GLU A   4      19.000  10.000  10.000  1.00 20.00           N\n"
                "ATOM     14  CA  GLU A   4      20.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM     15  C   GLU A   4      21.000  10.000  10.000  1.00 20.00           C\n"
                "ATOM     16  O   GLU A   4      21.500  11.000  10.000  1.00 20.00           O\n"
            )

        # Seed PDB has authentic donor TRP and GLU with full sidechains
        seed_pdb = os.path.join(self.test_dir, "seed.pdb")
        with open(seed_pdb, "w") as f:
            f.write(
                "ATOM      1  N   TRP A 312      16.100  10.050  10.000  1.00 20.00           N\n"
                "ATOM      2  CA  TRP A 312      17.100  10.050  10.000  1.00 20.00           C\n"
                "ATOM      3  C   TRP A 312      18.100  10.050  10.000  1.00 20.00           C\n"
                "ATOM      4  O   TRP A 312      18.600  11.050  10.000  1.00 20.00           O\n"
                "ATOM      5  CB  TRP A 312      17.200   9.000  10.500  1.00 20.00           C\n"
                "ATOM      6  CG  TRP A 312      17.500   8.000  11.000  1.00 20.00           C\n"
                "ATOM      7  N   GLU A 313      19.100  10.050  10.000  1.00 20.00           N\n"
                "ATOM      8  CA  GLU A 313      20.100  10.050  10.000  1.00 20.00           C\n"
                "ATOM      9  C   GLU A 313      21.100  10.050  10.000  1.00 20.00           C\n"
                "ATOM     10  O   GLU A 313      21.600  11.050  10.000  1.00 20.00           O\n"
                "ATOM     11  CB  GLU A 313      20.200   9.000  10.500  1.00 20.00           C\n"
            )

        out_pdb = os.path.join(self.test_dir, "threaded.pdb")
        # Suppose ProteinMPNN redesigned the entire sequence to "AAAA" (mutating W->A and E->A)
        designed_seq = "AAAA"

        # Thread with SACT locked donor positions = [3, 4]
        success = thread_sequence_onto_pdb(
            bb_pdb,
            designed_seq,
            out_pdb,
            seed_pdb=seed_pdb,
            fixed_donor_positions=[3, 4],
        )
        self.assertTrue(success)

        # Inspect resulting PDB: Res 1 & 2 should be ALA, but Res 3 MUST be TRP and Res 4 MUST be GLU!
        with open(out_pdb, "r") as f:
            pdb_lines = [line for line in f if line.startswith("ATOM  ")]

        res_names = {}
        for atom_line in pdb_lines:
            resi = int(atom_line[22:26])
            resn = atom_line[17:20].strip()
            res_names[resi] = resn

        self.assertEqual(res_names.get(1), "ALA")
        self.assertEqual(res_names.get(2), "ALA")
        self.assertEqual(res_names.get(3), "TRP", "SACT must strictly preserve authentic donor TRP (W)")
        self.assertEqual(res_names.get(4), "GLU", "SACT must strictly preserve authentic donor GLU (E)")

    def test_untangle_severe_clashes(self):
        """
        Verify that soft-core geometric pre-pass detects clashing sidechains (d < 1.2 Å)
        and gently nudges them apart without altering backbone coordinates.
        """
        class DummyElement:
            def __init__(self, sym):
                self.symbol = sym

        class DummyResidue:
            def __init__(self, idx):
                self.index = idx

        class DummyAtom:
            def __init__(self, name, res_idx, sym="C"):
                self.name = name
                self.residue = DummyResidue(res_idx)
                self.element = DummyElement(sym)

        class DummyPos:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class DummyTopology:
            def __init__(self, atoms):
                self._atoms = atoms

            def atoms(self):
                return iter(self._atoms)

        # Res 1: CA at (0,0,0), CB (sidechain) at (0.1, 0, 0)
        # Res 2: CA at (0.5, 0, 0), CD (sidechain) at (0.105, 0, 0) -> overlapping at 0.05 Å!
        atoms = [
            DummyAtom("CA", 0),
            DummyAtom("CB", 0),
            DummyAtom("CA", 1),
            DummyAtom("CD", 1),
        ]
        positions = [
            DummyPos(0.0, 0.0, 0.0),
            DummyPos(0.1, 0.0, 0.0),
            DummyPos(0.5, 0.0, 0.0),
            DummyPos(0.105, 0.0, 0.0),  # Extreme clash: 0.005 nm = 0.05 Å!
        ]
        top = DummyTopology(atoms)

        def get_val(v):
            return getattr(v, "value_in_unit_system", v)

        nudged_positions = untangle_severe_clashes(top, positions, min_dist_angstrom=1.2)
        # Backbone CA atoms must not move
        self.assertAlmostEqual(get_val(nudged_positions[0].x), 0.0, delta=1e-5)
        self.assertAlmostEqual(get_val(nudged_positions[2].x), 0.5, delta=1e-5)

        # Clashing sidechain atoms CB and CD must be nudged apart
        p1 = np.array([get_val(nudged_positions[1].x), get_val(nudged_positions[1].y), get_val(nudged_positions[1].z)])
        p2 = np.array([get_val(nudged_positions[3].x), get_val(nudged_positions[3].y), get_val(nudged_positions[3].z)])
        new_dist = np.linalg.norm(p1 - p2)
        self.assertGreater(new_dist, 0.03, "Clashing sidechain atoms must be separated")

    def test_find_default_input_pdb_discovers_cterm(self):
        """
        Verify that find_default_input_pdb discovers PDB files located inside c_term/ or results/.
        """
        cterm_dir = os.path.join(self.test_dir, "c_term", "results")
        os.makedirs(cterm_dir, exist_ok=True)
        target_pdb = os.path.join(cterm_dir, "chimera_full_atoms.pdb")
        with open(target_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.0  0.0  0.0  1.00 20.00           C\n")

        cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            found = find_default_input_pdb()
            self.assertTrue(found.endswith(".pdb"))
            self.assertIn("c_term", found)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
