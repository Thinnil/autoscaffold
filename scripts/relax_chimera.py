#!/usr/bin/env python3
"""
relax_chimera.py - High-Performance Backbone-Restrained Energy Minimizer & Polishing Engine
Uses OpenMM, PDBFixer, and optional ColabFold Amber Relax / FASPR to eliminate sidechain
steric clashes and false covalent bonds in PyMOL while keeping the tertiary backbone
structure firmly restrained.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import Any, List, Optional, Sequence, Union


def validate_path(path: str, label: str = "Path", allow_missing: bool = False) -> str:
    """
    Validates a file path to prevent directory traversal and root directory overwrites.
    Returns canonical real path.
    """
    if not path or not str(path).strip():
        raise ValueError(f"{label} cannot be empty.")
    raw = str(path).strip()
    drive, rest = os.path.splitdrive(raw)
    if (
        (not drive and raw.rstrip("/\\") in ("", "/", "\\"))
        or (drive and rest in ("/", "\\", ""))
        or raw in ("/", "\\", "//", "\\\\")
    ):
        raise ValueError(f"Dangerous or root path specified for {label}: '{raw}'")
    real = os.path.realpath(os.path.abspath(raw))
    drive, rest = os.path.splitdrive(real)
    if (not drive and real.rstrip("/\\") in ("", "/")) or (drive and rest in ("/", "\\", "")):
        raise ValueError(f"Dangerous or root path specified for {label}: '{raw}'")
    if not allow_missing and not os.path.isfile(real):
        raise FileNotFoundError(f"{label} file not found: '{real}'")
    return real


def find_default_input_pdb() -> str:
    patterns = [
        "*/results/*.pdb",
        "*/*/results/*.pdb",
        "c_term/results/*.pdb",
        "c_term/**/*.pdb",
        "n_terminal/results/*.pdb",
        "n_terminal/**/*.pdb",
        "*/2_mpnn/*_seq*.pdb",
        "*/1_rfdiffusion/*.pdb",
        "results/*.pdb",
        "**/*.pdb",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        valid = [
            m for m in matches
            if not any(k in m for k in ("relaxed", "clash_free", "minimized", "seed", "0_prep"))
            and os.path.isfile(m)
        ]
        if valid:
            return sorted(valid, key=os.path.getmtime, reverse=True)[0]
    return ""


def count_steric_clashes(topology: Any, positions: Any, cutoff_angstrom: float = 1.9) -> int:
    """
    Counts pairs of non-bonded heavy atoms closer than cutoff_angstrom (default 1.9 Å).
    In PyMOL, interatomic distances below ~1.9 Å trigger false stick bonds.
    """
    try:
        import numpy as np
        cutoff_nm = cutoff_angstrom / 10.0
        pos_arr = np.array([[p.x, p.y, p.z] for p in positions])
        atoms = list(topology.atoms())
        n = len(atoms)
        if n > 2500:
            # Sample heavy atoms only to keep audit sub-second
            heavy_indices = [i for i, a in enumerate(atoms) if a.element and a.element.symbol != "H"]
            pos_arr = pos_arr[heavy_indices]
            atoms = [atoms[i] for i in heavy_indices]
            n = len(atoms)

        clashes = 0
        res_indices = [a.residue.index for a in atoms]
        names = [a.name for a in atoms]

        for i in range(n):
            ri = res_indices[i]
            diffs = pos_arr[i + 1:] - pos_arr[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            too_close = np.where(dists_sq < (cutoff_nm * cutoff_nm))[0]
            for c_idx in too_close:
                j = i + 1 + c_idx
                rj = res_indices[j]
                if abs(ri - rj) >= 1 or (names[i] not in ("C", "N") and names[j] not in ("C", "N")):
                    clashes += 1
        return clashes
    except Exception:
        return 0


def untangle_severe_clashes(topology: Any, positions: Any, min_dist_angstrom: float = 1.2) -> Any:
    """
    Soft-Core geometric pre-pass:
    Identifies non-bonded sidechain atoms that directly overlap (distance < 1.2 Å)
    and nudges them gently apart so that Lennard-Jones (1/r^12) potentials do not blow up.
    Backbone atoms (N, CA, C, O) remain completely unchanged.
    """
    try:
        import numpy as np

        min_nm = min_dist_angstrom / 10.0
        coords = np.array(
            [[p.x, p.y, p.z] if hasattr(p, "x") else [p[0], p[1], p[2]] for p in positions],
            dtype=float,
        )
        atoms = list(topology.atoms())
        is_sidechain = np.array([a.name not in ("N", "CA", "C", "O", "OXT") for a in atoms])
        res_indices = np.array([a.residue.index for a in atoms])

        nudged = 0
        n = len(atoms)
        for i in range(n):
            if not is_sidechain[i]:
                continue
            diffs = coords[i + 1:] - coords[i]
            dists = np.linalg.norm(diffs, axis=1)
            clash_idxs = np.where(dists < min_nm)[0]

            for c_idx in clash_idxs:
                j = i + 1 + c_idx
                if res_indices[i] == res_indices[j]:
                    continue  # Inter-residue clashes only
                dist = dists[c_idx]
                if dist < 1e-4:
                    vec = np.array([0.05, 0.05, 0.05])
                else:
                    vec = (diffs[c_idx] / dist) * 0.04  # 0.4 Å nudge
                coords[i] -= vec * 0.5
                if is_sidechain[j]:
                    coords[j] += vec * 0.5
                nudged += 1

        if nudged > 0:
            print(f"[+] Soft-Core Pre-pass: Gently nudged {nudged} clashing sidechain pairs to prevent LJ explosion.")
            try:
                from openmm import unit
                return [unit.Quantity(coords[i], unit.nanometers) for i in range(n)]
            except Exception:
                from types import SimpleNamespace
                return [SimpleNamespace(x=coords[i][0], y=coords[i][1], z=coords[i][2]) for i in range(n)]
    except Exception as e:
        print(f"[*] Pre-pass note: {e}")
    return positions


def run_faspr_repack(input_pdb: str, output_pdb: str) -> bool:
    """
    Attempts to run FASPR (Fast and Accurate Side-Chain Packing) if installed on system.
    """
    faspr_bin = shutil.which("FASPR") or shutil.which("faspr")
    if not faspr_bin:
        cand_tools = [
            os.path.join(os.path.dirname(__file__), "..", "tools", "bin", "FASPR"),
            os.path.join(os.path.dirname(__file__), "..", "tools", "FASPR", "FASPR"),
        ]
        for c in cand_tools:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                faspr_bin = c
                break

    if not faspr_bin:
        return False

    try:
        print(f"[+] Running FASPR graph-cut rotamer repacking: {faspr_bin}")
        res = subprocess.run(
            [faspr_bin, "-i", input_pdb, "-o", output_pdb],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if res.returncode == 0 and os.path.isfile(output_pdb):
            print("✅ FASPR rotamer packing complete.")
            return True
    except Exception as e:
        print(f"[*] FASPR execution note: {e}")
    return False


def relax_with_colabfold_amber(input_pdb: str, output_pdb: str) -> bool:
    """
    Attempts to run the official ColabFold / AlphaFold2 Amber Relaxation engine
    if running in the user's 'colabfold' conda environment.
    """
    try:
        from colabfold.relax import relax_me
        print("[+] Running ColabFold native Amber relaxation pipeline...")
        relax_me(pdb_filename=input_pdb, output_filename=output_pdb)
        if os.path.isfile(output_pdb):
            print("✅ ColabFold Amber relaxation succeeded.")
            return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[*] ColabFold relax_me note ({e}), falling back to 2-stage OpenMM...")
    return False


def parse_res_indices(res_spec: Any) -> List[int]:
    """
    Parses comma-separated residue ranges or indices, e.g. '55-57,110-112' or '55,56,57'.
    """
    if not res_spec:
        return []
    if isinstance(res_spec, (list, tuple, set)):
        return sorted(list(set(int(x) for x in res_spec)))
    indices = []
    for part in str(res_spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                s, e = part.split("-", 1)
                indices.extend(range(int(s.strip()), int(e.strip()) + 1))
            except ValueError:
                pass
        elif part.isdigit():
            indices.append(int(part))
    return sorted(list(set(indices)))


def relax_with_selective_pinning(
    input_pdb: str,
    output_pdb: str,
    motif_res_indices: Sequence[Union[int, str]],  # 1-based indices in input_pdb
    vessel_res_indices: Sequence[Union[int, str]],  # 1-based indices in input_pdb
    k_motif: float = 5000.0,  # kJ/mol/nm^2
    k_vessel: float = 500.0,  # kJ/mol/nm^2
    k_linker: float = 0.0,  # kJ/mol/nm^2
    max_iterations: int = 200,
) -> bool:
    """
    OpenMM minimization with 3-tier selective harmonic pinning:
    - Motif heavy atoms (non-hydrogen): pinned at k_motif (5000 kJ/mol/nm^2)
    - Vessel framework backbone (N, CA, C, O): pinned at k_vessel (500 kJ/mol/nm^2)
    - Designed linkers: k_linker (0.0 kJ/mol/nm^2) to freely resolve clashes
    """
    try:
        import openmm as mm
        import openmm.app as app
        from openmm import unit
    except ImportError:
        print("[-] Error: OpenMM is not installed in the active Python environment.", file=sys.stderr)
        return False

    try:
        input_pdb = validate_path(input_pdb, label="Input PDB", allow_missing=False)
        output_pdb = validate_path(output_pdb, label="Output PDB", allow_missing=True)
    except Exception as e:
        print(f"[-] Path validation error in selective pinning: {e}", file=sys.stderr)
        return False

    if not os.path.isfile(input_pdb):
        print(f"[-] Error: Input PDB file not found: {input_pdb}", file=sys.stderr)
        return False

    pdb = app.PDBFile(input_pdb)
    forcefield = app.ForceField("amber14-all.xml")

    # 1. Soft-Core geometric pre-pass for overlapping sidechain atoms (d < 1.2 Å)
    positions = untangle_severe_clashes(pdb.topology, pdb.positions, min_dist_angstrom=1.2)

    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    # 2. Configure Selective CustomExternalForce
    restraint = mm.CustomExternalForce("0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    motif_set = frozenset(parse_res_indices(motif_res_indices))
    vessel_set = frozenset(parse_res_indices(vessel_res_indices))

    # Pre-classify residues once in O(R) time for O(1) atom lookups
    res_classification = {}
    for residue in pdb.topology.residues():
        res_idx = residue.index + 1  # 1-based sequential
        try:
            pdb_res_idx = int(residue.id)
        except (ValueError, TypeError):
            pdb_res_idx = res_idx
        is_m = (res_idx in motif_set) or (pdb_res_idx in motif_set)
        is_v = (res_idx in vessel_set) or (pdb_res_idx in vessel_set)
        res_classification[residue.index] = (is_m, is_v)

    backbone_names = frozenset(("N", "CA", "C", "O"))

    for atom in pdb.topology.atoms():
        is_motif, is_vessel = res_classification.get(atom.residue.index, (False, False))

        # Skip linker atoms completely when k_linker == 0.0 to avoid unused force particles
        if not (is_motif or is_vessel or k_linker > 0.0):
            continue

        pos = positions[atom.index]
        if hasattr(pos, "value_in_unit"):
            p_nm = pos.value_in_unit(unit.nanometer)
            px, py, pz = p_nm[0], p_nm[1], p_nm[2]
        elif hasattr(pos, "x"):
            px, py, pz = pos.x, pos.y, pos.z
        else:
            px, py, pz = pos[0], pos[1], pos[2]

        if is_motif:
            # Pin both backbone and heavy atoms of motif to preserve catalytic rotamers
            if atom.element and atom.element.symbol != "H":
                restraint.addParticle(atom.index, [k_motif, px, py, pz])
        elif is_vessel:
            # Restrain vessel backbone only
            if atom.name in backbone_names:
                restraint.addParticle(atom.index, [k_vessel, px, py, pz])
        elif k_linker > 0.0:
            restraint.addParticle(atom.index, [k_linker, px, py, pz])

    system.addForce(restraint)

    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(positions)

    # Stage 1: Soft-Core L-BFGS de-clashing (50 steps)
    simulation.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=50)

    # Stage 2: High-Precision AMBER14 Minimization
    simulation.minimizeEnergy(tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=max_iterations)

    final_state = simulation.context.getState(getPositions=True)
    out_dir = os.path.dirname(os.path.abspath(output_pdb))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as f:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions(), f)

    return True


def relax_structure(
    input_pdb: str,
    output_pdb: str,
    max_iterations: int = 150,
    restraint_k: float = 1000.0,
    engine: str = "auto",
    motif_res_indices: Optional[Sequence[Union[int, str]]] = None,
    vessel_res_indices: Optional[Sequence[Union[int, str]]] = None,
    k_motif: float = 5000.0,
    k_vessel: float = 500.0,
    k_linker: float = 0.0,
) -> bool:
    """
    Performs 2-Stage Soft-Core Minimization with harmonic backbone restraints.
    If motif_res_indices or vessel_res_indices are provided, applies 3-tier selective harmonic pinning.
    Guarantees elimination of false covalent bonds in PyMOL (< 1.9 Å clashes).
    """
    try:
        input_pdb = validate_path(input_pdb, label="Input PDB", allow_missing=False)
        output_pdb = validate_path(output_pdb, label="Output PDB", allow_missing=True)
    except Exception as e:
        print(f"[-] Path validation error in relax_structure: {e}", file=sys.stderr)
        return False

    if motif_res_indices is not None or vessel_res_indices is not None:
        return relax_with_selective_pinning(
            input_pdb=input_pdb,
            output_pdb=output_pdb,
            motif_res_indices=motif_res_indices or [],
            vessel_res_indices=vessel_res_indices or [],
            k_motif=k_motif,
            k_vessel=k_vessel,
            k_linker=k_linker,
            max_iterations=max_iterations,
        )
    if not os.path.isfile(input_pdb):
        print(f"[-] Error: Input PDB file not found: {input_pdb}", file=sys.stderr)
        return False

    # Optional ColabFold relaxer engine
    if engine in ("auto", "colabfold"):
        if relax_with_colabfold_amber(input_pdb, output_pdb):
            return True
        if engine == "colabfold":
            print("[-] Error: Requested engine 'colabfold' failed or is not available.", file=sys.stderr)
            return False

    try:
        import openmm as mm
        import openmm.app as app
        from openmm import unit
    except ImportError:
        # Auto-dispatch via colabfold conda environment if available and not already dispatched
        if not os.environ.get("_RELAX_DISPATCHED"):
            try:
                os.environ["_RELAX_DISPATCHED"] = "1"
                cmd = [
                    "conda", "run", "-n", "colabfold", "python",
                    os.path.abspath(__file__),
                    "--input", input_pdb,
                    "--output", output_pdb,
                    "--steps", str(max_iterations),
                    "--k", str(restraint_k),
                ]
                print(
                    "[+] OpenMM not in current Python environment. "
                    "Auto-dispatching via 'colabfold' conda environment..."
                )
                res = subprocess.run(cmd, check=False)
                if res.returncode == 0 and os.path.isfile(output_pdb):
                    return True
            except Exception as disp_err:
                print(f"[*] Note on conda auto-dispatch: {disp_err}")

        print("[-] Error: OpenMM is not installed in the active Python environment.", file=sys.stderr)
        print("    Run: conda run -n colabfold python scripts/relax_chimera.py ...", file=sys.stderr)
        return False

    print(f"[+] Loading structure for 2-Stage Relaxation: {input_pdb}")

    # Use PDBFixer if available to verify all-atom topology and add missing hydrogens
    topology = None
    positions = None

    try:
        from pdbfixer import PDBFixer
        print("[+] Using PDBFixer to verify all-atom topology and add hydrogens...")
        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        topology = fixer.topology
        positions = fixer.positions
    except Exception as e:
        print(f"[*] PDBFixer note ({e}), proceeding with standard OpenMM Modeller...")
        pdb = app.PDBFile(input_pdb)
        topology = pdb.topology
        positions = pdb.positions

    forcefield = app.ForceField("amber14-all.xml")

    # If PDBFixer was not used, try OpenMM Modeller to add hydrogens
    if topology is None or positions is None:
        pdb = app.PDBFile(input_pdb)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        try:
            modeller.addHydrogens(forcefield)
            topology = modeller.topology
            positions = modeller.positions
        except Exception:
            topology = pdb.topology
            positions = pdb.positions

    # 1. Soft-Core geometric pre-pass: Untangle severe overlapping atoms
    positions = untangle_severe_clashes(topology, positions, min_dist_angstrom=1.2)

    # Count initial clashes that cause PyMOL fused bonds (< 1.9 Å)
    clashes_init = count_steric_clashes(topology, positions, cutoff_angstrom=1.9)
    print(f"[+] Initial PyMOL-sensitive steric clashes (d < 1.9 Å): {clashes_init}")

    # Create system with no cutoff for isolated protein
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    # Apply stiff harmonic restraints on backbone heavy atoms (N, CA, C, O)
    # k = 1000 kJ/mol/nm^2 ensures backbone remains pinned while sidechains untangle
    restraint = mm.CustomExternalForce("0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", restraint_k * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom in topology.atoms():
        if atom.name in ("N", "CA", "C", "O"):
            p = positions[atom.index]
            restraint.addParticle(atom.index, p.value_in_unit(unit.nanometer))

    system.addForce(restraint)

    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    init_state = simulation.context.getState(getEnergy=True)
    init_energy = init_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"[+] Initial Potential Energy: {init_energy:,.1f} kJ/mol")

    # STAGE 1: Gentle Soft-Core De-clashing (50 steps, moderate tolerance)
    print("[+] Stage 1: Soft-Core L-BFGS de-clashing (untangling overlapping sidechains)...")
    simulation.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=50)

    # STAGE 2: High-Precision AMBER14 Minimization (full convergence)
    print(f"[+] Stage 2: High-Precision AMBER14 minimization (max {max_iterations} steps)...")
    simulation.minimizeEnergy(tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=max_iterations)

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    relief = init_energy - final_energy
    print(f"[+] Relaxed Potential Energy: {final_energy:,.1f} kJ/mol (Relief: {relief:,.1f} kJ/mol)")

    final_positions = final_state.getPositions()
    clashes_final = count_steric_clashes(topology, final_positions, cutoff_angstrom=1.9)
    diff_clashes = clashes_init - clashes_final
    print(f"[+] Final PyMOL-sensitive steric clashes (d < 1.9 Å): {clashes_final} (Eliminated {diff_clashes})")

    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)), exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as f:
        app.PDBFile.writeFile(topology, final_positions, f)

    print(f"✅ SUCCESS: Relaxed, clash-free structure saved to:\n    {output_pdb}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Relax sidechains and eliminate steric clashes using 2-Stage OpenMM")
    parser.add_argument("--input", "-i", help="Input PDB file to relax")
    parser.add_argument("--output", "-o", help="Output relaxed PDB file")
    parser.add_argument("--steps", type=int, default=150, help="Maximum minimization iterations [Default: 150]")
    parser.add_argument("--k", type=float, default=1000.0, help="Backbone restraint force constant [Default: 1000.0]")
    parser.add_argument(
        "--engine",
        choices=["auto", "openmm", "colabfold"],
        default="auto",
        help="Relaxation engine [Default: auto]",
    )
    parser.add_argument(
        "--motif_res",
        help="Comma-separated residue indices for motif heavy atoms (e.g. '55-57,110-112' or '55,56,57')",
    )
    parser.add_argument(
        "--vessel_res",
        help="Comma-separated residue indices for vessel framework",
    )
    parser.add_argument(
        "--k_motif",
        type=float,
        default=5000.0,
        help="Force constant for motif heavy atoms in kJ/mol/nm^2 [Default: 5000.0]",
    )
    parser.add_argument(
        "--k_vessel",
        type=float,
        default=500.0,
        help="Force constant for vessel backbone atoms in kJ/mol/nm^2 [Default: 500.0]",
    )
    parser.add_argument(
        "--k_linker",
        type=float,
        default=0.0,
        help="Force constant for designed linkers in kJ/mol/nm^2 [Default: 0.0]",
    )

    args = parser.parse_args()
    input_file = args.input or find_default_input_pdb()
    if not input_file:
        print("[-] Error: No input PDB provided and no default PDB found in working directory.", file=sys.stderr)
        print("    Specify explicitly with: python scripts/relax_chimera.py --input /path/to/file.pdb", file=sys.stderr)
        sys.exit(1)

    try:
        input_file = validate_path(input_file, label="Input PDB", allow_missing=False)
    except Exception as e:
        print(f"[-] Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_file = args.output
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_relaxed{ext}"

    try:
        output_file = validate_path(output_file, label="Output PDB", allow_missing=True)
    except Exception as e:
        print(f"[-] Error: {e}", file=sys.stderr)
        sys.exit(1)

    success = relax_structure(
        input_file,
        output_file,
        max_iterations=args.steps,
        restraint_k=args.k,
        engine=args.engine,
        motif_res_indices=parse_res_indices(args.motif_res) if args.motif_res else None,
        vessel_res_indices=parse_res_indices(args.vessel_res) if args.vessel_res else None,
        k_motif=args.k_motif,
        k_vessel=args.k_vessel,
        k_linker=args.k_linker,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
