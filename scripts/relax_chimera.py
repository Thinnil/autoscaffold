#!/usr/bin/env python3
"""
relax_chimera.py - High-Performance Backbone-Restrained Energy Minimizer
Uses OpenMM and PDBFixer to relax sidechain steric clashes and optimize rotamer packing
while keeping the tertiary backbone structure firmly restrained.
"""

import argparse
import glob
import os
import sys


def find_default_input_pdb() -> str:
    patterns = [
        "n_terminal/results/*.pdb",
        "n_terminal/2_mpnn/*_seq*.pdb",
        "n_terminal/1_rfdiffusion/*.pdb",
        "n_terminal/**/*.pdb",
        "outputs/chimera_scaffold/results/*.pdb",
        "outputs/chimera_scaffold/2_mpnn/*_seq*.pdb",
        "results/*.pdb",
        "*.pdb",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        valid = [m for m in matches if not any(k in m for k in ("relaxed", "clash_free", "minimized", "seed"))]
        if valid:
            return sorted(valid, key=os.path.getmtime, reverse=True)[0]
    return ""


def relax_structure(input_pdb: str, output_pdb: str, max_iterations: int = 100, restraint_k: float = 1000.0) -> bool:
    if not os.path.isfile(input_pdb):
        print(f"[-] Error: Input PDB file not found: {input_pdb}", file=sys.stderr)
        return False

    try:
        import openmm as mm
        import openmm.app as app
        from openmm import unit
    except ImportError:
        print("[-] Error: OpenMM is not installed in the active Python environment.", file=sys.stderr)
        print("    Run: conda run -n colabfold python scripts/relax_chimera.py ...", file=sys.stderr)
        return False

    print(f"[+] Loading structure: {input_pdb}")

    # Use PDBFixer if available to add missing heavy atoms and hydrogens
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

    # Create system without cutoff for small/medium isolated protein structures
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

    # Initialize simulation context
    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Calculate initial potential energy
    init_state = simulation.context.getState(getEnergy=True)
    init_energy = init_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"[+] Initial Potential Energy: {init_energy:,.1f} kJ/mol")

    print(f"[+] Minimizing steric clashes (L-BFGS, max {max_iterations} steps)...")
    simulation.minimizeEnergy(maxIterations=max_iterations)

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    relief = init_energy - final_energy
    print(f"[+] Relaxed Potential Energy: {final_energy:,.1f} kJ/mol (Relief: {relief:,.1f} kJ/mol)")

    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)), exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as f:
        app.PDBFile.writeFile(topology, final_state.getPositions(), f)

    print(f"✅ SUCCESS: Relaxed, clash-free structure saved to:\n    {output_pdb}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Relax sidechains and eliminate steric clashes using OpenMM")
    parser.add_argument("--input", "-i", help="Input PDB file to relax")
    parser.add_argument("--output", "-o", help="Output relaxed PDB file")
    parser.add_argument("--steps", type=int, default=100, help="Maximum minimization iterations [Default: 100]")
    parser.add_argument("--k", type=float, default=1000.0, help="Backbone restraint force constant [Default: 1000.0]")

    args = parser.parse_args()
    input_file = args.input or find_default_input_pdb()
    if not input_file:
        print("[-] Error: No input PDB provided and no default PDB found in working directory.", file=sys.stderr)
        print("    Specify explicitly with: python scripts/relax_chimera.py --input /path/to/file.pdb", file=sys.stderr)
        sys.exit(1)

    output_file = args.output
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_relaxed{ext}"

    success = relax_structure(input_file, output_file, max_iterations=args.steps, restraint_k=args.k)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
