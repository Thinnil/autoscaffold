#!/usr/bin/env python3
"""
scripts/kabsch_rmsd.py
High-precision Kabsch structural alignment and RMSD calculation.

Calculates:
- Optimal rotation matrix (R) and translation vector (t) via Kabsch algorithm (SVD)
- Handles right-handed vs left-handed reflection coordinate corrections
- C-alpha RMSD and all-atom / backbone (N, CA, C, O) RMSD between reference motif
  and candidate designs
- Residue-by-residue alignment and distance distribution
- PDB file parsing for coordinates of specific residue ranges/chains
"""

import argparse
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def kabsch_fit(
    P: np.ndarray,
    Q: np.ndarray,
    return_aligned: bool = False
) -> Any:
    """
    Computes optimal rotation matrix R and translation vector t to align P onto Q:
        Q_approx = np.dot(P, R) + t
    Both P and Q must be (N, 3) arrays where N >= 3.
    Returns (R, t, rmsd) or (R, t, rmsd, P_aligned, sq_dist) if return_aligned is True.
    Optimized for in-memory execution without redundant matrix allocations.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch in Kabsch fit: P is {P.shape}, Q is {Q.shape}")
    if len(P.shape) != 2 or P.shape[1] != 3:
        raise ValueError(f"Arrays must have shape (N, 3), got {P.shape}")

    n = P.shape[0]
    if n == 0:
        if return_aligned:
            return np.eye(3), np.zeros(3), 0.0, np.empty((0, 3)), np.empty(0)
        return np.eye(3), np.zeros(3), 0.0

    p_center = np.mean(P, axis=0)
    q_center = np.mean(Q, axis=0)

    p_centered = P - p_center
    q_centered = Q - q_center

    # Covariance matrix H = P^T * Q (3x3)
    H = np.dot(p_centered.T, q_centered)

    # Singular Value Decomposition
    V, S, Wt = np.linalg.svd(H)

    # Reflection check: det(V)*det(Wt) < 0 indicates reflection
    if (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0:
        V[:, -1] *= -1.0

    # Compute optimal rotation matrix R = V * Wt once
    R = np.dot(V, Wt)

    # Translation vector
    t = q_center - np.dot(p_center, R)

    # Calculate aligned coordinates and RMSD with minimal intermediate allocations
    P_aligned = np.dot(P, R)
    P_aligned += t
    diff = P_aligned - Q
    sq_dist = np.sum(diff ** 2, axis=1)
    rmsd = float(np.sqrt(np.mean(sq_dist)))

    if return_aligned:
        return R, t, rmsd, P_aligned, sq_dist
    return R, t, rmsd


def extract_coordinates(
    pdb_path: str,
    chain_id: Optional[str] = None,
    residue_list: Optional[List[int]] = None,
    atom_names: Optional[List[str]] = None,
    extract_meta: bool = True,
    buffer_size: int = 65536
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts atom coordinates from a PDB file using 64KB buffered stream reading.
    If extract_meta is False, skips per-atom dictionary allocations for maximum speed.
    Returns (coords_array (N, 3), atom_metadata_list).
    """
    if atom_names is None:
        atom_names = ["CA"]
    atom_names_upper = set(a.upper().strip() for a in atom_names)

    coords = []
    meta = []

    res_set = set(residue_list) if residue_list is not None else None

    with open(pdb_path, "r", encoding="utf-8", errors="ignore", buffering=buffer_size) as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                ch = line[21].strip() or "A"
                if chain_id is not None and ch != chain_id:
                    continue

                try:
                    resi = int(line[22:26])
                except ValueError:
                    continue

                if res_set is not None and resi not in res_set:
                    continue

                atom_name = line[12:16].strip().upper()
                if atom_name not in atom_names_upper:
                    continue

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue

                coords.append([x, y, z])
                if extract_meta:
                    meta.append({
                        "chain": ch,
                        "resi": resi,
                        "resn": line[17:20].strip(),
                        "atom": atom_name,
                        "line": line.rstrip()
                    })

    return np.array(coords, dtype=float), meta


def calculate_motif_rmsd(
    ref_pdb: str,
    query_pdb: str,
    ref_chain: Optional[str] = None,
    ref_residues: Optional[List[int]] = None,
    query_chain: Optional[str] = None,
    query_residues: Optional[List[int]] = None,
    atom_type: str = "CA"
) -> Dict[str, Any]:
    """
    Calculates the Kabsch RMSD between a reference motif and query design.
    Reuses aligned coordinates from SVD fit to prevent redundant matrix operations.
    """
    atoms = ["CA"] if atom_type == "CA" else ["N", "CA", "C", "O"]

    ref_coords, _ = extract_coordinates(
        ref_pdb, chain_id=ref_chain, residue_list=ref_residues, atom_names=atoms, extract_meta=False
    )
    query_coords, _ = extract_coordinates(
        query_pdb, chain_id=query_chain, residue_list=query_residues, atom_names=atoms, extract_meta=False
    )

    if len(ref_coords) == 0:
        raise ValueError(f"No matching atoms found in reference PDB {ref_pdb} for chain={ref_chain}, res={ref_residues}")
    if len(query_coords) == 0:
        raise ValueError(f"No matching atoms found in query PDB {query_pdb} for chain={query_chain}, res={query_residues}")

    if len(ref_coords) != len(query_coords):
        # Match as many corresponding atoms as possible by position
        min_len = min(len(ref_coords), len(query_coords))
        ref_coords = ref_coords[:min_len]
        query_coords = query_coords[:min_len]

    R, t, rmsd, _, sq_dist = kabsch_fit(query_coords, ref_coords, return_aligned=True)

    # Per-residue deviations derived directly from in-memory squared distance array
    dists = np.sqrt(sq_dist).tolist()

    return {
        "rmsd": round(float(rmsd), 4),
        "num_atoms": int(len(ref_coords)),
        "atom_type": atom_type,
        "max_deviation": round(float(max(dists)), 4) if dists else 0.0,
        "mean_deviation": round(float(np.mean(dists)), 4) if dists else 0.0,
        "per_atom_deviations": [round(d, 4) for d in dists]
    }


def main():
    parser = argparse.ArgumentParser(description="Kabsch RMSD structural alignment between two structures.")
    parser.add_argument("--ref", "-r", required=True, help="Reference PDB (containing target motif)")
    parser.add_argument("--query", "-q", required=True, help="Query/Candidate PDB")
    parser.add_argument("--ref_chain", default=None, help="Reference chain ID")
    parser.add_argument("--ref_res", default=None, help="Reference residue range (e.g. 30-45 or 30,31,32)")
    parser.add_argument("--query_chain", default=None, help="Query chain ID")
    parser.add_argument("--query_res", default=None, help="Query residue range (e.g. 50-65)")
    parser.add_argument("--atoms", default="CA", choices=["CA", "backbone"], help="Atoms to align: CA or backbone (N,CA,C,O)")

    args = parser.parse_args()

    # Parse ranges if given
    from scripts.extract_fixed_motif import parse_residue_spec
    ref_res_list = [r for _, r in parse_residue_spec(args.ref_res)] if args.ref_res else None
    query_res_list = [r for _, r in parse_residue_spec(args.query_res)] if args.query_res else None

    result = calculate_motif_rmsd(
        args.ref,
        args.query,
        ref_chain=args.ref_chain,
        ref_residues=ref_res_list,
        query_chain=args.query_chain,
        query_residues=query_res_list,
        atom_type=args.atoms
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
