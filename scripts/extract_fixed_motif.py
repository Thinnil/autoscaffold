#!/usr/bin/env python3
"""
scripts/extract_fixed_motif.py
Extract designated motif residue coordinates and sequence from PDB files.

Supports:
- Parsing PDB files
- Extracting specific residues (e.g. "30-45" or "10,12,30-45" or "A30-A45")
- Writing extracted motif coordinates to a clean PDB file
- Generating sequence strings and residue indices
- CLI and Python API usage
"""

import os
import argparse
import re
from typing import List, Dict, Any, Tuple

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEC": "U", "PYL": "O"
}


def parse_residue_spec(spec: str, default_chain: str = "A") -> List[Tuple[str, int]]:
    """
    Parses residue specification string like '30-45,78-82' or 'A30-A45,B12' or '30,31,32'.
    Returns list of (chain_id, res_seq) in order.
    """
    results: List[Tuple[str, int]] = []
    # Strip whitespace and split by comma or whitespace
    tokens = [t.strip() for t in re.split(r'[,;\s]+', spec.strip()) if t.strip()]

    for token in tokens:
        # Check if chain prefix is present, e.g. A30-A45 or A:30-45 or 30-45
        chain = default_chain
        # Only treat as chain prefix if it starts with letters followed immediately by digits: e.g. A30-A45, ChainA:30-45
        m_chain = re.match(r'^([a-zA-Z]+)[:_]?(\d+.*)$', token)
        if m_chain:
            chain = m_chain.group(1)
            token = m_chain.group(2)
        elif ':' in token:
            prefix, remainder = token.split(':', 1)
            chain = prefix.strip()
            token = remainder.strip()

        # Now token is either "START-END" or a single number "NUM"
        if '-' in token:
            parts = token.split('-')
            if len(parts) == 2:
                # Handle possible chain prefix on second part too, e.g. A30-A45 -> 30-A45
                p0 = re.sub(r'^[a-zA-Z]+[:_]?', '', parts[0].strip())
                p1 = re.sub(r'^[a-zA-Z]+[:_]?', '', parts[1].strip())
                try:
                    start_res = int(p0)
                    end_res = int(p1)
                    step = 1 if start_res <= end_res else -1
                    for r in range(start_res, end_res + step, step):
                        results.append((chain, r))
                except ValueError:
                    continue
        else:
            p = re.sub(r'^[a-zA-Z]+[:_]?', '', token.strip())
            try:
                r = int(p)
                results.append((chain, r))
            except ValueError:
                continue

    # Preserve order while removing duplicates
    seen = set()
    ordered = []
    for item in results:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_motif_from_pdb(
    pdb_path: str,
    motif_spec: str,
    default_chain: str = "A",
    out_pdb: str = None
) -> Dict[str, Any]:
    """
    Extracts specified motif residues from pdb_path and optionally writes to out_pdb.
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    target_residues = parse_residue_spec(motif_spec, default_chain=default_chain)
    target_set = set(target_residues)

    extracted_lines = []
    res_info: Dict[Tuple[str, int], str] = {}
    atom_count = 0

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                chain_id = line[21].strip() or default_chain
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    continue

                if (chain_id, res_seq) in target_set:
                    extracted_lines.append(line)
                    atom_count += 1
                    res_name = line[17:20].strip()
                    if (chain_id, res_seq) not in res_info:
                        res_info[(chain_id, res_seq)] = res_name

    # Sequence in user-specified order
    seq_chars = []
    found_residues = []
    for ch, r in target_residues:
        if (ch, r) in res_info:
            r3 = res_info[(ch, r)]
            seq_chars.append(AA_3TO1.get(r3.upper(), "X"))
            found_residues.append((ch, r))

    sequence = "".join(seq_chars)

    if out_pdb and extracted_lines:
        os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
        with open(out_pdb, "w", encoding="utf-8") as out:
            for ln in extracted_lines:
                out.write(ln)
            out.write("TER\nEND\n")

    return {
        "source_pdb": pdb_path,
        "motif_spec": motif_spec,
        "num_residues": len(found_residues),
        "target_residues": target_residues,
        "found_residues": found_residues,
        "sequence": sequence,
        "atom_count": atom_count,
        "out_pdb": out_pdb
    }


def main():
    parser = argparse.ArgumentParser(description="Extract fixed motif residues from a PDB file.")
    parser.add_argument("--pdb", "-p", required=True, help="Input PDB file")
    parser.add_argument("--motif", "-m", required=True, help="Residue range/spec (e.g. 30-45 or 30-45,78-82 or A30-A45)")
    parser.add_argument("--chain", "-c", default="A", help="Default chain ID if not prefixed in spec [Default: A]")
    parser.add_argument("--out", "-o", help="Optional output PDB path")

    args = parser.parse_args()
    res = extract_motif_from_pdb(args.pdb, args.motif, default_chain=args.chain, out_pdb=args.out)
    print(f"Extracted {res['num_residues']} residues ({res['atom_count']} atoms). Sequence: {res['sequence']}")
    if res.get("out_pdb"):
        print(f"Saved to: {res['out_pdb']}")


if __name__ == "__main__":
    main()
