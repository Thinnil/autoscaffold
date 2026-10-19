#!/usr/bin/env python3
"""
scripts/format_complex_fasta.py
Utilities to format complex FASTA files and ProteinMPNN fixed-position masks.

Supports:
- Extracting sequence from PDB files
- Multi-chain complex formatting (using ':' or separate headers)
- Generating fixed_positions_dict.jsonl for ProteinMPNN masking
"""

import os
import argparse
import json
from typing import Dict, List, Any, Optional

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEC": "U", "PYL": "O"
}


def get_pdb_chains_sequences(pdb_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses PDB and extracts per-chain sequence and 1-based residue numbers.
    """
    chain_residues: Dict[str, Dict[int, str]] = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                chain_id = line[21].strip() or "A"
                res_name = line[17:20].strip()
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    continue

                if chain_id not in chain_residues:
                    chain_residues[chain_id] = {}
                if res_seq not in chain_residues[chain_id]:
                    chain_residues[chain_id][res_seq] = res_name

    results = {}
    for ch, r_map in chain_residues.items():
        sorted_res = sorted(r_map.keys())
        seq = "".join(AA_3TO1.get(r_map[r].upper(), "X") for r in sorted_res)
        results[ch] = {
            "sequence": seq,
            "residue_indices": sorted_res,
            "length": len(sorted_res)
        }
    return results


def format_complex_fasta(
    pdb_path: str,
    out_fasta: Optional[str] = None,
    header_prefix: Optional[str] = None
) -> str:
    """
    Creates a FASTA representation of all chains in a PDB structure.
    """
    chains = get_pdb_chains_sequences(pdb_path)
    base_name = header_prefix or os.path.splitext(os.path.basename(pdb_path))[0]

    lines = []
    for ch, data in chains.items():
        lines.append(f">{base_name}|Chain_{ch}|Length_{data['length']}")
        lines.append(data["sequence"])

    fasta_content = "\n".join(lines) + "\n"

    if out_fasta:
        os.makedirs(os.path.dirname(os.path.abspath(out_fasta)), exist_ok=True)
        with open(out_fasta, "w", encoding="utf-8") as f:
            f.write(fasta_content)

    return fasta_content


def make_fixed_positions_dict(
    pdb_name: str,
    fixed_positions: Dict[str, List[int]],
    out_jsonl: Optional[str] = None
) -> Dict[str, Dict[str, List[int]]]:
    """
    Generates a ProteinMPNN compatible fixed_positions_dict.jsonl structure:
    {"pdb_name": {"A": [1, 2, 3, ...], "B": [...]}}
    Positions must be 1-based indices relative to the chain in the PDB.
    """
    data = {pdb_name: fixed_positions}
    if out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description="Format PDB into complex FASTA or fixed positions jsonl.")
    parser.add_argument("--pdb", "-p", required=True, help="Input PDB file")
    parser.add_argument("--out_fasta", "-o", help="Output FASTA file")
    parser.add_argument("--fixed_chain", help="Chain to fix positions in")
    parser.add_argument("--fixed_res", help="Comma-separated residue numbers or range (e.g. 10-25)")
    parser.add_argument("--out_jsonl", help="Output jsonl for ProteinMPNN")

    args = parser.parse_args()

    fasta = format_complex_fasta(args.pdb, out_fasta=args.out_fasta)
    print(fasta)

    if args.fixed_chain and args.fixed_res and args.out_jsonl:
        from scripts.extract_fixed_motif import parse_residue_spec
        res_list = [r for _, r in parse_residue_spec(args.fixed_res)]
        base = os.path.splitext(os.path.basename(args.pdb))[0]
        make_fixed_positions_dict(base, {args.fixed_chain: res_list}, out_jsonl=args.out_jsonl)
        print(f"Wrote fixed positions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
