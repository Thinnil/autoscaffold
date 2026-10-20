#!/usr/bin/env python3
"""
scripts/pse_pdb_extractor.py
Inspect and parse PyMOL session (.pse) files or multi-chain PDB files.

Supports:
1. inspect: Output structured JSON summary listing all detected objects/chains with IDs,
   names, sequence lengths, residue intervals, and sequence strings.
2. extract: Extract individual .pdb files for each object/chain into a target directory.

Features robust headless PyMOL integration if installed, and a comprehensive
standalone fallback for .pse decompression (gzip/zlib + pickle session dict parsing)
and robust multi-chain standard PDB parsing without external dependencies.
"""

import os
import io
import re
import gzip
import zlib
import pickle
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple


class SafeSessionUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that permits only primitive data types, collections,
    and numpy arrays (for scientific coordinates and RFdiffusion metadata),
    strictly blocking arbitrary Python object instantiation and code execution.
    """
    SAFE_CLASSES = {
        # Python 3 builtins
        ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
        ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "int"),
        ("builtins", "float"), ("builtins", "str"), ("builtins", "bytes"),
        ("builtins", "bool"), ("builtins", "bytearray"), ("builtins", "complex"),
        # Python 2 builtins
        ("__builtin__", "dict"), ("__builtin__", "list"), ("__builtin__", "tuple"),
        ("__builtin__", "set"), ("__builtin__", "frozenset"), ("__builtin__", "int"),
        ("__builtin__", "float"), ("__builtin__", "str"), ("__builtin__", "unicode"),
        ("__builtin__", "long"), ("__builtin__", "bool"),
        # Collections
        ("collections", "OrderedDict"), ("collections", "defaultdict"),
        # Codecs / reconstruction
        ("_codecs", "encode"),
        ("copy_reg", "_reconstructor"),
        # NumPy data structures (scientific metadata & RFdiffusion .trb)
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):
        # Safely stub PyMOL internal session classes so sessions can be unpickled without pymol installed
        if module == "pymol" or module.startswith("pymol.") or module.startswith("chempy."):
            class DummySessionClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
            return DummySessionClass
        if (module, name) in self.SAFE_CLASSES:
            return super().find_class(module, name)
        if module in ("builtins", "__builtin__") and name in (
            "dict", "list", "tuple", "set", "frozenset", "int", "float",
            "str", "unicode", "long", "bytes", "bool", "bytearray"
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Security Error: Deserialization of '{module}.{name}' is prohibited to prevent arbitrary code execution."
        )


def safe_pickle_loads(data: bytes, encoding: str = "latin1") -> Any:
    """Safely loads pickled bytes using restricted SafeSessionUnpickler."""
    return SafeSessionUnpickler(io.BytesIO(data), encoding=encoding).load()


def safe_join_path(base_dir: str, *paths: str) -> str:
    """
    Safely joins base_dir and relative paths, preventing directory traversal.
    Sanitizes each path component and ensures the canonical resolved path
    stays strictly within the canonical base_dir.
    """
    base_abs = os.path.realpath(os.path.abspath(base_dir))
    clean_parts = []
    for p in paths:
        sanitized = os.path.basename(str(p).strip())
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', sanitized)
        if not sanitized or sanitized in ('.', '..'):
            sanitized = "unnamed"
        clean_parts.append(sanitized)

    target_abs = os.path.realpath(os.path.abspath(os.path.join(base_abs, *clean_parts)))
    try:
        if os.path.commonpath([base_abs, target_abs]) != base_abs:
            raise ValueError(f"Path traversal detected: {paths} resolves outside {base_dir}")
    except ValueError:
        raise ValueError(f"Path traversal detected: {paths} resolves outside {base_dir}")
    return target_abs

# Standard 3-letter to 1-letter amino acid conversion dictionary
AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEC": "U", "PYL": "O"
}


def aa_to_one_letter(resn: str) -> str:
    return AA_3TO1.get(resn.strip().upper(), "X")


# -----------------------------------------------------------------------------
# PDB Parsing and Extraction Routines
# -----------------------------------------------------------------------------

def parse_pdb_file(pdb_path: str, buffer_size: int = 65536) -> Dict[str, Any]:
    """
    Parses a PDB file and groups atom records by chain ID in a single streaming pass.
    Uses buffered stream reading and avoids redundant line allocations in memory.
    Returns metadata including residue ranges, sequence lengths, and sequences.
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    chains_data: Dict[str, Dict[str, Any]] = {}
    # chain_id -> {res_seq: res_name}
    chain_residues: Dict[str, Dict[int, str]] = {}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore", buffering=buffer_size) as f:
        for line in f:
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                chain_id = line[21].strip() or "A"
                try:
                    res_seq = int(line[22:26])
                except ValueError:
                    continue

                if chain_id not in chain_residues:
                    chain_residues[chain_id] = {}

                if res_seq not in chain_residues[chain_id]:
                    chain_residues[chain_id][res_seq] = line[17:20].strip()

    for chain_id, res_map in chain_residues.items():
        sorted_res_seqs = sorted(res_map.keys())
        if not sorted_res_seqs:
            continue
        min_res = sorted_res_seqs[0]
        max_res = sorted_res_seqs[-1]
        seq = "".join(aa_to_one_letter(res_map[r]) for r in sorted_res_seqs)

        # Build intervals
        intervals: List[Tuple[int, int]] = []
        int_start = sorted_res_seqs[0]
        prev_res = sorted_res_seqs[0]
        for r in sorted_res_seqs[1:]:
            if r == prev_res + 1:
                prev_res = r
            else:
                intervals.append((int_start, prev_res))
                int_start = r
                prev_res = r
        intervals.append((int_start, prev_res))

        interval_strings = [f"{start}-{end}" if start != end else f"{start}" for start, end in intervals]

        chains_data[chain_id] = {
            "id": chain_id,
            "name": f"Chain_{chain_id}",
            "type": "pdb_chain",
            "source_file": pdb_path,
            "length": len(sorted_res_seqs),
            "range": (min_res, max_res),
            "range_str": f"{min_res}-{max_res}",
            "intervals": interval_strings,
            "sequence": seq,
            "residue_seqs": sorted_res_seqs
        }

    return {
        "file_type": "pdb",
        "file_path": pdb_path,
        "objects": chains_data
    }


def extract_pdb_chains(pdb_path: str, out_dir: str, buffer_size: int = 65536) -> Dict[str, str]:
    """
    Extracts individual chains from a multi-chain PDB file into isolated PDB files
    in a single streaming pass using 64KB buffered file readers and writers.
    Avoids re-opening the source file and minimizes memory overhead.
    Returns mapping of chain_id -> extracted_pdb_path.
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    base_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', base_name)

    chain_handles: Dict[str, Any] = {}
    extracted_files: Dict[str, str] = {}

    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore", buffering=buffer_size) as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    ch = line[21].strip() or "A"
                    if ch not in chain_handles:
                        safe_ch = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(ch).strip()) or "A"
                        out_file = safe_join_path(out_dir, f"{base_name}_chain_{safe_ch}.pdb")
                        chain_handles[ch] = open(out_file, "w", encoding="utf-8", buffering=buffer_size)
                        extracted_files[ch] = out_file
                    chain_handles[ch].write(line)
    finally:
        for ch, handle in chain_handles.items():
            handle.write("TER\nEND\n")
            handle.close()

    return extracted_files


# -----------------------------------------------------------------------------
# PyMOL Session (.pse) Parsing and Extraction Routines
# -----------------------------------------------------------------------------

def try_extract_pse_with_pymol(pse_path: str, out_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Attempts to use headless PyMOL if installed in the current environment.
    """
    try:
        import pymol
        pymol.finish_launching(['pymol', '-cq'])
        from pymol import cmd

        cmd.reinitialize()
        cmd.load(pse_path)
        obj_names = cmd.get_names("objects")

        objects_data = {}
        extracted_files = {}

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        for obj in obj_names:
            pdb_str = cmd.get_pdbstr(obj)
            if not pdb_str or not pdb_str.strip():
                continue

            # Parse the pdb string to get metadata
            mem_parsed = _parse_pdb_string(pdb_str, name=obj, source=pse_path)
            objects_data[obj] = mem_parsed

            if out_dir:
                safe_obj = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(obj).strip()) or "obj"
                out_path = safe_join_path(out_dir, f"{safe_obj}.pdb")
                cmd.save(out_path, obj)
                extracted_files[obj] = out_path

        return {
            "file_type": "pse",
            "file_path": pse_path,
            "engine": "pymol_headless",
            "objects": objects_data,
            "extracted_files": extracted_files
        }
    except Exception:
        return None


def _parse_pdb_string(pdb_str: str, name: str, source: str) -> Dict[str, Any]:
    """Helper to parse a PDB string into object metadata."""
    res_map: Dict[int, Dict[str, Any]] = {}
    chains = set()

    for line in pdb_str.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            ch = line[21].strip() or "A"
            chains.add(ch)
            resn = line[17:20].strip()
            try:
                resi = int(line[22:26].strip())
            except ValueError:
                continue
            if resi not in res_map:
                res_map[resi] = {"res_seq": resi, "res_name": resn}

    sorted_res = sorted(res_map.keys())
    if not sorted_res:
        return {
            "id": name,
            "name": name,
            "type": "pse_object",
            "source_file": source,
            "length": 0,
            "range": (0, 0),
            "range_str": "0-0",
            "intervals": [],
            "sequence": "",
            "chains": list(chains)
        }

    seq = "".join(aa_to_one_letter(res_map[r]["res_name"]) for r in sorted_res)
    min_res = sorted_res[0]
    max_res = sorted_res[-1]

    intervals: List[Tuple[int, int]] = []
    int_start = sorted_res[0]
    prev_res = sorted_res[0]
    for r in sorted_res[1:]:
        if r == prev_res + 1:
            prev_res = r
        else:
            intervals.append((int_start, prev_res))
            int_start = r
            prev_res = r
    intervals.append((int_start, prev_res))

    interval_strings = [f"{start}-{end}" if start != end else f"{start}" for start, end in intervals]

    return {
        "id": name,
        "name": name,
        "type": "pse_object",
        "source_file": source,
        "length": len(sorted_res),
        "range": (min_res, max_res),
        "range_str": f"{min_res}-{max_res}",
        "intervals": interval_strings,
        "sequence": seq,
        "chains": sorted(list(chains)),
        "residue_seqs": sorted_res
    }


def parse_pse_standalone(pse_path: str) -> Dict[str, Any]:
    """
    Robust standalone fallback parser for PyMOL session (.pse) files.
    Decompresses gzip or zlib streams and unpickles session data structures.
    PyMOL session dictionaries contain an entry 'names' where objects are defined.
    Object structures contain atom records with coordinates, residue names, numbers,
    chains, and atom names.
    """
    if not os.path.isfile(pse_path):
        raise FileNotFoundError(f"PSE file not found: {pse_path}")

    # Step 1: Read raw bytes and decompress
    with open(pse_path, "rb") as f:
        raw_data = f.read()

    decompressed_data = None
    # Try gzip decompression
    try:
        decompressed_data = gzip.decompress(raw_data)
    except Exception:
        pass

    # Try zlib decompression if gzip failed
    if decompressed_data is None:
        try:
            decompressed_data = zlib.decompress(raw_data)
        except Exception:
            pass

    # If already uncompressed or raw pickle
    if decompressed_data is None:
        decompressed_data = raw_data

    # Step 2: Safely unpickle session object using restricted SafeSessionUnpickler
    # PyMOL sessions can be pickled with Python 2 or 3 protocol
    session_data = None
    try:
        session_data = safe_pickle_loads(decompressed_data, encoding="latin1")
    except Exception:
        try:
            session_data = safe_pickle_loads(decompressed_data)
        except Exception as e:
            raise RuntimeError(f"Failed to safely unpickle PyMOL session file: {e}")

    # In PyMOL PSE, session_data is typically a dict or list.
    # Top-level dict usually contains keys like 'names', 'version', 'settings'.
    # If session_data is a list, the dict is usually session_data[0].
    session_dict = {}
    if isinstance(session_data, dict):
        session_dict = session_data
    elif isinstance(session_data, list) and len(session_data) > 0 and isinstance(session_data[0], dict):
        session_dict = session_data[0]
    elif isinstance(session_data, (list, tuple)):
        for item in session_data:
            if isinstance(item, dict) and "names" in item:
                session_dict = item
                break

    objects_dict = {}
    extracted_obj_atoms: Dict[str, List[Dict[str, Any]]] = {}

    # Locate 'names' list/dict in session_dict
    names_entry = session_dict.get("names", None)
    if names_entry is None:
        # Search recursively for 'names' key
        def find_names(obj, depth=0):
            if depth > 4:
                return None
            if isinstance(obj, dict):
                if "names" in obj and isinstance(obj["names"], (list, dict)):
                    return obj["names"]
                for v in obj.values():
                    found = find_names(v, depth + 1)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = find_names(item, depth + 1)
                    if found is not None:
                        return found
            return None
        names_entry = find_names(session_dict)

    if names_entry is not None:
        items = names_entry.items() if isinstance(names_entry, dict) else enumerate(names_entry)
        raw_molecules = {}
        selections = {}

        for _, obj_record in items:
            if not isinstance(obj_record, (list, tuple)) or len(obj_record) < 6:
                continue

            obj_name = str(obj_record[0]).strip()
            if not obj_name or obj_name.lower() in ("none", "null"):
                continue

            mol_data = obj_record[5]
            if not isinstance(mol_data, (list, tuple)):
                continue

            # Check if this is a PyMOL Selection object (maps to parent object atom indices)
            if len(mol_data) == 1 and isinstance(mol_data[0], (list, tuple)) and len(mol_data[0]) >= 2:
                parent_name = str(mol_data[0][0]).strip()
                atom_indices = mol_data[0][1]
                if isinstance(atom_indices, (list, tuple)):
                    selections[obj_name] = (parent_name, set(atom_indices))
                    continue

            # Check if this is a Molecule object containing an atom list
            if len(mol_data) > 7 and isinstance(mol_data[7], (list, tuple)) and len(mol_data[7]) > 0:
                raw_molecules[obj_name] = mol_data

        def parse_atoms_from_mol(atom_list, mol):
            flat_coords = None
            idx_map = None
            for c_idx in [4, 3, 2, 5]:
                if c_idx < len(mol) and isinstance(mol[c_idx], (list, tuple)) and len(mol[c_idx]) > 0:
                    cs0 = mol[c_idx][0]
                    if isinstance(cs0, (list, tuple)) and len(cs0) >= 3:
                        for sub in cs0:
                            if isinstance(sub, (list, tuple)) and len(sub) > 10 and isinstance(sub[0], (int, float)):
                                flat_coords = sub
                                break
                        if len(cs0) > 4 and isinstance(cs0[4], (list, tuple)) and len(cs0[4]) == len(atom_list):
                            idx_map = cs0[4]
                    if flat_coords is not None:
                        break

            atoms = []
            for at_idx, at in enumerate(atom_list):
                if not isinstance(at, (list, tuple)) or len(at) < 7:
                    continue

                if len(at) > 6 and isinstance(at[5], str) and len(at[5]) == 3:
                    chain = str(at[1]).strip() or (str(at[4]).strip() if len(at) > 4 else "A") or "A"
                    resn = str(at[5]).strip()
                    atom_name = str(at[6]).strip()
                    resi_str = str(at[3]).strip()
                    elem = str(at[7]).strip() if len(at) > 7 and isinstance(at[7], str) else (atom_name[0] if atom_name else "C")
                else:
                    atom_name = str(at[1]).strip() if len(at) > 1 else "CA"
                    resn = str(at[2]).strip() if len(at) > 2 else "UNK"
                    resi_str = str(at[3]).strip() if len(at) > 3 else "1"
                    chain = str(at[4]).strip() if len(at) > 4 else "A"
                    elem = atom_name[0] if atom_name else "C"

                try:
                    resi = int(resi_str)
                except ValueError:
                    digits = "".join([c for c in resi_str if c.isdigit() or c == '-'])
                    resi = int(digits) if digits else 1

                coord_idx = idx_map[at_idx] if idx_map is not None and at_idx < len(idx_map) else at_idx
                if coord_idx < 0 or flat_coords is None or (coord_idx * 3 + 2) >= len(flat_coords):
                    continue
                x = float(flat_coords[coord_idx * 3])
                y = float(flat_coords[coord_idx * 3 + 1])
                z = float(flat_coords[coord_idx * 3 + 2])

                atoms.append({
                    "atom_name": atom_name,
                    "resn": resn,
                    "resi": resi,
                    "chain": chain,
                    "x": x,
                    "y": y,
                    "z": z,
                    "b_factor": 20.0,
                    "element": elem,
                    "_orig_idx": at_idx
                })
            return atoms

        # Extract atoms for each molecule
        for obj_name, mol in raw_molecules.items():
            atoms = parse_atoms_from_mol(mol[7], mol)
            if not atoms:
                continue
            extracted_obj_atoms[obj_name] = atoms

            # If multi-chain, also create discrete sub-entities for each chain
            chains = sorted(list(set(a["chain"] for a in atoms)))
            if len(chains) > 1:
                for ch in chains:
                    sub_atoms = [a for a in atoms if a["chain"] == ch]
                    if sub_atoms:
                        extracted_obj_atoms[f"{obj_name}_chain_{ch}"] = sub_atoms

        # Extract atoms for any selections
        for sel_name, (parent_name, indices) in selections.items():
            if parent_name in extracted_obj_atoms:
                p_atoms = extracted_obj_atoms[parent_name]
                sel_atoms = [a for a in p_atoms if a.get("_orig_idx") in indices]
                if sel_atoms:
                    extracted_obj_atoms[sel_name] = sel_atoms

    # Generate metadata for detected objects
    for obj_name, atoms in extracted_obj_atoms.items():
        res_map = {}
        chains = set()
        for a in atoms:
            chains.add(a["chain"])
            resi = a["resi"]
            if resi not in res_map:
                res_map[resi] = a["resn"]

        sorted_res = sorted(res_map.keys())
        min_res = sorted_res[0] if sorted_res else 0
        max_res = sorted_res[-1] if sorted_res else 0
        seq = "".join(aa_to_one_letter(res_map[r]) for r in sorted_res)

        intervals: List[Tuple[int, int]] = []
        if sorted_res:
            int_start = sorted_res[0]
            prev_res = sorted_res[0]
            for r in sorted_res[1:]:
                if r == prev_res + 1:
                    prev_res = r
                else:
                    intervals.append((int_start, prev_res))
                    int_start = r
                    prev_res = r
            intervals.append((int_start, prev_res))
        interval_strings = [f"{start}-{end}" if start != end else f"{start}" for start, end in intervals]

        objects_dict[obj_name] = {
            "id": obj_name,
            "name": obj_name,
            "type": "pse_object",
            "source_file": pse_path,
            "length": len(sorted_res),
            "range": (min_res, max_res),
            "range_str": f"{min_res}-{max_res}",
            "intervals": interval_strings,
            "sequence": seq,
            "chains": sorted(list(chains)),
            "residue_seqs": sorted_res,
            "_atoms": atoms
        }

    return {
        "file_type": "pse",
        "file_path": pse_path,
        "engine": "standalone_pickle_fallback",
        "objects": objects_dict
    }


def write_atoms_to_pdb(atoms: List[Dict[str, Any]], out_path: str, buffer_size: int = 65536):
    """Writes atom dictionaries to a standard formatted PDB file using 64KB buffered I/O."""
    lines = []
    for idx, a in enumerate(atoms, start=1):
        name = a["atom_name"]
        atom_fmt = f" {name:<3}" if len(name) < 4 else f"{name[:4]}"
        resn = a["resn"][:3]
        chain = a["chain"][:1] if a["chain"] else "A"
        resi = a["resi"]
        x, y, z = a["x"], a["y"], a["z"]
        b_factor = a.get("b_factor", 20.0)
        elem = a.get("element", "C")[:2]
        line = f"ATOM  {idx:>5} {atom_fmt} {resn:>3} {chain}{resi:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}          {elem:>2}\n"
        lines.append(line)
    lines.append("TER\nEND\n")

    with open(out_path, "w", encoding="utf-8", buffering=buffer_size) as out:
        out.writelines(lines)


def extract_pse_objects(pse_path: str, out_dir: str) -> Dict[str, str]:
    """
    Extracts each object in the PSE file to an individual PDB file.
    Tries headless PyMOL first, falls back to standalone pickle parser.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Try headless PyMOL
    result = try_extract_pse_with_pymol(pse_path, out_dir)
    if result and result.get("extracted_files"):
        return result["extracted_files"]

    # Fallback to standalone parser
    parsed = parse_pse_standalone(pse_path)
    extracted_files = {}
    for obj_name, obj_meta in parsed["objects"].items():
        atoms = obj_meta.get("_atoms", [])
        if not atoms:
            continue
        safe_obj = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(obj_name).strip()) or "obj"
        out_file = safe_join_path(out_dir, f"{safe_obj}.pdb")
        write_atoms_to_pdb(atoms, out_file)
        extracted_files[obj_name] = out_file

    return extracted_files


# -----------------------------------------------------------------------------
# Unified Inspector / CLI API
# -----------------------------------------------------------------------------

def inspect_file(input_path: str) -> Dict[str, Any]:
    """
    Inspects either a PDB or PSE file and returns a structured dictionary
    describing all detected structural entities (chains or objects).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pse":
        # Try PyMOL first
        result = try_extract_pse_with_pymol(input_path)
        if not result or not result.get("objects"):
            result = parse_pse_standalone(input_path)
        # Clean private fields like _atoms before returning
        clean_objs = {}
        for k, v in result["objects"].items():
            clean_v = {ck: cv for ck, cv in v.items() if not ck.startswith("_")}
            clean_objs[k] = clean_v
        result["objects"] = clean_objs
        return result
    else:
        # Default treat as PDB
        return parse_pdb_file(input_path)


def extract_file(input_path: str, out_dir: str) -> Dict[str, str]:
    """
    Extracts individual entities (chains for PDB, objects for PSE) into out_dir.
    Returns mapping: id -> extracted_pdb_path.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pse":
        return extract_pse_objects(input_path, out_dir)
    else:
        return extract_pdb_chains(input_path, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Inspect and extract chains/objects from PDB or PyMOL session (.pse) files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect PDB or PSE and output JSON summary")
    inspect_parser.add_argument("--input", "-i", required=True, help="Input PDB or PSE file")
    inspect_parser.add_argument("--out_json", "-o", help="Optional output JSON file")

    # Subcommand: extract
    extract_parser = subparsers.add_parser("extract", help="Extract individual chains/objects to PDBs")
    extract_parser.add_argument("--input", "-i", required=True, help="Input PDB or PSE file")
    extract_parser.add_argument("--out_dir", "-d", required=True, help="Directory to save extracted PDB files")

    args = parser.parse_args()

    if args.command == "inspect":
        info = inspect_file(args.input)
        json_output = json.dumps(info, indent=2)
        if args.out_json:
            out_json_path = os.path.realpath(os.path.abspath(args.out_json))
            if out_json_path in ("/", "\\") or (len(out_json_path) <= 3 and out_json_path.endswith(":\\")):
                raise ValueError(f"Dangerous output JSON path: {out_json_path}")
            out_json_dir = os.path.dirname(out_json_path)
            if out_json_dir:
                os.makedirs(out_json_dir, exist_ok=True)
            with open(out_json_path, "w", encoding="utf-8", buffering=65536) as f:
                f.write(json_output)
        print(json_output)

    elif args.command == "extract":
        out_dir = os.path.realpath(os.path.abspath(args.out_dir))
        if out_dir in ("/", "\\") or (len(out_dir) <= 3 and out_dir.endswith(":\\")):
            raise ValueError(f"Dangerous output directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        extracted = extract_file(args.input, out_dir)
        print(json.dumps({"status": "SUCCESS", "extracted_files": extracted}, indent=2))


if __name__ == "__main__":
    main()
