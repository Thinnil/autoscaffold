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
        ("copyreg", "_reconstructor"),
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


# Pre-compiled regular expressions for path validation, deduplication, and identifier sanitization
_RE_SANITIZE_FILENAME = re.compile(r'[^a-zA-Z0-9_\-\.]')
_RE_SANITIZE_IDENT = re.compile(r'[^a-zA-Z0-9_\-]')
_RE_COMMA_SPLIT = re.compile(r'[,]+')
_RE_DRIVE_ROOT = re.compile(r'^[a-zA-Z]:[\\/]?$')
_RE_DRIVE_PREFIX = re.compile(r'^[a-zA-Z]:')
_RE_OUTPUTS_DEDUP_START = re.compile(r'(^|[/\\])outputs([/\\]outputs)+', re.IGNORECASE)
_RE_OUTPUTS_DEDUP_MID = re.compile(r'([/\\]outputs)+([/\\]outputs)', re.IGNORECASE)

_SYS_EXACT: Tuple[str, ...] = (
    "/etc", "/usr", "/bin", "/sbin", "/boot", "/dev", "/sys", "/proc",
    "/windows", "/program files", "/program files (x86)", "/system32"
)
_SYS_PREFIXES: Tuple[str, ...] = tuple(s + "/" for s in _SYS_EXACT)


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
        sanitized = _RE_SANITIZE_FILENAME.sub('_', sanitized)
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


def validate_output_dir(out_dir: str) -> str:
    """
    Validates that out_dir is non-empty, not root, and not a dangerous system directory.
    De-duplicates accidental nested 'outputs/outputs' path segments using pre-compiled regexes.
    Returns canonical real path.
    """
    if not out_dir or not str(out_dir).strip():
        raise ValueError("Output directory cannot be empty.")
    raw_str = str(out_dir).strip()

    if _RE_DRIVE_ROOT.match(raw_str) or raw_str in ("/", "\\", "//", "\\\\"):
        raise ValueError(f"Dangerous or root output directory specified: '{out_dir}'")

    # Fast check: only execute regex deduplication if 'outputs' substring is present
    if "outputs" in raw_str.lower():
        dedup_str = _RE_OUTPUTS_DEDUP_START.sub(r'\1outputs', raw_str)
        dedup_str = _RE_OUTPUTS_DEDUP_MID.sub(r'\1', dedup_str)
    else:
        dedup_str = raw_str

    real_out = os.path.realpath(os.path.abspath(dedup_str))

    # Also collapse any duplicate outputs segments resulting from abspath joining with PWD
    if "outputs" in real_out.lower():
        real_out = _RE_OUTPUTS_DEDUP_MID.sub(r'\1', real_out)
        real_out = os.path.realpath(os.path.abspath(real_out))

    # Bulletproof root check for POSIX (/), Windows drive roots (C:\, C:/, C:), UNC roots (\\, //)
    drive, rest = os.path.splitdrive(real_out)
    if (not drive and real_out.rstrip("/\\") == "") or (drive and rest in ("/", "\\", "")):
        raise ValueError(f"Dangerous or root output directory specified: '{out_dir}'")
    if real_out in ("/", "\\", "//", "\\\\") or (len(real_out) <= 3 and (real_out.endswith(":\\") or real_out.endswith(":/"))):
        raise ValueError(f"Dangerous or root output directory specified: '{out_dir}'")

    # Check normalized paths with forward slashes across POSIX and Windows drives
    norm_posix = real_out.replace("\\", "/").lower()
    raw_posix = raw_str.replace("\\", "/").lower()
    drive_stripped = _RE_DRIVE_PREFIX.sub('', norm_posix)

    for check_path in (norm_posix, drive_stripped, raw_posix):
        if check_path in _SYS_EXACT or check_path.startswith(_SYS_PREFIXES):
            raise ValueError(f"Output directory cannot be in system directory: '{out_dir}'")
    return real_out


def validate_pse_safety(pse_path: str) -> None:
    """
    Strictly pre-validates that a .pse file does not contain unsafe pickle classes
    by executing parse_pse_standalone with SafeSessionUnpickler.
    Raises RuntimeError or pickle.UnpicklingError if prohibited classes are present.
    """
    parse_pse_standalone(pse_path)


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


# In-memory session cache for parsed PDB metadata:
# Key: (real_path, mtime) -> dict
_PDB_PARSE_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}


def clear_pdb_parse_cache() -> None:
    """Clears the in-memory parsed PDB file metadata cache."""
    _PDB_PARSE_CACHE.clear()


# -----------------------------------------------------------------------------
# PDB Parsing and Extraction Routines
# -----------------------------------------------------------------------------

def parse_pdb_file(pdb_path: str, buffer_size: int = 65536, use_cache: bool = True) -> Dict[str, Any]:
    """
    Parses a PDB file and groups atom records by chain ID in a single streaming pass.
    Uses buffered stream reading and avoids redundant line allocations in memory.
    Caches parsed metadata by file mtime to avoid redundant re-reading across the session.
    Returns metadata including residue ranges, sequence lengths, and sequences.
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    real_path = os.path.realpath(pdb_path)
    try:
        mtime = os.path.getmtime(real_path)
    except OSError:
        mtime = 0.0

    cache_key = (real_path, mtime)
    if use_cache and cache_key in _PDB_PARSE_CACHE:
        # Return a copy of cached metadata so caller modifications don't mutate cache
        cached = _PDB_PARSE_CACHE[cache_key]
        return {
            "file_type": cached["file_type"],
            "file_path": pdb_path,
            "objects": {k: dict(v) for k, v in cached["objects"].items()}
        }

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

    result = {
        "file_type": "pdb",
        "file_path": pdb_path,
        "objects": chains_data
    }

    if use_cache:
        _PDB_PARSE_CACHE[cache_key] = result

    return result


def extract_pdb_chains(pdb_path: str, out_dir: str, buffer_size: int = 65536) -> Dict[str, str]:
    """
    Extracts individual chains from a multi-chain PDB file into isolated PDB files
    in a single streaming pass using 64KB buffered file readers and writers.
    Avoids re-opening the source file and minimizes memory overhead.
    Returns mapping of chain_id -> extracted_pdb_path.
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    out_dir = validate_output_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    base_name = _RE_SANITIZE_IDENT.sub('_', base_name)

    chain_handles: Dict[str, Any] = {}
    extracted_files: Dict[str, str] = {}

    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore", buffering=buffer_size) as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    ch = line[21].strip() or "A"
                    if ch not in chain_handles:
                        safe_ch = _RE_SANITIZE_IDENT.sub('_', str(ch).strip()) or "A"
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
    Strictly pre-validates the session file using SafeSessionUnpickler to prevent
    arbitrary code execution prior to invoking PyMOL's native loader.
    """
    if out_dir:
        out_dir = validate_output_dir(out_dir)

    # Security Pre-Validation: strictly enforce SafeSessionUnpickler on the session file
    # before allowing PyMOL to load or unpickle it.
    validate_pse_safety(pse_path)

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
                safe_obj = _RE_SANITIZE_IDENT.sub('_', str(obj).strip()) or "obj"
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
                    elem = (
                        str(at[7]).strip()
                        if len(at) > 7 and isinstance(at[7], str)
                        else (atom_name[0] if atom_name else "C")
                    )
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


def write_atoms_to_pdb(atoms: List[Dict[str, Any]], out_path: str, buffer_size: int = 65536) -> None:
    """Writes atom dictionaries to a standard formatted PDB file using 64KB buffered stream I/O."""
    with open(out_path, "w", encoding="utf-8", buffering=buffer_size) as out:
        for idx, a in enumerate(atoms, start=1):
            name = a["atom_name"]
            atom_fmt = f" {name:<3}" if len(name) < 4 else f"{name[:4]}"
            resn = a["resn"][:3]
            chain = a["chain"][:1] if a["chain"] else "A"
            resi = a["resi"]
            x, y, z = a["x"], a["y"], a["z"]
            b_factor = a.get("b_factor", 20.0)
            elem = a.get("element", "C")[:2]
            out.write(
                f"ATOM  {idx:>5} {atom_fmt} {resn:>3} {chain}{resi:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}          {elem:>2}\n"
            )
        out.write("TER\nEND\n")


def extract_pse_objects(pse_path: str, out_dir: str) -> Dict[str, str]:
    """
    Extracts each object in the PSE file to an individual PDB file.
    Tries headless PyMOL first, falls back to standalone pickle parser.
    """
    out_dir = validate_output_dir(out_dir)
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
        safe_obj = _RE_SANITIZE_IDENT.sub('_', str(obj_name).strip()) or "obj"
        out_file = safe_join_path(out_dir, f"{safe_obj}.pdb")
        write_atoms_to_pdb(atoms, out_file)
        extracted_files[obj_name] = out_file

    return extracted_files


def extract_and_merge_objects(
    input_path: str,
    target_ids: Any,
    out_dir: str,
    prefix: str = "merged_donor",
    buffer_size: int = 65536,
    entities_atoms: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """
    Extracts and merges multiple designated objects or chains from a PDB or PSE file.
    Re-indexes residues contiguously onto a unified chain 'D' to prevent residue collisions,
    preserves exact 3D Cartesian coordinates (x,y,z) with zero distortion or translation,
    and writes both the merged PDB and 'donor_segments_manifest.json' with 64KB buffering.
    Permits passing pre-parsed entities_atoms to eliminate redundant disk re-reading.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    out_dir = validate_output_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Parse target_ids into clean token list with path traversal validation
    tokens: List[str] = []
    if isinstance(target_ids, str):
        for part in _RE_COMMA_SPLIT.split(target_ids.strip()):
            p = part.strip()
            if p:
                if ".." in p or "/" in p or "\\" in p:
                    raise ValueError(f"Path traversal detected in target ID: '{p}'")
                tokens.append(p)
    elif isinstance(target_ids, (list, tuple)):
        for item in target_ids:
            for part in _RE_COMMA_SPLIT.split(str(item).strip()):
                p = part.strip()
                if p:
                    if ".." in p or "/" in p or "\\" in p:
                        raise ValueError(f"Path traversal detected in target ID: '{p}'")
                    tokens.append(p)
    else:
        p = str(target_ids).strip()
        if ".." in p or "/" in p or "\\" in p:
            raise ValueError(f"Path traversal detected in target ID: '{p}'")
        tokens = [p]

    if not tokens:
        raise ValueError("No target IDs specified for extraction and merge.")

    # Collect available objects/chains and their raw atoms if not provided
    if entities_atoms is None:
        entities_atoms = {}
        ext = os.path.splitext(input_path)[1].lower()

        if ext == ".pse":
            parsed = parse_pse_standalone(input_path)
            for obj_name, obj_meta in parsed.get("objects", {}).items():
                atoms = obj_meta.get("_atoms", [])
                if atoms:
                    entities_atoms[obj_name] = atoms
        else:
            # Standard PDB file: read atoms grouped by chain in a single buffered pass
            with open(input_path, "r", encoding="utf-8", errors="ignore", buffering=buffer_size) as f:
                for line in f:
                    if line.startswith("ATOM  ") or line.startswith("HETATM"):
                        chain = line[21].strip() or "A"
                        atom_name = line[12:16].strip()
                        resn = line[17:20].strip()
                        try:
                            resi = int(line[22:26])
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                        except ValueError:
                            continue
                        b_factor = float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else 20.0
                        elem = (
                            line[76:78].strip()
                            if len(line) >= 78 and line[76:78].strip()
                            else (atom_name[0] if atom_name else "C")
                        )

                        atom_dict = {
                            "atom_name": atom_name,
                            "resn": resn,
                            "resi": resi,
                            "chain": chain,
                            "x": x,
                            "y": y,
                            "z": z,
                            "b_factor": b_factor,
                            "element": elem
                        }
                        if chain not in entities_atoms:
                            entities_atoms[chain] = []
                        entities_atoms[chain].append(atom_dict)

    available_keys = list(entities_atoms.keys())
    if not available_keys:
        raise ValueError(f"No structural entities or atom records found in: {input_path}")

    # Match tokens to entities
    matched_entities: List[str] = []
    for tok in tokens:
        matched_key = None
        # Exact match
        if tok in entities_atoms:
            matched_key = tok
        elif tok.isdigit():
            # 1-based index first
            idx = int(tok) - 1
            if 0 <= idx < len(available_keys):
                matched_key = available_keys[idx]
            elif 0 <= int(tok) < len(available_keys):
                matched_key = available_keys[int(tok)]
        if not matched_key:
            # Case-insensitive match
            for k in available_keys:
                if k.lower() == tok.lower():
                    matched_key = k
                    break
        if not matched_key:
            # Suffix / chain match (e.g. 'A' matching 'Chain_A' or 'obj_chain_A')
            for k in available_keys:
                if k.lower().endswith(f"_{tok.lower()}") or k.lower() == f"chain_{tok.lower()}":
                    matched_key = k
                    break
        if matched_key and matched_key not in matched_entities:
            matched_entities.append(matched_key)

    if not matched_entities:
        # Fallback to first available if none matched
        matched_entities = [available_keys[0]]

    # Process segments and atoms
    all_merged_atoms: List[Dict[str, Any]] = []
    manifest_segments: List[Dict[str, Any]] = []
    total_aa = 0
    new_resi_counter = 1

    for obj_idx, obj_name in enumerate(matched_entities, start=1):
        raw_atoms = entities_atoms[obj_name]
        # Group atoms by residue (preserving original order of residues)
        res_groups: Dict[int, List[Dict[str, Any]]] = {}
        res_names: Dict[int, str] = {}
        res_chains: Dict[int, str] = {}
        res_ca_coords: Dict[int, Tuple[float, float, float]] = {}

        for a in raw_atoms:
            r = a["resi"]
            if r not in res_groups:
                res_groups[r] = []
                res_names[r] = a["resn"]
                res_chains[r] = a["chain"]
            res_groups[r].append(a)
            if a["atom_name"].upper() == "CA" and r not in res_ca_coords:
                res_ca_coords[r] = (a["x"], a["y"], a["z"])

        sorted_res_seqs = sorted(res_groups.keys())
        if not sorted_res_seqs:
            continue

        # Partition this entity's residues into continuous segments:
        # Break on Delta resi > 1 OR Euclidean CA distance > 4.20 Angstrom (sq_dist > 17.64)
        segments_resis: List[List[int]] = []
        cur_seg: List[int] = [sorted_res_seqs[0]]

        for i in range(1, len(sorted_res_seqs)):
            prev_r = sorted_res_seqs[i - 1]
            curr_r = sorted_res_seqs[i]
            delta_resi = curr_r - prev_r

            sq_dist = 0.0
            has_ca = (prev_r in res_ca_coords and curr_r in res_ca_coords)
            if has_ca:
                c1 = res_ca_coords[prev_r]
                c2 = res_ca_coords[curr_r]
                dx, dy, dz = c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]
                sq_dist = dx * dx + dy * dy + dz * dz

            if delta_resi > 1 or (has_ca and sq_dist > 17.64):  # 4.20^2 = 17.64 avoids sqrt
                segments_resis.append(cur_seg)
                cur_seg = [curr_r]
            else:
                cur_seg.append(curr_r)

        if cur_seg:
            segments_resis.append(cur_seg)

        # Merge segments into unified chain 'D'
        for seg_idx, s_resis in enumerate(segments_resis, start=1):
            seg_start_new = new_resi_counter
            orig_start = s_resis[0]
            orig_end = s_resis[-1]
            seg_seq = "".join(aa_to_one_letter(res_names[r]) for r in s_resis)
            source_chain = res_chains[s_resis[0]]

            for r in s_resis:
                for a in res_groups[r]:
                    merged_atom = {
                        "atom_name": a["atom_name"],
                        "resn": a["resn"],
                        "resi": new_resi_counter,
                        "chain": "D",
                        "x": a["x"],
                        "y": a["y"],
                        "z": a["z"],
                        "b_factor": a.get("b_factor", 20.0),
                        "element": a.get("element", "C")
                    }
                    all_merged_atoms.append(merged_atom)
                new_resi_counter += 1

            seg_end_new = new_resi_counter - 1
            seg_len = len(s_resis)
            total_aa += seg_len

            manifest_segments.append({
                "segment_id": f"D{obj_idx}_S{seg_idx}",
                "source_object": obj_name,
                "source_chain": source_chain,
                "residue_range": [orig_start, orig_end],
                "new_residue_range": [seg_start_new, seg_end_new],
                "length": seg_len,
                "sequence": seg_seq
            })

    # Write merged PDB with 64KB buffering
    safe_prefix = _RE_SANITIZE_IDENT.sub('_', str(prefix).strip()) or "merged_donor"
    out_merged_pdb = safe_join_path(out_dir, f"{safe_prefix}.pdb")
    write_atoms_to_pdb(all_merged_atoms, out_merged_pdb, buffer_size=buffer_size)

    # Write manifest JSON with 64KB buffering
    manifest_data = {
        "donor_sources": matched_entities,
        "total_amino_acids": total_aa,
        "segments": manifest_segments
    }
    manifest_path = safe_join_path(out_dir, "donor_segments_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", buffering=buffer_size) as mf:
        json.dump(manifest_data, mf, indent=2)

    return {
        "status": "SUCCESS",
        "merged_pdb": out_merged_pdb,
        "manifest_path": manifest_path,
        "manifest": manifest_data,
        "donor_sources": matched_entities,
        "total_amino_acids": total_aa,
        "segments": manifest_segments,
        "entities_atoms": entities_atoms
    }


def extract_and_merge_donors(
    input_path: str,
    donor_ids: Any,
    out_merged_pdb: str,
    buffer_size: int = 65536
) -> Dict[str, Any]:
    """
    Alias/wrapper for extract_and_merge_objects conforming to MULTI_DONOR_CHIMERA_SPEC.md signature.
    """
    out_dir = validate_output_dir(os.path.dirname(os.path.realpath(out_merged_pdb)))
    prefix = os.path.splitext(os.path.basename(out_merged_pdb))[0]
    return extract_and_merge_objects(
        input_path=input_path,
        target_ids=donor_ids,
        out_dir=out_dir,
        prefix=prefix,
        buffer_size=buffer_size
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and extract chains/objects from PDB or PyMOL session (.pse) files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect PDB or PSE and output JSON summary")
    inspect_parser.add_argument("--input", "-i", required=True, help="Input PDB or PSE file")
    inspect_parser.add_argument("--out_json", "-o", help="Optional output JSON file")

    # Subcommand: extract
    extract_parser = subparsers.add_parser("extract", help="Extract individual chains/objects to PDBs")
    extract_parser.add_argument("--input", "-i", required=True, help="Input PDB or PSE file")
    extract_parser.add_argument("--out_dir", "-d", required=True, help="Directory to save extracted PDB files")
    extract_parser.add_argument(
        "--target_ids", "-t",
        help="Comma-separated IDs/names/indices to extract and merge into a single donor PDB"
    )
    extract_parser.add_argument(
        "--prefix", default="merged_donor",
        help="Prefix for merged output PDB [Default: merged_donor]"
    )

    args = parser.parse_args()

    if args.command == "inspect":
        info = inspect_file(args.input)
        json_output = json.dumps(info, indent=2)
        if args.out_json:
            out_json_path = os.path.realpath(os.path.abspath(args.out_json))
            drive, rest = os.path.splitdrive(out_json_path)
            if (not drive and out_json_path.rstrip("/\\") == "") or (drive and rest in ("/", "\\", "")) or out_json_path in ("/", "\\", "//", "\\\\"):
                raise ValueError(f"Dangerous output JSON path: {out_json_path}")
            out_json_dir = os.path.dirname(out_json_path)
            if out_json_dir:
                os.makedirs(out_json_dir, exist_ok=True)
            with open(out_json_path, "w", encoding="utf-8", buffering=65536) as f:
                f.write(json_output)
        print(json_output)

    elif args.command == "extract":
        out_dir = validate_output_dir(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        if args.target_ids:
            merge_res = extract_and_merge_objects(args.input, args.target_ids, out_dir, prefix=args.prefix)
            print(json.dumps(merge_res, indent=2))
        else:
            extracted = extract_file(args.input, out_dir)
            print(json.dumps({"status": "SUCCESS", "extracted_files": extracted}, indent=2))


if __name__ == "__main__":
    main()
