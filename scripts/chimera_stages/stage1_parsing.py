"""
scripts/chimera_stages/stage1_parsing.py
Stage 1: PDB Coordinate Parsing, Multi-Chain Manifest Parsing, Coordinate Caches, and Validation.

Handles:
- PDB coordinate and anchor extraction with 64KB buffering and session caching.
- Multi-chain and multi-PDB motif grammar parsing with auto-segmentation.
- Structural preparation and multi-chain entity extraction/merging.
- Strict security sanitization (directory traversal, root escapes, injection rejection).
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Resolve scripts directory for helper modules
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.pse_pdb_extractor import (
        extract_file,
        parse_pdb_file,
        extract_and_merge_objects,
        safe_join_path,
        validate_output_dir,
        write_atoms_to_pdb,
        SafeSessionUnpickler,
    )
except ImportError:
    from pse_pdb_extractor import (
        extract_file,
        parse_pdb_file,
        extract_and_merge_objects,
        safe_join_path,
        validate_output_dir,
        write_atoms_to_pdb,
        SafeSessionUnpickler,
    )

ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# Pre-compiled regular expressions for path validation, residue ranges, and linkers
_RE_SANITIZE_IDENT = re.compile(r"[^a-zA-Z0-9_\-]")
_RE_OUTPUTS_DEDUP_MID = re.compile(r"([/\\]outputs)+([/\\]outputs)", re.IGNORECASE)
_RE_RESIDUE_RANGE = re.compile(r"^[A-Za-z]?(\d+)(?:\s*-\s*[A-Za-z]?(\d+))?$")
_RE_NUM_OR_SPAN = re.compile(r"^(\d+)(?:-(\d+))?$")
_RE_HYPHEN_RANGE = re.compile(r"^[A-Za-z]?(\d+)\s*-\s*[A-Za-z]?(\d+)$")
_RE_LINKER_VAL = re.compile(r"^(auto|\d+(-\d+)?)$", re.IGNORECASE)
_RE_SHELL_INJECTION = re.compile(r"[;&|`><\n\r\$]")

# In-memory session cache for parsed PDB residue coordinates:
# Key: (real_path, mtime) -> {(chain, resi): np.ndarray}
_CA_COORDS_CACHE: Dict[Tuple[str, float], Dict[Tuple[str, int], np.ndarray]] = {}
_FALLBACK_COORDS_CACHE: Dict[Tuple[str, float], Dict[Tuple[str, int], np.ndarray]] = {}
_RESI_COORDS_CACHE: Dict[Tuple[str, float], Dict[int, np.ndarray]] = {}

# In-memory session cache for parsed residue anchor atoms:
# Key: (real_path, mtime) -> {(chain, resi, atom_name): np.ndarray}
_ANCHOR_COORDS_CACHE: Dict[Tuple[str, float], Dict[Tuple[str, int, str], np.ndarray]] = {}
_ANCHOR_FALLBACK_CACHE: Dict[Tuple[str, float], Dict[Tuple[str, int], np.ndarray]] = {}
_ANCHOR_RESI_CACHE: Dict[Tuple[str, float], Dict[int, Dict[str, np.ndarray]]] = {}
_ANCHOR_RESI_FALLBACK_CACHE: Dict[Tuple[str, float], Dict[int, np.ndarray]] = {}

# In-memory session cache for parsed reference seed CA coordinates:
# Key: (real_path, mtime) -> {seed_resi: np.ndarray}
_SEED_CA_CACHE: Dict[Tuple[str, float], Dict[int, np.ndarray]] = {}

# In-memory canonical path cache to avoid repeated os.path.realpath filesystem traversals
_REALPATH_CACHE: Dict[str, str] = {}


def _get_path_and_mtime(path: str) -> Optional[Tuple[str, float]]:
    """Resolves canonical realpath and mtime using a single cached stat call."""
    if not path:
        return None
    try:
        real_path = _REALPATH_CACHE.get(path)
        if real_path is None:
            real_path = os.path.realpath(path)
            _REALPATH_CACHE[path] = real_path
        st = os.stat(real_path)
        return real_path, st.st_mtime
    except (OSError, ValueError):
        return None


def clear_ca_coords_cache() -> None:
    """Clears the in-memory parsed coordinate session cache."""
    _CA_COORDS_CACHE.clear()
    _FALLBACK_COORDS_CACHE.clear()
    _RESI_COORDS_CACHE.clear()
    _ANCHOR_COORDS_CACHE.clear()
    _ANCHOR_FALLBACK_CACHE.clear()
    _ANCHOR_RESI_CACHE.clear()
    _ANCHOR_RESI_FALLBACK_CACHE.clear()
    _SEED_CA_CACHE.clear()
    _REALPATH_CACHE.clear()


@dataclass(frozen=True)
class MotifAtom:
    atom_name: str
    element: str
    coords: np.ndarray  # shape: (3,), dtype: float64
    occupancy: float = 1.0
    b_factor: float = 0.0


@dataclass
class MotifResidue:
    source_file: str
    source_chain: str
    orig_resi: int
    resn: str
    insertion_code: str = ""
    atoms: Dict[str, MotifAtom] = field(default_factory=dict)

    @property
    def ca_coords(self) -> Optional[np.ndarray]:
        if "CA" in self.atoms:
            return self.atoms["CA"].coords
        for name in ("N", "C", "CB"):
            if name in self.atoms:
                return self.atoms[name].coords
        return None


@dataclass
class MotifSegment:
    segment_id: str  # e.g. "D1_S1", "D2_S1", "D1", "D2"
    source_file: str
    source_chain: str
    residues: List[MotifResidue]

    @property
    def start_resi(self) -> int:
        return self.residues[0].orig_resi

    @property
    def end_resi(self) -> int:
        return self.residues[-1].orig_resi

    @property
    def length(self) -> int:
        return len(self.residues)

    @property
    def n_term_ca(self) -> Optional[np.ndarray]:
        return self.residues[0].ca_coords

    @property
    def c_term_ca(self) -> Optional[np.ndarray]:
        return self.residues[-1].ca_coords

    @property
    def n_term_n(self) -> Optional[np.ndarray]:
        if not self.residues:
            return None
        res = self.residues[0]
        if "N" in res.atoms:
            return res.atoms["N"].coords
        return res.ca_coords

    @property
    def c_term_c(self) -> Optional[np.ndarray]:
        if not self.residues:
            return None
        res = self.residues[-1]
        if "C" in res.atoms:
            return res.atoms["C"].coords
        return res.ca_coords


@dataclass
class MultiChainMotifManifest:
    source_files: List[str]
    segments: List[MotifSegment]
    total_residues: int
    provenance_map: Dict[int, Dict[str, Any]]  # seed_resi -> original metadata


def parse_multi_chain_motif_spec(
    motif_spec: str,
    default_source_file: str = "",
    default_chain: str = "A",
) -> List[Tuple[str, str, int]]:
    """
    Parses a multi-chain or multi-PDB motif specification into ordered (source_file, chain_id, resi) tuples.
    Grammar:
      <motif_spec>       ::= <motif_entry> ( "," <motif_entry> )*
      <motif_entry>      ::= [ <source_file> ":" ] <chain_id> ":" <residue_ranges>
      <source_file>      ::= [a-zA-Z0-9_\\-\\.\\/\\\\]+
      <chain_id>         ::= [a-zA-Z0-9]
      <residue_ranges>   ::= <range_item> ( "+" <range_item> | "," <range_item> )*
      <range_item>       ::= <integer> [ "-" <integer> ]
    """
    if not motif_spec or not str(motif_spec).strip():
        return []

    tokens = [t.strip() for t in str(motif_spec).split(",") if t.strip()]
    results: List[Tuple[str, str, int]] = []

    curr_file = default_source_file
    curr_chain = default_chain

    for token in tokens:
        parts = token.split(":")
        if len(parts) >= 3:
            range_str = parts[-1].strip()
            chain_cand = parts[-2].strip()
            file_cand = ":".join(parts[:-2]).strip()
            curr_file = file_cand
            curr_chain = chain_cand
        elif len(parts) == 2:
            first, second = parts[0].strip(), parts[1].strip()
            if len(first) <= 4:
                curr_chain = first
                range_str = second
            else:
                curr_file = first
                curr_chain = second
                range_str = ""
        else:
            range_str = token.strip()

        # Validate chain ID against directory traversal and shell injection
        if ".." in curr_chain or "/" in curr_chain or "\\" in curr_chain:
            raise ValueError(f"Path traversal detected in motif chain ID: '{curr_chain}'")
        if _RE_SHELL_INJECTION.search(curr_chain):
            raise ValueError(f"Shell injection metacharacter detected in motif chain ID: '{curr_chain}'")

        # Validate file path against root system directory paths and shell injection
        if curr_file:
            if _RE_SHELL_INJECTION.search(curr_file):
                raise ValueError(f"Shell injection metacharacter detected in motif file: '{curr_file}'")
            p_drive, p_rest = os.path.splitdrive(curr_file)
            if (
                (not p_drive and curr_file.rstrip("/\\") in ("", "/", "\\"))
                or (p_drive and p_rest in ("/", "\\", ""))
                or curr_file in ("/", "\\", "//", "\\\\")
            ):
                raise ValueError(f"Dangerous or root file specified in motif specification: '{curr_file}'")

        # Parse range items split by '+'
        for sub in range_str.split("+"):
            sub = sub.strip()
            if not sub:
                continue
            if "-" in sub:
                try:
                    s_str, e_str = sub.split("-", 1)
                    s_int = int(s_str.strip())
                    e_int = int(e_str.strip())
                    step = 1 if s_int <= e_int else -1
                    for r in range(s_int, e_int + step, step):
                        results.append((curr_file, curr_chain, r))
                except ValueError:
                    pass
            elif sub.isdigit():
                results.append((curr_file, curr_chain, int(sub)))

    return results


def build_multi_chain_manifest(
    motif_spec: str,
    default_source_file: str = "",
    default_chain: str = "A",
    file_atom_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> MultiChainMotifManifest:
    """
    Parses motif specification and partitions into continuous MotifSegment instances.
    Auto-segmentation triggers when:
      1. source_file changes
      2. source_chain changes
      3. Delta resi > 1
      4. CA-CA distance > 4.20 Å
    """
    ordered_items = parse_multi_chain_motif_spec(motif_spec, default_source_file, default_chain)
    if not ordered_items:
        raise ValueError(f"No valid residues found in motif specification: '{motif_spec}'")

    unique_files = []
    for f, _, _ in ordered_items:
        if f and f not in unique_files:
            unique_files.append(f)
    if not unique_files and default_source_file:
        unique_files.append(default_source_file)

    # Cache atom lines per file
    atoms_by_file: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    for fpath in unique_files:
        if not fpath:
            continue
        p_drive, p_rest = os.path.splitdrive(fpath)
        if (
            (not p_drive and fpath.rstrip("/\\") in ("", "/", "\\"))
            or (p_drive and p_rest in ("/", "\\", ""))
            or fpath in ("/", "\\", "//", "\\\\")
        ):
            raise ValueError(f"Dangerous or root file specified in multi-PDB motif: '{fpath}'")
        real_fpath = os.path.realpath(os.path.abspath(fpath))
        p_drive, p_rest = os.path.splitdrive(real_fpath)
        if (not p_drive and real_fpath.rstrip("/\\") in ("", "/")) or (p_drive and p_rest in ("/", "\\", "")):
            raise ValueError(f"Dangerous or root file specified in multi-PDB motif: '{fpath}'")
        if not os.path.isfile(real_fpath):
            continue
        res_atoms: Dict[Tuple[str, int], Dict[str, Any]] = {}
        with open(real_fpath, "r", encoding="utf-8", errors="ignore", buffering=65536) as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    try:
                        resi = int(line[22:26])
                    except ValueError:
                        continue
                    ch = line[21].strip() or "A"
                    atom_name = line[12:16].strip()
                    resn = line[17:20].strip()
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue
                    try:
                        occ = float(line[54:60])
                    except (ValueError, IndexError):
                        occ = 1.0
                    try:
                        b_fac = float(line[60:66])
                    except (ValueError, IndexError):
                        b_fac = 0.0
                    elem = line[76:78].strip() if len(line) >= 78 else atom_name[:1]

                    key = (ch, resi)
                    if key not in res_atoms:
                        res_atoms[key] = {"resn": resn, "atoms": {}}
                    res_atoms[key]["atoms"][atom_name] = MotifAtom(
                        atom_name=atom_name,
                        element=elem,
                        coords=np.array([x, y, z], dtype=np.float64),
                        occupancy=occ,
                        b_factor=b_fac,
                    )
        atoms_by_file[fpath] = res_atoms

    # Construct MotifResidue objects
    residues: List[MotifResidue] = []
    for fpath, ch, resi in ordered_items:
        file_data = atoms_by_file.get(fpath, {})
        key = (ch, resi)
        if key in file_data:
            resn = file_data[key]["resn"]
            atoms = file_data[key]["atoms"]
        else:
            resn = "GLY"
            atoms = {}
        residues.append(
            MotifResidue(
                source_file=fpath,
                source_chain=ch,
                orig_resi=resi,
                resn=resn,
                atoms=atoms,
            )
        )

    # Auto-segmentation
    segments: List[MotifSegment] = []
    curr_seg_res: List[MotifResidue] = []
    seg_idx = 1

    for i, res in enumerate(residues):
        if i == 0:
            curr_seg_res.append(res)
            continue

        prev_res = residues[i - 1]
        break_segment = False

        if res.source_file != prev_res.source_file:
            break_segment = True
        elif res.source_chain != prev_res.source_chain:
            break_segment = True
        elif (res.orig_resi - prev_res.orig_resi) != 1:
            break_segment = True
        else:
            c1 = prev_res.ca_coords
            c2 = res.ca_coords
            if c1 is not None and c2 is not None:
                dist = float(np.linalg.norm(c1 - c2))
                if dist > 4.20:
                    break_segment = True

        if break_segment:
            segments.append(
                MotifSegment(
                    segment_id=f"D{seg_idx}",
                    source_file=curr_seg_res[0].source_file,
                    source_chain=curr_seg_res[0].source_chain,
                    residues=curr_seg_res,
                )
            )
            seg_idx += 1
            curr_seg_res = [res]
        else:
            curr_seg_res.append(res)

    if curr_seg_res:
        segments.append(
            MotifSegment(
                segment_id=f"D{seg_idx}",
                source_file=curr_seg_res[0].source_file,
                source_chain=curr_seg_res[0].source_chain,
                residues=curr_seg_res,
            )
        )

    # Build provenance map (1-based sequential seed residue index)
    provenance_map: Dict[int, Dict[str, Any]] = {}
    seed_idx = 1
    for seg in segments:
        for res in seg.residues:
            provenance_map[seed_idx] = {
                "source_file": seg.source_file,
                "source_chain": seg.source_chain,
                "orig_resi": res.orig_resi,
                "resn": res.resn,
                "segment_id": seg.segment_id,
            }
            seed_idx += 1

    return MultiChainMotifManifest(
        source_files=unique_files,
        segments=segments,
        total_residues=len(residues),
        provenance_map=provenance_map,
    )


def get_residue_anchor_coords(
    pdb_path: str, chain: str, resi: int, atom_type: str = "CA", use_cache: bool = True
) -> Optional[np.ndarray]:
    """
    Returns 3D coords for atom_type ('C', 'N', 'CA') of specified residue in pdb_path.
    Falls back to 'CA' or first heavy atom.
    Caches parsed atom coordinates per (file, mtime) for O(1) terminal and junction anchor lookups.
    """
    path_info = _get_path_and_mtime(pdb_path)
    if path_info is None:
        return None
    real_path, mtime = path_info

    cache_key = (real_path, mtime)
    if not use_cache or cache_key not in _ANCHOR_COORDS_CACHE:
        atom_map: Dict[Tuple[str, int, str], np.ndarray] = {}
        fallback_map: Dict[Tuple[str, int], np.ndarray] = {}
        resi_atom_map: Dict[int, Dict[str, np.ndarray]] = {}
        resi_fb_map: Dict[int, np.ndarray] = {}
        with open(real_path, "r", encoding="utf-8", errors="ignore", buffering=65536) as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    try:
                        r = int(line[22:26])
                        ch = line[21].strip() or "A"
                        name = line[12:16].strip().upper()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue
                    pt = np.array([x, y, z], dtype=np.float64)
                    atom_key = (ch, r, name)
                    if atom_key not in atom_map:
                        atom_map[atom_key] = pt
                    fb_key = (ch, r)
                    if fb_key not in fallback_map:
                        fallback_map[fb_key] = pt
                    elif name == "CA":
                        fallback_map[fb_key] = pt

                    # Secondary O(1) indices for unchained queries
                    if r not in resi_atom_map:
                        resi_atom_map[r] = {}
                    if name not in resi_atom_map[r]:
                        resi_atom_map[r][name] = pt
                    if r not in resi_fb_map:
                        resi_fb_map[r] = pt
                    elif name == "CA":
                        resi_fb_map[r] = pt

        if use_cache:
            _ANCHOR_COORDS_CACHE[cache_key] = atom_map
            _ANCHOR_FALLBACK_CACHE[cache_key] = fallback_map
            _ANCHOR_RESI_CACHE[cache_key] = resi_atom_map
            _ANCHOR_RESI_FALLBACK_CACHE[cache_key] = resi_fb_map
    else:
        atom_map = _ANCHOR_COORDS_CACHE[cache_key]
        fallback_map = _ANCHOR_FALLBACK_CACHE[cache_key]
        resi_atom_map = _ANCHOR_RESI_CACHE.get(cache_key, {})
        resi_fb_map = _ANCHOR_RESI_FALLBACK_CACHE.get(cache_key, {})

    target_chain = chain.strip() if chain else ""
    target_resi = int(resi)
    target_atom = atom_type.strip().upper()

    if target_chain:
        k = (target_chain, target_resi, target_atom)
        if k in atom_map:
            return atom_map[k].copy()
        ca_k = (target_chain, target_resi, "CA")
        if ca_k in atom_map:
            return atom_map[ca_k].copy()
        fb_k = (target_chain, target_resi)
        if fb_k in fallback_map:
            return fallback_map[fb_k].copy()
        return None

    # O(1) fast lookup when target_chain is empty/omitted
    res_atoms = resi_atom_map.get(target_resi)
    if res_atoms is not None:
        if target_atom in res_atoms:
            return res_atoms[target_atom].copy()
        if "CA" in res_atoms:
            return res_atoms["CA"].copy()
    fb_pt = resi_fb_map.get(target_resi)
    if fb_pt is not None:
        return fb_pt.copy()
    return None


def get_residue_ca_coords(
    pdb_path: str,
    chain_id: Optional[str] = None,
    resi: int = 1,
    buffer_size: int = 65536,
    use_cache: bool = True,
) -> Optional[np.ndarray]:
    """
    Returns 3D coordinates [x, y, z] of the CA atom for given chain and residue number.
    Falls back to the geometric center (mean coordinates of all atoms) of the residue
    if the CA atom is missing. Returns None if residue is not found.
    Caches parsed coordinates per file session so multi-megabyte PDB files are not
    re-parsed on multiple anchor queries. Uses O(1) residue indexing when chain_id is None.
    """
    path_info = _get_path_and_mtime(pdb_path)
    if path_info is None:
        return None
    real_path, mtime = path_info

    cache_key = (real_path, mtime)
    if not use_cache or cache_key not in _CA_COORDS_CACHE:
        ca_map: Dict[Tuple[str, int], np.ndarray] = {}
        all_atom_accum: Dict[Tuple[str, int], List[float]] = {}
        resi_map: Dict[int, np.ndarray] = {}

        with open(real_path, "r", encoding="utf-8", errors="ignore", buffering=buffer_size) as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    ch = line[21].strip() or "A"
                    try:
                        r = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue

                    key = (ch, r)
                    atom_name = line[12:16].strip().upper()
                    if atom_name == "CA" and key not in ca_map:
                        coord_arr = np.array([x, y, z], dtype=float)
                        ca_map[key] = coord_arr
                        if r not in resi_map:
                            resi_map[r] = coord_arr

                    acc = all_atom_accum.get(key)
                    if acc is None:
                        all_atom_accum[key] = [x, y, z, 1.0]
                    else:
                        acc[0] += x
                        acc[1] += y
                        acc[2] += z
                        acc[3] += 1.0

        fallback_map: Dict[Tuple[str, int], np.ndarray] = {}
        for key, acc in all_atom_accum.items():
            if key not in ca_map and acc[3] > 0:
                fallback_coord = np.array([acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3]], dtype=float)
                fallback_map[key] = fallback_coord
                if key[1] not in resi_map:
                    resi_map[key[1]] = fallback_coord

        if use_cache:
            _CA_COORDS_CACHE[cache_key] = ca_map
            _FALLBACK_COORDS_CACHE[cache_key] = fallback_map
            _RESI_COORDS_CACHE[cache_key] = resi_map
    else:
        ca_map = _CA_COORDS_CACHE[cache_key]
        fallback_map = _FALLBACK_COORDS_CACHE[cache_key]
        resi_map = _RESI_COORDS_CACHE.get(cache_key, {})

    target_resi = int(resi)
    clean_chain = str(chain_id).strip() if chain_id is not None else None

    if clean_chain is not None:
        coord = ca_map.get((clean_chain, target_resi))
        if coord is not None:
            return coord.copy()
        fallback = fallback_map.get((clean_chain, target_resi))
        if fallback is not None:
            return fallback.copy()
        return None
    else:
        coord = resi_map.get(target_resi)
        if coord is not None:
            return coord.copy()
        return None


def prepare_input_structures(
    input_file: Optional[str] = None,
    donor_pdb: Optional[str] = None,
    vessel_pdb: Optional[str] = None,
    donor_id: Any = "A",
    vessel_id: Any = "B",
    out_dir: str = "outputs/chimera_scaffold",
    is_multichain: bool = False,
) -> Dict[str, Any]:
    """
    Extracts, merges, or validates donor and vessel PDB files into the 0_prep directory.
    Returns metadata dictionary with working files, chain IDs, and object summaries.
    """
    out_dir = validate_output_dir(out_dir)
    work_prep = safe_join_path(out_dir, "0_prep")
    os.makedirs(work_prep, exist_ok=True)

    for id_label, id_val in [("donor_id", donor_id), ("vessel_id", vessel_id)]:
        if isinstance(id_val, str):
            if ".." in id_val or "/" in id_val or "\\" in id_val:
                raise ValueError(f"Path traversal detected in {id_label}: '{id_val}'")
            if _RE_SHELL_INJECTION.search(id_val):
                raise ValueError(f"Shell injection metacharacter detected in {id_label}: '{id_val}'")
        elif isinstance(id_val, (list, tuple)):
            for item in id_val:
                if ".." in str(item) or "/" in str(item) or "\\" in str(item):
                    raise ValueError(f"Path traversal detected in {id_label}: '{item}'")
                if _RE_SHELL_INJECTION.search(str(item)):
                    raise ValueError(f"Shell injection metacharacter detected in {id_label}: '{item}'")

    for p_label, p_val in [("input_file", input_file), ("donor_pdb", donor_pdb), ("vessel_pdb", vessel_pdb)]:
        if p_val:
            if _RE_SHELL_INJECTION.search(str(p_val)):
                raise ValueError(f"Shell injection metacharacter detected in {p_label}: '{p_val}'")
            p_drive, p_rest = os.path.splitdrive(str(p_val))
            if (
                (not p_drive and str(p_val).rstrip("/\\") in ("", "/", "\\"))
                or (p_drive and p_rest in ("/", "\\", ""))
                or str(p_val) in ("/", "\\", "//", "\\\\")
            ):
                raise ValueError(f"Dangerous or root file specified for {p_label}: '{p_val}'")

    is_multi_donor = False
    if isinstance(donor_id, (list, tuple)) and len(donor_id) > 1:
        is_multi_donor = True
    elif isinstance(donor_id, str) and "," in donor_id:
        is_multi_donor = True

    working_donor_pdb = ""
    working_vessel_pdb = ""
    donor_chain_id = "A"
    vessel_chain_id = "B"

    if is_multi_donor:
        source_input = input_file or donor_pdb
        if not source_input or not os.path.exists(source_input):
            raise FileNotFoundError(f"Input file for multi-donor extraction not found: {source_input}")
        merge_res = extract_and_merge_objects(source_input, donor_id, work_prep, prefix="merged_donor")
        working_donor_pdb = merge_res["merged_pdb"]
        donor_chain_id = "D"

        if input_file:
            entities_atoms = merge_res.get("entities_atoms")
            if entities_atoms and os.path.abspath(source_input) == os.path.abspath(input_file):
                extracted = {}

                for k, atoms in entities_atoms.items():
                    safe_k = _RE_SANITIZE_IDENT.sub("_", str(k).strip()) or "chain"
                    out_path = safe_join_path(work_prep, f"{safe_k}.pdb")
                    if not os.path.exists(out_path):
                        write_atoms_to_pdb(atoms, out_path, buffer_size=65536)
                    extracted[k] = out_path
            else:
                extracted = extract_file(input_file, work_prep)
            keys_list = list(extracted.keys())

            def resolve_entity(val: str, fallback_idx: int = 1) -> str:
                s = str(val).strip()
                if s.isdigit():
                    idx = int(s) - 1
                    if 0 <= idx < len(keys_list):
                        return extracted[keys_list[idx]]
                if s in extracted:
                    return extracted[s]
                matched = [extracted[k] for k in keys_list if k.lower() == s.lower()]
                if matched:
                    return matched[0]
                if 0 <= fallback_idx < len(keys_list):
                    return extracted[keys_list[fallback_idx]]
                return list(extracted.values())[0] if extracted else ""

            working_vessel_pdb = resolve_entity(vessel_id, 1 if len(keys_list) > 1 else 0)
        else:
            if not vessel_pdb or not os.path.isfile(vessel_pdb):
                raise FileNotFoundError(f"Vessel PDB not found: {vessel_pdb}")
            working_vessel_pdb = vessel_pdb

    elif input_file:
        extracted = extract_file(input_file, work_prep)
        keys_list = list(extracted.keys())

        def resolve_entity(val: str, fallback_idx: int = 0) -> str:
            s = str(val).strip()
            if s.isdigit():
                idx = int(s) - 1
                if 0 <= idx < len(keys_list):
                    return extracted[keys_list[idx]]
            if s in extracted:
                return extracted[s]
            matched = [extracted[k] for k in keys_list if k.lower() == s.lower()]
            if matched:
                return matched[0]
            if 0 <= fallback_idx < len(keys_list):
                return extracted[keys_list[fallback_idx]]
            return list(extracted.values())[0] if extracted else ""

        working_donor_pdb = resolve_entity(donor_id, 0)
        working_vessel_pdb = resolve_entity(vessel_id, 1 if len(keys_list) > 1 else 0)

    else:
        if not donor_pdb or not os.path.isfile(donor_pdb):
            raise FileNotFoundError(f"Donor PDB not found: {donor_pdb}")
        if not vessel_pdb or not os.path.isfile(vessel_pdb):
            raise FileNotFoundError(f"Vessel PDB not found: {vessel_pdb}")
        working_donor_pdb = donor_pdb
        working_vessel_pdb = vessel_pdb

    donor_parsed = parse_pdb_file(working_donor_pdb)
    if os.path.abspath(working_vessel_pdb) == os.path.abspath(working_donor_pdb):
        vessel_parsed = donor_parsed
    else:
        vessel_parsed = parse_pdb_file(working_vessel_pdb)

    d_objs = list(donor_parsed["objects"].values())
    v_objs = list(vessel_parsed["objects"].values())

    if not d_objs:
        raise ValueError(f"No valid atom/residue records in donor structure: {working_donor_pdb}")
    if not v_objs:
        raise ValueError(f"No valid atom/residue records in vessel structure: {working_vessel_pdb}")

    if len(d_objs) > 1 and not is_multichain:
        merge_res = extract_and_merge_objects(
            working_donor_pdb, [obj["id"] for obj in d_objs], work_prep, prefix="merged_multichain_donor"
        )
        working_donor_pdb = merge_res["merged_pdb"]
        donor_parsed = parse_pdb_file(working_donor_pdb)
        d_objs = list(donor_parsed["objects"].values())

    donor_metadata = d_objs[0]
    vessel_metadata = v_objs[0]
    if not is_multi_donor:
        donor_chain_id = donor_metadata["id"]
    vessel_chain_id = vessel_metadata["id"]

    return {
        "working_donor_pdb": working_donor_pdb,
        "working_vessel_pdb": working_vessel_pdb,
        "donor_metadata": donor_metadata,
        "vessel_metadata": vessel_metadata,
        "donor_chain_id": donor_chain_id,
        "vessel_chain_id": vessel_chain_id,
    }
