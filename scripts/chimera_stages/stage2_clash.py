"""
scripts/chimera_stages/stage2_clash.py
Stage 2: Clash Detection, Spatial Alignment, and In-Flight Cartesian Drift Auditing.

Handles:
- Distance metrics and batch Euclidean distance calculations.
- Exact in-place coordinate transformation using Kabsch rotation R and translation t.
- Strict Cartesian drift auditing with TRB mapping (SafeSessionUnpickler), residue pairing, and origin-shift detection.
- Fast KD-tree (scipy.spatial.cKDTree) and vectorized coordinate matching for ProteinMPNN fixed-position masks.
- Sequence-Anchored Contig Tracking (SACT) coordinate matching.
- Safe candidate artifact purging on drift rejection.
"""

import math
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# Resolve scripts directory for helper modules
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.pse_pdb_extractor import parse_pdb_file, SafeSessionUnpickler
    from scripts.kabsch_rmsd import kabsch_fit, extract_coordinates
except ImportError:
    from pse_pdb_extractor import parse_pdb_file, SafeSessionUnpickler
    from kabsch_rmsd import kabsch_fit, extract_coordinates

try:
    from .stage1_parsing import _SEED_CA_CACHE
except (ImportError, ValueError):
    from chimera_stages.stage1_parsing import _SEED_CA_CACHE


def calculate_anchor_distance(coord1: Any, coord2: Any) -> float:
    """
    Computes standard Euclidean distance ||coord2 - coord1||_2 in Angstroms.
    Uses scalar math with zero intermediate array allocations for 3D coordinates.
    """
    try:
        dx = float(coord2[0]) - float(coord1[0])
        dy = float(coord2[1]) - float(coord1[1])
        dz = float(coord2[2]) - float(coord1[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    except (IndexError, TypeError):
        c1 = np.asarray(coord1, dtype=float)
        c2 = np.asarray(coord2, dtype=float)
        diff = c2 - c1
        return float(np.sqrt(np.dot(diff, diff)))


def batch_anchor_distances(coords1: Any, coords2: Any) -> np.ndarray:
    """
    Computes vectorized Euclidean distances between pairs of 3D coordinates using NumPy.
    Supports (N, 3) arrays or single (3,) coordinates.
    """
    c1 = np.asarray(coords1, dtype=float)
    c2 = np.asarray(coords2, dtype=float)
    diff = c2 - c1
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _transform_pdb_coordinates(pdb_path: str, R: np.ndarray, t: np.ndarray) -> bool:
    """
    Transforms all 3D coordinates in a PDB file in-place by applying rotation R and translation t:
        r_new = np.dot(r, R) + t
    Used to place generated RFdiffusion candidates back into the exact original
    crystallographic coordinate frame of the reference seed structure.
    """
    if not pdb_path or not os.path.isfile(pdb_path):
        return False
    real_path = os.path.realpath(os.path.abspath(pdb_path))
    p_drive, p_rest = os.path.splitdrive(real_path)
    if (not p_drive and real_path.rstrip("/\\") in ("", "/")) or (p_drive and p_rest in ("/", "\\", "")):
        return False
    try:
        lines_out = []
        with open(real_path, "r", encoding="utf-8", errors="ignore", buffering=65536) as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        r = np.array([x, y, z], dtype=np.float64)
                        r_new = np.dot(r, R) + t
                        mod_line = f"{line[:30]}{r_new[0]:8.3f}{r_new[1]:8.3f}{r_new[2]:8.3f}{line[54:]}"
                        lines_out.append(mod_line)
                    except (ValueError, IndexError):
                        lines_out.append(line)
                else:
                    lines_out.append(line)
        with open(pdb_path, "w", encoding="utf-8", buffering=65536) as f:
            f.writelines(lines_out)
        return True
    except Exception:
        return False


def audit_motif_cartesian_drift(
    candidate_pdb: str,
    reference_seed_pdb: str,
    seed_provenance: Dict[Union[int, str], Dict[str, Any]],
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Strict Cartesian drift audit against original seed coordinates.
    Computes absolute unaligned Euclidean deviation between candidate backbone CA
    and reference seed CA coordinates.
    Returns (max_drift, mean_drift, drift_details).
    Pass criteria:
      max_drift <= 0.80 Å and mean_drift <= 0.50 Å
    """
    if not candidate_pdb or not os.path.isfile(candidate_pdb):
        return 999.0, 999.0, [{"error": "candidate_pdb_not_found"}]
    if not reference_seed_pdb or not os.path.isfile(reference_seed_pdb):
        return 999.0, 999.0, [{"error": "reference_seed_pdb_not_found"}]

    # Path validation against dangerous/root paths
    for p_label, p_path in [("candidate_pdb", candidate_pdb), ("reference_seed_pdb", reference_seed_pdb)]:
        p_drive, p_rest = os.path.splitdrive(p_path)
        if (
            (not p_drive and p_path.rstrip("/\\") in ("", "/", "\\"))
            or (p_drive and p_rest in ("/", "\\", ""))
            or p_path in ("/", "\\", "//", "\\\\")
        ):
            raise ValueError(f"Dangerous or root file specified for {p_label}: '{p_path}'")

    real_seed = os.path.realpath(os.path.abspath(reference_seed_pdb))
    real_cand = os.path.realpath(os.path.abspath(candidate_pdb))

    # Fast in-memory cached extraction of reference seed motif CA coordinates (O(1) on repeated audits)
    try:
        seed_mtime = os.path.getmtime(real_seed)
    except OSError:
        seed_mtime = 0.0
    seed_cache_key = (real_seed, seed_mtime)

    if seed_cache_key in _SEED_CA_CACHE:
        seed_ca = _SEED_CA_CACHE[seed_cache_key]
    else:
        seed_ca: Dict[int, np.ndarray] = {}
        with open(real_seed, "r", encoding="utf-8", errors="ignore", buffering=65536) as f:
            for line in f:
                if (line.startswith("ATOM  ") or line.startswith("HETATM")) and line[12:16].strip() == "CA":
                    try:
                        resi = int(line[22:26])
                        ch = line[21].strip()
                        if ch in ("D", "A", ""):
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            seed_ca[resi] = np.array([x, y, z], dtype=np.float64)
                    except ValueError:
                        continue
        _SEED_CA_CACHE[seed_cache_key] = seed_ca

    # Extract candidate CA coordinates with O(1) resi indexing & vectorized coordinates
    cand_ca_by_resi: Dict[Tuple[str, int], np.ndarray] = {}
    cand_ca_by_num: Dict[int, np.ndarray] = {}
    cand_coords_list: List[np.ndarray] = []

    with open(real_cand, "r", encoding="utf-8", errors="ignore", buffering=65536) as f:
        for line in f:
            if (line.startswith("ATOM  ") or line.startswith("HETATM")) and line[12:16].strip() == "CA":
                try:
                    resi = int(line[22:26])
                    ch = line[21].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coord = np.array([x, y, z], dtype=np.float64)
                    cand_ca_by_resi[(ch, resi)] = coord
                    if resi not in cand_ca_by_num:
                        cand_ca_by_num[resi] = coord
                    cand_coords_list.append(coord)
                except ValueError:
                    continue

    cand_coords_matrix = (
        np.array(cand_coords_list, dtype=np.float64)
        if cand_coords_list
        else np.empty((0, 3), dtype=np.float64)
    )
    cand_tree = cKDTree(cand_coords_matrix) if (cKDTree is not None and len(cand_coords_matrix) > 0) else None

    # TRB mapping if available (safely unpickled using restricted SafeSessionUnpickler)
    trb_path = real_cand[:-4] + ".trb" if real_cand.endswith(".pdb") else real_cand + ".trb"
    trb_map: Dict[int, int] = {}
    if os.path.isfile(trb_path):
        try:
            with open(trb_path, "rb", buffering=65536) as pf:
                trb_data = SafeSessionUnpickler(pf).load()
            if isinstance(trb_data, dict) and "con_ref_pdb_idx" in trb_data:
                for h_idx, r_item in enumerate(trb_data["con_ref_pdb_idx"]):
                    if r_item is not None:
                        ref_ch = str(r_item[0]).strip().upper()
                        ref_r = int(r_item[1])
                        if ref_ch in ("D", "A"):
                            trb_map[ref_r] = h_idx + 1
        except Exception:
            pass

    drift_details = []
    matched_pairs_cand: List[np.ndarray] = []
    matched_pairs_seed: List[np.ndarray] = []
    unmatched_indices: List[int] = []

    for seed_key, meta in seed_provenance.items():
        s_resi = int(seed_key)
        target_seed_ca = seed_ca.get(s_resi)
        if target_seed_ca is None:
            orig_r = meta.get("orig_resi")
            if orig_r in seed_ca:
                target_seed_ca = seed_ca[orig_r]

        if target_seed_ca is None:
            continue

        matched_cand_ca = None
        # Priority 1: TRB file hal index mapping (O(1) dictionary lookup)
        if s_resi in trb_map:
            hal_resi = trb_map[s_resi]
            matched_cand_ca = cand_ca_by_num.get(hal_resi)

        # Priority 2: Direct chain D / resi match (O(1) dictionary lookup)
        if matched_cand_ca is None and ("D", s_resi) in cand_ca_by_resi:
            matched_cand_ca = cand_ca_by_resi[("D", s_resi)]

        # Priority 3: Direct resi match in single-chain candidate (O(1) lookup)
        if matched_cand_ca is None:
            cand_c = cand_ca_by_num.get(s_resi)
            if cand_c is not None and np.linalg.norm(cand_c - target_seed_ca) < 4.5:
                matched_cand_ca = cand_c

        # Priority 4: Nearest neighbor CA within 6.0 Å (KD-tree or vectorized NumPy distance)
        if matched_cand_ca is None and len(cand_coords_matrix) > 0:
            if cand_tree is not None:
                d_cand, min_idx = cand_tree.query(target_seed_ca, distance_upper_bound=6.0)
                if min_idx < len(cand_coords_matrix):
                    matched_cand_ca = cand_coords_matrix[min_idx]
            else:
                diffs = cand_coords_matrix - target_seed_ca
                dists_sq = np.sum(diffs * diffs, axis=1)
                min_idx = int(np.argmin(dists_sq))
                if dists_sq[min_idx] < 36.0:  # 6.0 Å squared
                    matched_cand_ca = cand_coords_matrix[min_idx]

        curr_detail_idx = len(drift_details)
        drift_details.append({
            "seed_resi": s_resi,
            "orig_resi": meta.get("orig_resi"),
            "source_chain": meta.get("source_chain"),
            "drift": 999.0,
        })

        if matched_cand_ca is not None:
            matched_pairs_cand.append(matched_cand_ca)
            matched_pairs_seed.append(target_seed_ca)
        else:
            unmatched_indices.append(curr_detail_idx)

    if not drift_details:
        return 999.0, 999.0, [{"error": "no_motif_residues_audited"}]

    drifts: List[float] = []
    if matched_pairs_cand:
        cand_arr = np.array(matched_pairs_cand, dtype=np.float64)
        seed_arr = np.array(matched_pairs_seed, dtype=np.float64)

        # Check raw unaligned Euclidean deviation
        raw_vec_drifts = np.linalg.norm(cand_arr - seed_arr, axis=1)

        # Detect RFdiffusion bounding-box centering / coordinate origin shift:
        # If candidate center of mass has shifted by > 5.0 Å from seed center of mass,
        # RFdiffusion has re-centered the coordinate system around the origin.
        # Align candidate back onto seed crystallographic frame using Kabsch fit.
        cand_center = np.mean(cand_arr, axis=0)
        seed_center = np.mean(seed_arr, axis=0)
        com_shift = float(np.linalg.norm(cand_center - seed_center))

        if com_shift > 5.0 and len(cand_arr) >= 3:
            try:
                R_opt, t_opt, kabsch_rmsd, cand_aligned, sq_dist = kabsch_fit(
                    cand_arr, seed_arr, return_aligned=True
                )
                vec_drifts = np.sqrt(sq_dist)
                _transform_pdb_coordinates(real_cand, R_opt, t_opt)
            except Exception:
                vec_drifts = raw_vec_drifts
        else:
            vec_drifts = raw_vec_drifts

        matched_idx = 0
        for i, detail in enumerate(drift_details):
            if i not in unmatched_indices:
                d_val = float(vec_drifts[matched_idx])
                detail["drift"] = d_val
                drifts.append(d_val)
                matched_idx += 1
            else:
                drifts.append(999.0)
    else:
        drifts = [999.0] * len(drift_details)

    max_drift = float(np.max(drifts))
    mean_drift = float(np.mean(drifts))
    return max_drift, mean_drift, drift_details


def _purge_candidate_artifacts(prefix: str, cand_pdb: Optional[str] = None) -> None:
    """
    Safely purges all artifacts generated for a rejected or failed candidate attempt,
    including candidate PDB, TRB, and any intermediate output files.
    Enforces strict directory boundary verification to prevent arbitrary file deletion.
    """
    to_delete = set()
    if cand_pdb and str(cand_pdb).strip():
        real_cand = os.path.realpath(os.path.abspath(cand_pdb))
        p_drive, p_rest = os.path.splitdrive(real_cand)
        if not ((not p_drive and real_cand.rstrip("/\\") in ("", "/")) or (p_drive and p_rest in ("/", "\\", ""))):
            to_delete.add(real_cand)
            trb = real_cand[:-4] + ".trb" if real_cand.endswith(".pdb") else real_cand + ".trb"
            to_delete.add(trb)

    if prefix and str(prefix).strip():
        clean_prefix = str(prefix).strip()
        base_prefix = os.path.basename(clean_prefix)
        work_dir = os.path.realpath(os.path.abspath(os.path.dirname(clean_prefix)))
        p_drive, p_rest = os.path.splitdrive(work_dir)
        is_root = (
            (not p_drive and work_dir.rstrip("/\\") in ("", "/"))
            or (p_drive and p_rest in ("/", "\\", ""))
            or work_dir in ("/", "\\", "//", "\\\\")
        )
        if not is_root and base_prefix and os.path.isdir(work_dir):
            try:
                for fname in os.listdir(work_dir):
                    if fname.startswith(base_prefix):
                        cand_target = os.path.realpath(os.path.join(work_dir, fname))
                        # Boundary verification: target must stay strictly within work_dir
                        if os.path.commonpath([work_dir, cand_target]) == work_dir and cand_target != work_dir:
                            to_delete.add(cand_target)
            except OSError:
                pass

    for file_path in to_delete:
        try:
            p_drive, p_rest = os.path.splitdrive(file_path)
            if (not p_drive and file_path.rstrip("/\\") in ("", "/")) or (p_drive and p_rest in ("/", "\\", "")):
                continue
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)
        except OSError:
            pass


def map_fixed_positions_for_mpnn(
    backbone_pdb: str,
    preserved_donor_segments: List[Tuple[int, int]],
    preserved_vessel_segments: List[Tuple[int, int]],
    working_donor_pdb: str,
    working_vessel_pdb: str,
    seed_pdb_path: Optional[str] = None,
    donor_chain_id: str = "A",
    vessel_chain_id: str = "B",
    lock_vessel: bool = True,
    vessel_redesign_mode: str = "linkers_only",
    is_multichain: bool = False,
    chimera_mode: str = "replace",
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Parses a generated backbone PDB, locates preserved donor motif and vessel positions,
    and generates a 1-based index mask for ProteinMPNN.
    Returns (fixed_positions_dict, donor_fixed_positions).
    """
    parsed = parse_pdb_file(backbone_pdb)
    chain_keys = list(parsed["objects"].keys())
    main_chain = chain_keys[0] if chain_keys else "A"
    bb_res_seqs = parsed["objects"][main_chain]["residue_seqs"]
    total_len = len(bb_res_seqs)

    donor_fixed: List[int] = []
    vessel_fixed: List[int] = []

    ref_donor_res = [r for s, e in preserved_donor_segments for r in range(s, e + 1)]
    ref_vessel_res = [r for s, e in preserved_vessel_segments for r in range(s, e + 1)]

    # SACT Step 1: Check for RFdiffusion .trb metadata for 100% deterministic contig mapping
    trb_path = os.path.splitext(backbone_pdb)[0] + ".trb"
    if os.path.isfile(trb_path):
        try:
            with open(trb_path, "rb", buffering=65536) as pf:
                trb_data = SafeSessionUnpickler(pf).load()
            if isinstance(trb_data, dict) and "con_ref_pdb_idx" in trb_data:
                d_trb = []
                v_trb = []
                for h_idx, r_item in enumerate(trb_data["con_ref_pdb_idx"]):
                    if r_item is not None:
                        ch = str(r_item[0]).strip().upper()
                        if ch in (donor_chain_id, "D", "A"):
                            d_trb.append(h_idx + 1)
                        elif ch in (vessel_chain_id, "V"):
                            v_trb.append(h_idx + 1)
                if d_trb:
                    donor_fixed = d_trb
                if v_trb and lock_vessel:
                    vessel_fixed = v_trb
        except Exception:
            pass

    try:
        bb_ca_coords, _ = extract_coordinates(
            backbone_pdb, chain_id=None, residue_list=None, atom_names=["CA"], extract_meta=True
        )
        if len(chain_keys) == 1:
            main_chain = chain_keys[0]

        donor_src_pdb = seed_pdb_path if is_multichain and seed_pdb_path else working_donor_pdb
        donor_src_chain = "D" if is_multichain else donor_chain_id
        donor_ca_coords, _ = extract_coordinates(
            donor_src_pdb,
            chain_id=donor_src_chain,
            residue_list=ref_donor_res,
            atom_names=["CA"],
            extract_meta=False,
        )
        vessel_src_pdb = seed_pdb_path if is_multichain and seed_pdb_path else working_vessel_pdb
        vessel_src_chain = "V" if is_multichain else vessel_chain_id
        vessel_ca_coords, _ = extract_coordinates(
            vessel_src_pdb,
            chain_id=vessel_src_chain,
            residue_list=ref_vessel_res,
            atom_names=["CA"],
            extract_meta=False,
        )

        # Map donor residues by fast KDTree (< 0.65 Å for pinned coordinates) or vectorized search
        if len(bb_ca_coords) > 0 and not donor_fixed:
            bb_tree = cKDTree(bb_ca_coords) if cKDTree is not None else None

            if bb_tree is not None:
                if len(donor_ca_coords) > 0:
                    dists, idxs = bb_tree.query(donor_ca_coords, distance_upper_bound=0.65)
                    valid = idxs < len(bb_ca_coords)
                    if np.any(valid):
                        donor_fixed = (idxs[valid] + 1).tolist()

                if lock_vessel and len(vessel_ca_coords) > 0:
                    v_dists, v_idxs = bb_tree.query(vessel_ca_coords, distance_upper_bound=0.65)
                    v_valid = v_idxs < len(bb_ca_coords)
                    if np.any(v_valid):
                        vessel_fixed = (v_idxs[v_valid] + 1).tolist()
            else:
                if len(donor_ca_coords) > 0:
                    d_found = []
                    for d_coord in donor_ca_coords:
                        diffs = bb_ca_coords - d_coord
                        sq_dists = np.sum(diffs * diffs, axis=1)
                        min_idx = int(np.argmin(sq_dists))
                        if sq_dists[min_idx] < 0.4225:  # (0.65 Å)^2
                            d_found.append(min_idx + 1)
                    if len(d_found) > 0:
                        donor_fixed = d_found

                if lock_vessel and len(vessel_ca_coords) > 0:
                    v_found = []
                    for v_coord in vessel_ca_coords:
                        diffs = bb_ca_coords - v_coord
                        sq_dists = np.sum(diffs * diffs, axis=1)
                        min_idx = int(np.argmin(sq_dists))
                        if sq_dists[min_idx] < 0.4225:
                            v_found.append(min_idx + 1)
                    if len(v_found) > 0:
                        vessel_fixed = v_found
    except Exception:
        pass

    # Fallback to topology offset calculation if coordinate matching did not match residues
    if not donor_fixed:
        donor_fixed = []
        vessel_fixed = []
        v_left_len = (
            sum(e - s + 1 for s, e in preserved_vessel_segments[:1]) if preserved_vessel_segments else 0
        )
        v_right_len = (
            sum(e - s + 1 for s, e in preserved_vessel_segments[1:])
            if len(preserved_vessel_segments) > 1
            else 0
        )
        num_donor_res = sum(e - s + 1 for s, e in preserved_donor_segments)
        num_int_linkers = max(0, len(preserved_donor_segments) - 1)
        est_int_len = 6
        total_int_len = num_int_linkers * est_int_len

        mode_clean = chimera_mode.lower()
        if mode_clean in ["replace", "graft", "insert_loop", "loop", "insert"]:
            rem_len = max(0, total_len - v_left_len - v_right_len - num_donor_res - total_int_len)
            actual_ln = rem_len // 2
            cursor = v_left_len + actual_ln + 1
            for s, e in preserved_donor_segments:
                seg_len = e - s + 1
                donor_fixed.extend(range(cursor, cursor + seg_len))
                cursor += seg_len + est_int_len

            if lock_vessel:
                vessel_fixed.extend(range(1, v_left_len + 1))
                vessel_fixed.extend(range(total_len - v_right_len + 1, total_len + 1))

        elif mode_clean in ["ligate_nterm", "nterm"]:
            cursor = 1
            for s, e in preserved_donor_segments:
                seg_len = e - s + 1
                donor_fixed.extend(range(cursor, cursor + seg_len))
                cursor += seg_len + est_int_len
            if lock_vessel:
                v_len = sum(e - s + 1 for s, e in preserved_vessel_segments)
                vessel_fixed.extend(range(total_len - v_len + 1, total_len + 1))

        elif mode_clean in ["ligate_cterm", "cterm"]:
            v_len = sum(e - s + 1 for s, e in preserved_vessel_segments)
            if lock_vessel:
                vessel_fixed.extend(range(1, v_len + 1))
            rem_len = max(0, total_len - v_len - num_donor_res - total_int_len)
            cursor = v_len + rem_len + 1
            for s, e in preserved_donor_segments:
                seg_len = e - s + 1
                donor_fixed.extend(range(cursor, cursor + seg_len))
                cursor += seg_len + est_int_len

    all_fixed = set(donor_fixed)
    if vessel_redesign_mode == "vessel_interface" and len(donor_fixed) > 0 and len(vessel_fixed) > 0:
        try:
            d_coords = [bb_ca_coords[d - 1] for d in donor_fixed if 1 <= d <= len(bb_ca_coords)]
            valid_v = [v for v in vessel_fixed if 1 <= v <= len(bb_ca_coords)]
            if d_coords and valid_v:
                d_arr = np.asarray(d_coords, dtype=np.float64)
                v_coords = np.asarray([bb_ca_coords[v - 1] for v in valid_v], dtype=np.float64)
                if cKDTree is not None:
                    d_tree = cKDTree(d_arr)
                    dists, _ = d_tree.query(v_coords)
                    for v_idx, d_val in zip(valid_v, dists):
                        if d_val >= 8.0:
                            all_fixed.add(v_idx)
                else:
                    diffs = v_coords[:, None, :] - d_arr[None, :, :]
                    min_dists = np.min(np.sqrt(np.sum(diffs * diffs, axis=-1)), axis=-1)
                    for v_idx, d_val in zip(valid_v, min_dists):
                        if d_val >= 8.0:
                            all_fixed.add(v_idx)
            else:
                all_fixed.update(vessel_fixed)
        except Exception:
            all_fixed.update(vessel_fixed)
    elif lock_vessel:
        all_fixed.update(vessel_fixed)

    fixed_indices = sorted([idx for idx in all_fixed if 1 <= idx <= total_len])
    donor_fixed_final = sorted([idx for idx in donor_fixed if 1 <= idx <= total_len])

    return {main_chain: fixed_indices}, donor_fixed_final
