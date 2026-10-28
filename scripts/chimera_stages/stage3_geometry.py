"""
scripts/chimera_stages/stage3_geometry.py
Stage 3: Geometric Bridge & Loop Math, Contig Formulation, and Linker Budgeting.

Handles:
- Dynamic Euclidean linker calculations from 3D coordinates.
- Physical reach limits and biophysical clamping of unfeasible linkers (L_ext_min = ceil(d / 3.8)).
- Multi-chain contig generation using the Euclidean Gap Graph algorithm.
- 4-mode chimera architecture contig assembly: replace, ligate_nterm, ligate_cterm, insert_loop.
- Terminal and internal bridge measurement (measure_junctions).
"""

import functools
import math
import os
import sys
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
    from scripts.extract_fixed_motif import parse_residue_spec
    from scripts.pse_pdb_extractor import parse_pdb_file
except ImportError:
    from extract_fixed_motif import parse_residue_spec
    from pse_pdb_extractor import parse_pdb_file

try:
    from .stage1_parsing import (
        _RE_RESIDUE_RANGE,
        _RE_NUM_OR_SPAN,
        _RE_HYPHEN_RANGE,
        _RE_LINKER_VAL,
        _RE_SHELL_INJECTION,
        MultiChainMotifManifest,
        build_multi_chain_manifest,
        get_residue_anchor_coords,
        get_residue_ca_coords,
    )
    from .stage2_clash import calculate_anchor_distance
except (ImportError, ValueError):
    from chimera_stages.stage1_parsing import (
        _RE_RESIDUE_RANGE,
        _RE_NUM_OR_SPAN,
        _RE_HYPHEN_RANGE,
        _RE_LINKER_VAL,
        _RE_SHELL_INJECTION,
        MultiChainMotifManifest,
        build_multi_chain_manifest,
        get_residue_anchor_coords,
        get_residue_ca_coords,
    )
    from chimera_stages.stage2_clash import calculate_anchor_distance


@functools.lru_cache(maxsize=4096)
def compute_gap_linker_length(d_gap: float) -> Tuple[int, int]:
    """
    Calculates dynamic linker length interval [L_min, L_max] from Euclidean gap distance.
    L_min = max(3, ceil(d_gap / 3.2))
    L_max = L_min + 5
    """
    l_min = max(3, int(math.ceil(float(d_gap) / 3.2)))
    l_max = l_min + 5
    return (l_min, l_max)


@functools.lru_cache(maxsize=4096)
def compute_gap_linker_string(d_gap: float) -> str:
    """Returns formatted linker string 'L_min-L_max' from Euclidean gap distance."""
    l_min, l_max = compute_gap_linker_length(round(float(d_gap), 3))
    return f"{l_min}-{l_max}"


def compute_recommended_linker(distance_angstrom: float) -> Tuple[int, int]:
    """
    Calculates physically realistic linker residue bounds:
    min_len = max(3, ceil(distance / 3.0))
    max_len = min_len + max(6, ceil(0.25 * min_len))
    """
    if distance_angstrom <= 0:
        return (3, 9)
    min_len = max(3, int(math.ceil(distance_angstrom / 3.0)))
    max_len = min_len + max(6, int(math.ceil(0.25 * min_len)))
    return (min_len, max_len)


def compute_extension_minimum(distance_angstrom: float) -> int:
    """Calculates absolute physical extension limit L_ext_min = ceil(d / 3.8)."""
    if distance_angstrom <= 0:
        return 1
    return max(1, int(math.ceil(distance_angstrom / 3.8)))


def check_and_clamp_linker(
    user_val: str, param_name: str, distance_angstrom: Optional[float], junction_label: str
) -> str:
    """
    Validates linker against physical Euclidean distance and applies biophysical clamping.
    If user_val is 'auto', automatically injects the recommended range.
    If max_len < L_ext_min, prints a prominent warning and auto-clamps to feasible span.
    """
    user_clean = str(user_val).strip()
    if distance_angstrom is None or distance_angstrom <= 0:
        if user_clean.lower() == "auto":
            return "6-10"
        return user_clean

    l_ext_min = compute_extension_minimum(distance_angstrom)
    rec_min, rec_max = compute_recommended_linker(distance_angstrom)
    rec_str = f"{rec_min}-{rec_max}"

    if user_clean.lower() == "auto":
        print(
            f"[+] Dynamic linker budgeting for {param_name} ({junction_label}): "
            f"{rec_str} (measured gap: {distance_angstrom:.1f} Å)"
        )
        return rec_str

    m = _RE_NUM_OR_SPAN.match(user_clean)
    if not m:
        return rec_str

    u_min = int(m.group(1))
    u_max = int(m.group(2)) if m.group(2) is not None else u_min
    if u_min > u_max:
        u_min, u_max = u_max, u_min

    if u_max < l_ext_min:
        print(
            f"\n[WARNING] Physical Euclidean distance at {junction_label} is {distance_angstrom:.1f} Å.\n"
            f"Specified {param_name} '{user_clean}' is physically incapable of spanning {distance_angstrom:.1f} Å "
            f"(max reach = {u_max * 3.8:.1f} Å, requires at least {l_ext_min} aa).\n"
            f"This WILL produce unphysical chain breaks and dotted lines in PyMOL!\n"
            f"--> Auto-adjusting {param_name} to physically feasible span: '{rec_str}'\n"
        )
        return rec_str
    elif u_min < l_ext_min:
        u_min = l_ext_min
        return f"{u_min}-{u_max}"

    return user_clean


def format_segments_to_contig(segments: List[Tuple[int, int]], chain_label: str) -> str:
    """
    Formats a list of contiguous segments into an RFdiffusion contig string part.
    Preserves internal gaps (e.g. crystallographic voids) between segments as 'gap-gap'.
    """
    parts = []
    for idx, (s, e) in enumerate(segments):
        if idx > 0:
            prev_e = segments[idx - 1][1]
            gap = s - prev_e - 1
            if gap > 0:
                parts.append(f"{gap}-{gap}")
        parts.append(f"{chain_label}{s}-{e}")
    return "/".join(parts)


def parse_motif_ranges(
    motif_residues: str,
    donor_metadata: Dict[str, Any],
    working_donor_pdb: str,
    donor_chain_id: str,
) -> List[Tuple[int, int]]:
    """
    Parses motif residue intervals, e.g. '30-45,78-82' or 'all'.
    Partitions into contiguous segments based on Delta resi > 1 OR
    Euclidean CA distance > 4.20 Angstrom.
    Returns sorted contiguous intervals [(start, end), ...].
    """
    motif_clean = str(motif_residues).strip().lower()
    if motif_clean in ("all", "full", "whole", "100%", "*"):
        res_nums = donor_metadata.get("residue_seqs", [])
        if not res_nums:
            donor_parsed = parse_pdb_file(working_donor_pdb)
            for obj in donor_parsed["objects"].values():
                res_nums.extend(obj.get("residue_seqs", []))
            res_nums = sorted(list(set(res_nums)))
    else:
        ordered_tuples = parse_residue_spec(motif_residues, default_chain=donor_chain_id)
        if not ordered_tuples:
            raise ValueError(f"No valid residues found in motif specification: '{motif_residues}'")
        res_nums = sorted(list(set(r for _, r in ordered_tuples)))

    if not res_nums:
        raise ValueError(f"No residue numbers resolved for motif in donor structure: {working_donor_pdb}")

    intervals: List[Tuple[int, int]] = []
    coords_list = [get_residue_ca_coords(working_donor_pdb, donor_chain_id, r) for r in res_nums]

    if len(res_nums) > 1 and all(c is not None for c in coords_list):
        arr_res = np.array(res_nums, dtype=int)
        delta_resis = arr_res[1:] - arr_res[:-1]
        coords_arr = np.array(coords_list, dtype=float)
        diffs = coords_arr[1:] - coords_arr[:-1]
        sq_dists = np.sum(diffs * diffs, axis=1)
        breaks = (delta_resis > 1) | (sq_dists > 17.64)

        break_indices = np.where(breaks)[0]
        cur_start_idx = 0
        for b_idx in break_indices:
            intervals.append((res_nums[cur_start_idx], res_nums[b_idx]))
            cur_start_idx = b_idx + 1
        intervals.append((res_nums[cur_start_idx], res_nums[-1]))
    else:
        cur_start = res_nums[0]
        for i in range(len(res_nums) - 1):
            curr_r = res_nums[i]
            next_r = res_nums[i + 1]
            delta_resi = next_r - curr_r

            c1 = coords_list[i]
            c2 = coords_list[i + 1]
            has_ca = c1 is not None and c2 is not None
            ca_dist = calculate_anchor_distance(c1, c2) if has_ca else 0.0

            if delta_resi > 1 or (has_ca and ca_dist > 4.20):
                intervals.append((cur_start, curr_r))
                cur_start = next_r

        intervals.append((cur_start, res_nums[-1]))

    return intervals


def parse_vessel_intervals(vessel_metadata: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Parses vessel residue intervals from vessel_metadata['intervals'] or range.
    Handles unmodeled crystallographic loops and gaps.
    Returns list of sorted (start, end) tuples.
    """
    raw_intervals = vessel_metadata.get("intervals", [])
    intervals: List[Tuple[int, int]] = []
    if raw_intervals:
        for item in raw_intervals:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                intervals.append((int(item[0]), int(item[1])))
            elif isinstance(item, str):
                m = _RE_RESIDUE_RANGE.match(item.strip())
                if m:
                    s = int(m.group(1))
                    e = int(m.group(2)) if m.group(2) is not None else s
                    intervals.append((min(s, e), max(s, e)))
    if not intervals:
        v_min, v_max = vessel_metadata.get("range", (1, 100))
        intervals = [(v_min, v_max)]

    intervals.sort(key=lambda x: x[0])
    return intervals


def slice_intervals(
    intervals: List[Tuple[int, int]], keep_min: Optional[int], keep_max: Optional[int]
) -> List[Tuple[int, int]]:
    """
    Clips a list of (start, end) intervals to [keep_min, keep_max] (inclusive).
    Returns only the segments that fall within the bounds.
    """
    sliced = []
    for s, e in intervals:
        cur_s = s if keep_min is None else max(s, keep_min)
        cur_e = e if keep_max is None else min(e, keep_max)
        if cur_s <= cur_e:
            sliced.append((cur_s, cur_e))
    return sliced


def _make_anchor(resi: int, chain: str, coord: np.ndarray) -> Dict[str, Any]:
    return {"resi": resi, "chain": chain, "coord": coord.tolist() if hasattr(coord, "tolist") else list(coord)}


def measure_junctions(
    working_donor_pdb: str,
    working_vessel_pdb: str,
    donor_chain_id: str,
    vessel_chain_id: str,
    motif_residues: str,
    chimera_mode: str,
    vessel_metadata: Dict[str, Any],
    donor_metadata: Dict[str, Any],
    vessel_replace_range: Optional[str] = None,
    vessel_split_after: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculates exact 3D Euclidean distances and recommended linker lengths
    for all junctions in the specified chimera mode.
    """
    motif_segments = parse_motif_ranges(motif_residues, donor_metadata, working_donor_pdb, donor_chain_id)
    vessel_intervals = parse_vessel_intervals(vessel_metadata)
    v_min, v_max = vessel_metadata.get("range", (1, 100))

    donor_n_res = motif_segments[0][0]
    donor_c_res = motif_segments[-1][1]

    donor_n_coord = get_residue_ca_coords(working_donor_pdb, donor_chain_id, donor_n_res)
    donor_c_coord = get_residue_ca_coords(working_donor_pdb, donor_chain_id, donor_c_res)

    junctions: Dict[str, Any] = {}
    mode_clean = chimera_mode.lower()

    if mode_clean in ["replace", "graft"]:
        if not vessel_replace_range:
            mid = (v_min + v_max) // 2
            rep_start = max(v_min + 5, mid - 10)
            rep_end = min(v_max - 5, mid + 10)
        else:
            rep_match = _RE_HYPHEN_RANGE.match(str(vessel_replace_range).strip())
            if not rep_match:
                raise ValueError(
                    f"Invalid vessel_replace_range: '{vessel_replace_range}'. "
                    "Expected format: 'START-END' (e.g. '55-70')."
                )
            rep_start = int(rep_match.group(1))
            rep_end = int(rep_match.group(2))
            if rep_start > rep_end:
                rep_start, rep_end = rep_end, rep_start

        left_segs = slice_intervals(vessel_intervals, None, rep_start - 1)
        right_segs = slice_intervals(vessel_intervals, rep_end + 1, None)

        if left_segs:
            v_left_res = left_segs[-1][1]
            v_left_coord = get_residue_ca_coords(working_vessel_pdb, vessel_chain_id, v_left_res)
            if v_left_coord is not None and donor_n_coord is not None:
                d_n = calculate_anchor_distance(v_left_coord, donor_n_coord)
                r_min, r_max = compute_recommended_linker(d_n)
                junctions["n_term"] = {
                    "param": "linker_n",
                    "description": f"N-terminal Junction (Vessel res {v_left_res} -> Donor res {donor_n_res})",
                    "distance_angstrom": round(d_n, 2),
                    "ext_min": compute_extension_minimum(d_n),
                    "glob_min": max(3, int(math.ceil(d_n / 3.0))),
                    "recommended_range": f"{r_min}-{r_max}",
                    "recommended_min": r_min,
                    "recommended_max": r_max,
                    "vessel_anchor": _make_anchor(v_left_res, vessel_chain_id, v_left_coord),
                    "donor_anchor": _make_anchor(donor_n_res, donor_chain_id, donor_n_coord),
                }

        if right_segs:
            v_right_res = right_segs[0][0]
            v_right_coord = get_residue_ca_coords(working_vessel_pdb, vessel_chain_id, v_right_res)
            if donor_c_coord is not None and v_right_coord is not None:
                d_c = calculate_anchor_distance(donor_c_coord, v_right_coord)
                r_min, r_max = compute_recommended_linker(d_c)
                junctions["c_term"] = {
                    "param": "linker_c",
                    "description": f"C-terminal Junction (Donor res {donor_c_res} -> Vessel res {v_right_res})",
                    "distance_angstrom": round(d_c, 2),
                    "ext_min": compute_extension_minimum(d_c),
                    "glob_min": max(3, int(math.ceil(d_c / 3.0))),
                    "recommended_range": f"{r_min}-{r_max}",
                    "recommended_min": r_min,
                    "recommended_max": r_max,
                    "donor_anchor": _make_anchor(donor_c_res, donor_chain_id, donor_c_coord),
                    "vessel_anchor": _make_anchor(v_right_res, vessel_chain_id, v_right_coord),
                }

    elif mode_clean in ["insert_loop", "loop", "insert"]:
        split_at = vessel_split_after
        if split_at is None:
            split_at = (v_min + v_max) // 2

        left_segs = slice_intervals(vessel_intervals, None, split_at)
        right_segs = slice_intervals(vessel_intervals, split_at + 1, None)

        if left_segs:
            v_left_res = left_segs[-1][1]
            v_left_coord = get_residue_ca_coords(working_vessel_pdb, vessel_chain_id, v_left_res)
            if v_left_coord is not None and donor_n_coord is not None:
                d_n = calculate_anchor_distance(v_left_coord, donor_n_coord)
                r_min, r_max = compute_recommended_linker(d_n)
                junctions["n_term"] = {
                    "param": "linker_n",
                    "description": (
                        f"N-terminal Junction (Vessel split res {v_left_res} -> Donor res {donor_n_res})"
                    ),
                    "distance_angstrom": round(d_n, 2),
                    "ext_min": compute_extension_minimum(d_n),
                    "glob_min": max(3, int(math.ceil(d_n / 3.0))),
                    "recommended_range": f"{r_min}-{r_max}",
                    "recommended_min": r_min,
                    "recommended_max": r_max,
                    "vessel_anchor": _make_anchor(v_left_res, vessel_chain_id, v_left_coord),
                    "donor_anchor": _make_anchor(donor_n_res, donor_chain_id, donor_n_coord),
                }

        if right_segs:
            v_right_res = right_segs[0][0]
            v_right_coord = get_residue_ca_coords(working_vessel_pdb, vessel_chain_id, v_right_res)
            if donor_c_coord is not None and v_right_coord is not None:
                d_c = calculate_anchor_distance(donor_c_coord, v_right_coord)
                r_min, r_max = compute_recommended_linker(d_c)
                junctions["c_term"] = {
                    "param": "linker_c",
                    "description": f"C-terminal Junction (Donor res {donor_c_res} -> Vessel res {v_right_res})",
                    "distance_angstrom": round(d_c, 2),
                    "ext_min": compute_extension_minimum(d_c),
                    "glob_min": max(3, int(math.ceil(d_c / 3.0))),
                    "recommended_range": f"{r_min}-{r_max}",
                    "recommended_min": r_min,
                    "recommended_max": r_max,
                    "donor_anchor": _make_anchor(donor_c_res, donor_chain_id, donor_c_coord),
                    "vessel_anchor": _make_anchor(v_right_res, vessel_chain_id, v_right_coord),
                }

    elif mode_clean in ["ligate_nterm", "nterm"]:
        v_n_res = vessel_intervals[0][0]
        v_n_coord = get_residue_ca_coords(working_vessel_pdb, vessel_chain_id, v_n_res)
        if donor_c_coord is not None and v_n_coord is not None:
            d = calculate_anchor_distance(donor_c_coord, v_n_coord)
            r_min, r_max = compute_recommended_linker(d)
            junctions["c_term"] = {
                "param": "linker_c",
                "description": (
                    f"N-terminal Ligation (Donor C-term res {donor_c_res} -> Vessel N-term res {v_n_res})"
                ),
                "distance_angstrom": round(d, 2),
                "ext_min": compute_extension_minimum(d),
                "glob_min": max(3, int(math.ceil(d / 3.0))),
                "recommended_range": f"{r_min}-{r_max}",
                "recommended_min": r_min,
                "recommended_max": r_max,
                "donor_anchor": _make_anchor(donor_c_res, donor_chain_id, donor_c_coord),
                "vessel_anchor": _make_anchor(v_n_res, vessel_chain_id, v_n_coord),
            }

    elif mode_clean in ["ligate_cterm", "cterm"]:
        v_c_res = vessel_intervals[-1][1]
        v_c_coord = get_residue_ca_coords(working_vessel_pdb, vessel_chain_id, v_c_res)
        if v_c_coord is not None and donor_n_coord is not None:
            d = calculate_anchor_distance(v_c_coord, donor_n_coord)
            r_min, r_max = compute_recommended_linker(d)
            junctions["n_term"] = {
                "param": "linker_n",
                "description": (
                    f"C-terminal Ligation (Vessel C-term res {v_c_res} -> Donor N-term res {donor_n_res})"
                ),
                "distance_angstrom": round(d, 2),
                "ext_min": compute_extension_minimum(d),
                "glob_min": max(3, int(math.ceil(d / 3.0))),
                "recommended_range": f"{r_min}-{r_max}",
                "recommended_min": r_min,
                "recommended_max": r_max,
                "vessel_anchor": _make_anchor(v_c_res, vessel_chain_id, v_c_coord),
                "donor_anchor": _make_anchor(donor_n_res, donor_chain_id, donor_n_coord),
            }

    # Measure internal gaps between consecutive donor segments
    internal_gaps: List[Dict[str, Any]] = []
    if len(motif_segments) > 1:
        seg_ends = [motif_segments[k][1] for k in range(len(motif_segments) - 1)]
        seg_next_starts = [motif_segments[k + 1][0] for k in range(len(motif_segments) - 1)]
        c_ends = [get_residue_ca_coords(working_donor_pdb, donor_chain_id, r) for r in seg_ends]
        n_starts = [get_residue_ca_coords(working_donor_pdb, donor_chain_id, r) for r in seg_next_starts]

        for k in range(len(motif_segments) - 1):
            seg_k_end = seg_ends[k]
            seg_next_start = seg_next_starts[k]
            c_k = c_ends[k]
            n_next = n_starts[k]
            if c_k is not None and n_next is not None:
                d_k = calculate_anchor_distance(c_k, n_next)
            else:
                seq_gap = max(1, seg_next_start - seg_k_end - 1)
                d_k = float(seq_gap * 3.8)

            l_ext = compute_extension_minimum(d_k)
            l_glob = max(3, int(math.ceil(d_k / 3.0)))
            l_max = l_glob + 6
            rec_range = f"{l_glob}-{l_max}"

            internal_gaps.append(
                {
                    "bridge_idx": k + 1,
                    "from_segment": k + 1,
                    "to_segment": k + 2,
                    "from_resi": seg_k_end,
                    "to_resi": seg_next_start,
                    "distance_angstrom": round(d_k, 2),
                    "ext_min": l_ext,
                    "glob_min": l_glob,
                    "recommended_range": rec_range,
                    "description": (
                        f"Internal Bridge {k + 1} (Segment S{k + 1} res {seg_k_end} "
                        f"-> Segment S{k + 2} res {seg_next_start})"
                    ),
                }
            )

    donor_segments_breakdown: List[Dict[str, Any]] = []
    for idx, (s, e) in enumerate(motif_segments, start=1):
        next_dist_str = "Terminal (Junction C)"
        if idx <= len(internal_gaps):
            gap_item = internal_gaps[idx - 1]
            next_dist_str = f"{gap_item['distance_angstrom']} Å -> Linker: {gap_item['recommended_range']}"
        donor_segments_breakdown.append(
            {
                "segment_id": f"S{idx}",
                "start": s,
                "end": e,
                "range_str": f"{s} - {e}",
                "length": e - s + 1,
                "distance_to_next": next_dist_str,
            }
        )

    return {
        "mode": chimera_mode,
        "junctions": junctions,
        "internal_gaps": internal_gaps,
        "donor_segments": donor_segments_breakdown,
        "total_donor_residues": sum(e - s + 1 for s, e in motif_segments),
    }


def compute_contigs_and_segments(
    working_donor_pdb: str,
    working_vessel_pdb: str,
    donor_chain_id: str,
    vessel_chain_id: str,
    motif_residues: str,
    chimera_mode: str,
    vessel_metadata: Dict[str, Any],
    donor_metadata: Dict[str, Any],
    linker_n: str = "6-10",
    linker_c: str = "6-10",
    vessel_replace_range: Optional[str] = None,
    vessel_split_after: Optional[int] = None,
    clamp_linkers: bool = True,
    seed_chain_id: str = "A",
    is_multichain: bool = False,
) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]], int, str, str, Optional[MultiChainMotifManifest]]:
    """
    Computes RFdiffusion contigs and list of preserved donor & vessel segments
    based on the selected chimera_mode.
    Returns:
      (contig_string, preserved_donor_segments, preserved_vessel_segments,
       donor_seed_offset, clamped_linker_n, clamped_linker_c, manifest)
    """
    if _RE_SHELL_INJECTION.search(str(chimera_mode)):
        raise ValueError(f"Dangerous shell operator or metacharacter detected in chimera_mode: '{chimera_mode}'")
    if (
        _RE_SHELL_INJECTION.search(str(seed_chain_id))
        or ".." in seed_chain_id
        or "/" in seed_chain_id
        or "\\" in seed_chain_id
    ):
        raise ValueError(f"Dangerous character or traversal detected in seed_chain_id: '{seed_chain_id}'")

    if is_multichain:
        manifest = build_multi_chain_manifest(
            motif_spec=motif_residues,
            default_source_file=working_donor_pdb,
            default_chain=donor_chain_id,
        )
        donor_seed_segments = []
        curr_seed = 1
        for seg in manifest.segments:
            seg_start = curr_seed
            seg_end = curr_seed + seg.length - 1
            donor_seed_segments.append((seg, seg_start, seg_end))
            curr_seed = seg_end + 1
        preserved_donor_segments = [(s, e) for _, s, e in donor_seed_segments]

        vessel_intervals = parse_vessel_intervals(vessel_metadata)
        v_min, v_max = vessel_metadata.get("range", (1, 100))

        def get_vessel_anchor(resi: int, atom_type: str) -> Optional[np.ndarray]:
            return get_residue_anchor_coords(working_vessel_pdb, vessel_chain_id, resi, atom_type)

        left_vessel: List[Tuple[int, int]] = []
        right_vessel: List[Tuple[int, int]] = []
        mode_clean = chimera_mode.lower()

        if mode_clean in ["replace", "graft"]:
            if not vessel_replace_range:
                mid = (v_min + v_max) // 2
                rep_start = max(v_min + 5, mid - 10)
                rep_end = min(v_max - 5, mid + 10)
            else:
                rep_match = _RE_HYPHEN_RANGE.match(str(vessel_replace_range).strip())
                if not rep_match:
                    raise ValueError(
                        f"Invalid vessel_replace_range: '{vessel_replace_range}'. "
                        "Expected format: 'START-END' (e.g. '55-70')."
                    )
                rep_start = int(rep_match.group(1))
                rep_end = int(rep_match.group(2))
                if rep_start > rep_end:
                    rep_start, rep_end = rep_end, rep_start

            left_vessel = slice_intervals(vessel_intervals, None, rep_start - 1)
            right_vessel = slice_intervals(vessel_intervals, rep_end + 1, None)
            preserved_vessel_segments = left_vessel + right_vessel

        elif mode_clean in ["ligate_nterm", "nterm"]:
            right_vessel = list(vessel_intervals)
            preserved_vessel_segments = list(vessel_intervals)

        elif mode_clean in ["ligate_cterm", "cterm"]:
            left_vessel = list(vessel_intervals)
            preserved_vessel_segments = list(vessel_intervals)

        elif mode_clean in ["insert_loop", "loop", "insert"]:
            split_at = vessel_split_after
            if split_at is None:
                split_at = (v_min + v_max) // 2
            elif not (v_min <= split_at <= v_max):
                raise ValueError(f"vessel_split_after ({split_at}) is out of vessel residue range [{v_min}, {v_max}].")

            left_vessel = slice_intervals(vessel_intervals, None, split_at)
            right_vessel = slice_intervals(vessel_intervals, split_at + 1, None)
            preserved_vessel_segments = left_vessel + right_vessel
        else:
            raise ValueError(f"Unknown chimera_mode: '{chimera_mode}'")

        blocks: List[Dict[str, Any]] = []
        if left_vessel:
            v_str = format_segments_to_contig(left_vessel, "V")
            n_c = get_vessel_anchor(left_vessel[0][0], "N")
            c_c = get_vessel_anchor(left_vessel[-1][1], "C")
            blocks.append({"str": v_str, "n_coord": n_c, "c_coord": c_c})

        for seg, s_seed, e_seed in donor_seed_segments:
            d_str = f"D{s_seed}-{e_seed}"
            n_c = seg.n_term_n
            c_c = seg.c_term_c
            blocks.append({"str": d_str, "n_coord": n_c, "c_coord": c_c})

        if right_vessel:
            v_str = format_segments_to_contig(right_vessel, "V")
            n_c = get_vessel_anchor(right_vessel[0][0], "N")
            c_c = get_vessel_anchor(right_vessel[-1][1], "C")
            blocks.append({"str": v_str, "n_coord": n_c, "c_coord": c_c})

        contig_parts = []
        for i in range(len(blocks)):
            contig_parts.append(blocks[i]["str"])
            if i < len(blocks) - 1:
                c1 = blocks[i]["c_coord"]
                n2 = blocks[i + 1]["n_coord"]
                if c1 is not None and n2 is not None:
                    d_gap = float(np.linalg.norm(c1 - n2))
                else:
                    d_gap = 15.0
                l_str = compute_gap_linker_string(d_gap)
                contig_parts.append(l_str)

        contig_string = "/".join(contig_parts)
        return contig_string, preserved_donor_segments, preserved_vessel_segments, 0, linker_n, linker_c, manifest

    # Standard single-chain donor path
    motif_segments = parse_motif_ranges(motif_residues, donor_metadata, working_donor_pdb, donor_chain_id)
    vessel_intervals = parse_vessel_intervals(vessel_metadata)
    v_min, v_max = vessel_metadata.get("range", (1, 100))

    # Validate linker specifications initially
    for lname, lval in [("linker_n", linker_n), ("linker_c", linker_c)]:
        if not _RE_LINKER_VAL.match(str(lval).strip()):
            raise ValueError(
                f"Invalid format for {lname}: '{lval}'. "
                "Expected format: 'auto', 'LEN', or 'MIN-MAX' (e.g. '6-10')."
            )

    measure_info = measure_junctions(
        working_donor_pdb=working_donor_pdb,
        working_vessel_pdb=working_vessel_pdb,
        donor_chain_id=donor_chain_id,
        vessel_chain_id=vessel_chain_id,
        motif_residues=motif_residues,
        chimera_mode=chimera_mode,
        vessel_metadata=vessel_metadata,
        donor_metadata=donor_metadata,
        vessel_replace_range=vessel_replace_range,
        vessel_split_after=vessel_split_after,
    )
    junctions = measure_info.get("junctions", {})

    final_linker_n = linker_n
    final_linker_c = linker_c

    if clamp_linkers:
        if "n_term" in junctions and junctions["n_term"].get("param") == "linker_n":
            j = junctions["n_term"]
            final_linker_n = check_and_clamp_linker(
                final_linker_n, "linker_n", j["distance_angstrom"], j["description"]
            )
        elif "c_term" in junctions and junctions["c_term"].get("param") == "linker_n":
            j = junctions["c_term"]
            final_linker_n = check_and_clamp_linker(
                final_linker_n, "linker_n", j["distance_angstrom"], j["description"]
            )
        elif str(final_linker_n).strip().lower() == "auto":
            final_linker_n = "6-10"

        if "c_term" in junctions and junctions["c_term"].get("param") == "linker_c":
            j = junctions["c_term"]
            final_linker_c = check_and_clamp_linker(
                final_linker_c, "linker_c", j["distance_angstrom"], j["description"]
            )
        elif "n_term" in junctions and junctions["n_term"].get("param") == "linker_c":
            j = junctions["n_term"]
            final_linker_c = check_and_clamp_linker(
                final_linker_c, "linker_c", j["distance_angstrom"], j["description"]
            )
        elif str(final_linker_c).strip().lower() == "auto":
            final_linker_c = "6-10"
    else:
        if str(final_linker_n).strip().lower() == "auto":
            j = (
                junctions.get("n_term", {})
                if junctions.get("n_term", {}).get("param") == "linker_n"
                else junctions.get("c_term", {})
            )
            final_linker_n = j.get("recommended_range", "6-10")
        if str(final_linker_c).strip().lower() == "auto":
            j = (
                junctions.get("c_term", {})
                if junctions.get("c_term", {}).get("param") == "linker_c"
                else junctions.get("n_term", {})
            )
            final_linker_c = j.get("recommended_range", "6-10")

    unified_chain = seed_chain_id
    vessel_res_set = set(r for s, e in vessel_intervals for r in range(s, e + 1))
    donor_res_set = set(r for s, e in motif_segments for r in range(s, e + 1))

    if vessel_res_set.intersection(donor_res_set):
        max_vessel = max(vessel_res_set) if vessel_res_set else 0
        donor_seed_offset = max(max_vessel + 100, 1000)
    else:
        donor_seed_offset = 0

    d_label = unified_chain
    v_label = unified_chain

    motif_contig_parts = []
    for idx, (m_start, m_end) in enumerate(motif_segments):
        if idx > 0:
            prev_end = motif_segments[idx - 1][1]
            c_prev = get_residue_ca_coords(working_donor_pdb, donor_chain_id, prev_end)
            n_curr = get_residue_ca_coords(working_donor_pdb, donor_chain_id, m_start)
            if c_prev is not None and n_curr is not None:
                d_gap = calculate_anchor_distance(c_prev, n_curr)
                l_glob = max(3, int(math.ceil(d_gap / 3.0)))
                l_max = l_glob + 6
                motif_contig_parts.append(f"{l_glob}-{l_max}")
            else:
                gap_res = m_start - prev_end - 1
                if gap_res > 0:
                    d_gap = float(gap_res * 3.8)
                    l_glob = max(3, int(math.ceil(d_gap / 3.0)))
                    l_max = l_glob + 6
                    motif_contig_parts.append(f"{l_glob}-{l_max}")
                else:
                    motif_contig_parts.append("3-9")
        seed_m_start = m_start + donor_seed_offset
        seed_m_end = m_end + donor_seed_offset
        motif_contig_parts.append(f"{d_label}{seed_m_start}-{seed_m_end}")

    motif_contig_str = "/".join(motif_contig_parts)
    vessel_segments: List[Tuple[int, int]] = []
    mode_clean = chimera_mode.lower()

    if mode_clean in ["replace", "graft"]:
        if not vessel_replace_range:
            mid = (v_min + v_max) // 2
            rep_start = max(v_min + 5, mid - 10)
            rep_end = min(v_max - 5, mid + 10)
        else:
            rep_match = _RE_HYPHEN_RANGE.match(str(vessel_replace_range).strip())
            if not rep_match:
                raise ValueError(
                    f"Invalid vessel_replace_range: '{vessel_replace_range}'. "
                    "Expected format: 'START-END' (e.g. '55-70')."
                )
            rep_start = int(rep_match.group(1))
            rep_end = int(rep_match.group(2))
            if rep_start > rep_end:
                rep_start, rep_end = rep_end, rep_start

        left_segs = slice_intervals(vessel_intervals, None, rep_start - 1)
        right_segs = slice_intervals(vessel_intervals, rep_end + 1, None)

        contig_pieces = []
        if left_segs:
            contig_pieces.append(format_segments_to_contig(left_segs, v_label))
            vessel_segments.extend(left_segs)
        contig_pieces.append(final_linker_n)
        contig_pieces.append(motif_contig_str)
        contig_pieces.append(final_linker_c)
        if right_segs:
            contig_pieces.append(format_segments_to_contig(right_segs, v_label))
            vessel_segments.extend(right_segs)

        contig_string = "/".join(contig_pieces)

    elif mode_clean in ["ligate_nterm", "nterm"]:
        vessel_segments.extend(vessel_intervals)
        v_contig = format_segments_to_contig(vessel_intervals, v_label)
        contig_string = f"{motif_contig_str}/{final_linker_c}/{v_contig}"

    elif mode_clean in ["ligate_cterm", "cterm"]:
        vessel_segments.extend(vessel_intervals)
        v_contig = format_segments_to_contig(vessel_intervals, v_label)
        contig_string = f"{v_contig}/{final_linker_n}/{motif_contig_str}"

    elif mode_clean in ["insert_loop", "loop", "insert"]:
        split_at = vessel_split_after
        if split_at is None:
            split_at = (v_min + v_max) // 2
        elif not (v_min <= split_at <= v_max):
            raise ValueError(f"vessel_split_after ({split_at}) is out of vessel residue range [{v_min}, {v_max}].")

        left_segs = slice_intervals(vessel_intervals, None, split_at)
        right_segs = slice_intervals(vessel_intervals, split_at + 1, None)

        contig_pieces = []
        if left_segs:
            contig_pieces.append(format_segments_to_contig(left_segs, v_label))
            vessel_segments.extend(left_segs)
        contig_pieces.append(final_linker_n)
        contig_pieces.append(motif_contig_str)
        contig_pieces.append(final_linker_c)
        if right_segs:
            contig_pieces.append(format_segments_to_contig(right_segs, v_label))
            vessel_segments.extend(right_segs)

        contig_string = "/".join(contig_pieces)

    else:
        raise ValueError(
            f"Unknown chimera_mode: '{chimera_mode}'. "
            "Must be replace, ligate_nterm, ligate_cterm, or insert_loop."
        )

    return contig_string, motif_segments, vessel_segments, donor_seed_offset, final_linker_n, final_linker_c, None
