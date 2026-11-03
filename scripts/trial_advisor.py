#!/usr/bin/env python3
"""
scripts/trial_advisor.py
AutoScaffold Next-Trials Recommendation Engine ("Trial Advisor").

Analyzes biophysical run outcomes, geometric constraints, junction gap distances,
linker slack ratios, and Cartesian drift failure modes.
Generates actionable parameter adjustments and ready-to-run CLI commands
for subsequent iterative protein engineering trials.
"""

import math
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from scripts.version import __version__
except ImportError:
    try:
        from version import __version__
    except ImportError:
        __version__ = "2.2.0"

try:
    from scripts.pse_pdb_extractor import validate_output_dir, safe_join_path
except ImportError:
    try:
        from pse_pdb_extractor import validate_output_dir, safe_join_path
    except ImportError:
        def validate_output_dir(p: str) -> str:
            real = os.path.realpath(os.path.abspath(str(p).strip()))
            drive, rest = os.path.splitdrive(real)
            if (not drive and real.rstrip("/\\") in ("", "/")) or (drive and rest in ("/", "\\", "")):
                raise ValueError(f"Dangerous or root output directory specified: '{p}'")
            return real

        def safe_join_path(base_dir: str, *paths: str) -> str:
            base_abs = os.path.realpath(os.path.abspath(base_dir))
            clean_parts = [re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.basename(str(x))) for x in paths]
            target_abs = os.path.realpath(os.path.abspath(os.path.join(base_abs, *clean_parts)))
            if os.path.commonpath([base_abs, target_abs]) != base_abs:
                raise ValueError("Path traversal detected")
            return target_abs

_RE_LINKER_BOUNDS = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


def parse_linker_range(val: Union[str, int, float, None]) -> Tuple[int, int, float]:
    """
    Parses linker length specification (e.g. '6-10', '8', 10).
    Returns (min_len, max_len, mean_len). Defaults to (6, 10, 8.0) if unparseable.
    """
    if val is None:
        return (6, 10, 8.0)
    s = str(val).strip()
    m = _RE_LINKER_BOUNDS.match(s)
    if m:
        low = int(m.group(1))
        high = int(m.group(2)) if m.group(2) else low
        if low > high:
            low, high = high, low
        return (low, high, (low + high) / 2.0)
    return (6, 10, 8.0)


def compute_slack_ratio(linker_len: float, gap_distance: float) -> float:
    """
    Computes peptide linker slack ratio sigma = (L * 3.8 A) / d_gap.
    Returns 99.0 if gap_distance <= 0.
    """
    if gap_distance <= 0.05:
        return 99.0
    return float((linker_len * 3.8) / gap_distance)


def classify_slack_ratio(sigma: float) -> str:
    """Classifies biophysical tension level based on slack ratio."""
    if sigma < 1.0:
        return "CRITICAL_STRAIN (Physically Impossible Span)"
    elif sigma < 1.3:
        return "HIGH_STRAIN (Taut Backbone / Pulling Tension)"
    elif sigma <= 2.6:
        return "OPTIMAL_FLEXIBILITY (Feasible Loop Closure)"
    elif sigma <= 3.2:
        return "MODERATE_SLACK (Permissive Span)"
    else:
        return "EXCESSIVE_SLACK (Floppy Loop / Steric Collapse Risk)"


class TrialAdvisor:
    """Biophysical reasoning engine formulating actionable next-trial recommendations."""

    def __init__(
        self,
        context: Any,
        summary_data: Optional[Dict[str, Any]] = None,
        status: str = "SUCCESS",
        error: Optional[Any] = None,
    ):
        self.ctx = context
        self.summary = dict(summary_data) if summary_data else {}
        self.status = str(status).upper()
        self.error = error
        self.recommendations: List[Dict[str, Any]] = []
        self.analysis: Dict[str, Any] = {}

    def _get_ctx_val(self, key: str, default: Any = None) -> Any:
        if isinstance(self.ctx, dict):
            return self.ctx.get(key, default)
        return getattr(self.ctx, key, default)

    def analyze(self) -> Dict[str, Any]:
        """Performs complete biophysical analysis and heuristic decision tree evaluation."""
        linker_n_raw = self._get_ctx_val("linker_n", "6-10")
        linker_c_raw = self._get_ctx_val("linker_c", "6-10")
        min_n, max_n, mean_n = parse_linker_range(linker_n_raw)
        min_c, max_c, mean_c = parse_linker_range(linker_c_raw)

        # Junction distances
        dist_n = None
        dist_c = None
        junc_data = self.summary.get("junction_measurements", {})
        if not junc_data and isinstance(self._get_ctx_val("junction_measurements"), dict):
            junc_data = self._get_ctx_val("junction_measurements")

        if junc_data:
            if "n_terminal_distance" in junc_data:
                dist_n = float(junc_data["n_terminal_distance"])
            if "c_terminal_distance" in junc_data:
                dist_c = float(junc_data["c_terminal_distance"])

        # Fallback to defaults or parsed values if not measured
        dist_n_val = dist_n if dist_n is not None else 15.0
        dist_c_val = dist_c if dist_c is not None else 15.0

        reach_n_max = max_n * 3.8
        reach_c_max = max_c * 3.8
        sigma_n = compute_slack_ratio(max_n, dist_n_val)
        sigma_c = compute_slack_ratio(max_c, dist_c_val)

        # Drift statistics
        trial_history = self.summary.get("trial_history") or self._get_ctx_val("trial_history", [])
        if not trial_history and hasattr(self.error, "rejection_history"):
            trial_history = getattr(self.error, "rejection_history")

        total_attempts = len(trial_history) if trial_history else 0
        drift_rejections = sum(1 for a in trial_history if a.get("status") == "REJECTED_DRIFT")
        max_drifts = [a.get("max_drift", 0.0) for a in trial_history if isinstance(a.get("max_drift"), (int, float))]
        highest_drift = max(max_drifts) if max_drifts else None

        # Steric clashes
        clashes = self.summary.get("steric_clashes_detected", 0)

        self.analysis = {
            "n_terminal": {
                "configured": f"{min_n}-{max_n}",
                "gap_distance_angstrom": dist_n,
                "max_reach_angstrom": round(reach_n_max, 1),
                "slack_ratio": round(sigma_n, 2),
                "slack_classification": classify_slack_ratio(sigma_n),
            },
            "c_terminal": {
                "configured": f"{min_c}-{max_c}",
                "gap_distance_angstrom": dist_c,
                "max_reach_angstrom": round(reach_c_max, 1),
                "slack_ratio": round(sigma_c, 2),
                "slack_classification": classify_slack_ratio(sigma_c),
            },
            "drift_audit": {
                "total_attempts": total_attempts,
                "rejections": drift_rejections,
                "highest_drift": round(highest_drift, 3) if highest_drift is not None else None,
            },
            "clashes": clashes,
        }

        # Formulate Recommendations
        recs = []

        # Rule 1: N-terminal strain / reach deficit
        if dist_n is not None and (sigma_n < 1.2 or dist_n > reach_n_max):
            req_aa = math.ceil(dist_n / 3.8)
            suggested_min = max(min_n + 4, req_aa + 1)
            suggested_max = max(max_n + 8, req_aa + 6)
            recs.append({
                "id": "REC-01-LINKER-N-EXTEND",
                "priority": "HIGH",
                "title": "Extend N-Terminal Linker",
                "observation": (
                    f"N-terminal junction gap is {dist_n:.1f} Å (slack ratio {sigma_n:.2f}, "
                    f"max reach {reach_n_max:.1f} Å). Linker is under severe physical strain."
                ),
                "recommendation": (
                    f"Increase '--linker_n {linker_n_raw}' to '--linker_n {suggested_min}-{suggested_max}'. "
                    f"This provides optimal loop closure without dragging donor active site residues."
                ),
                "param_override": {"linker_n": f"{suggested_min}-{suggested_max}"},
            })

        # Rule 2: C-terminal strain / reach deficit
        if dist_c is not None and (sigma_c < 1.2 or dist_c > reach_c_max):
            req_aa = math.ceil(dist_c / 3.8)
            suggested_min = max(min_c + 4, req_aa + 1)
            suggested_max = max(max_c + 8, req_aa + 6)
            recs.append({
                "id": "REC-02-LINKER-C-EXTEND",
                "priority": "HIGH",
                "title": "Extend C-Terminal Linker",
                "observation": (
                    f"C-terminal junction gap is {dist_c:.1f} Å (slack ratio {sigma_c:.2f}, "
                    f"max reach {reach_c_max:.1f} Å). Configured linker cannot comfortably bridge this distance."
                ),
                "recommendation": (
                    f"Increase '--linker_c {linker_c_raw}' to '--linker_c {suggested_min}-{suggested_max}'."
                ),
                "param_override": {"linker_c": f"{suggested_min}-{suggested_max}"},
            })

        # Rule 3: Excessive Slack & Clashes
        if dist_n is not None and sigma_n > 3.2 and clashes > 5:
            recs.append({
                "id": "REC-03-LINKER-N-SHORTEN",
                "priority": "MEDIUM",
                "title": "Shorten N-Terminal Linker",
                "observation": (
                    f"N-terminal linker has excessive slack (sigma = {sigma_n:.2f}) "
                    f"with steric clashes detected."
                ),
                "recommendation": (
                    f"Reduce '--linker_n {linker_n_raw}' to "
                    f"'--linker_n {max(3, min_n - 4)}-{max(5, max_n - 4)}'."
                ),
                "param_override": {"linker_n": f"{max(3, min_n - 4)}-{max(5, max_n - 4)}"},
            })

        if dist_c is not None and sigma_c > 3.2 and clashes > 5:
            recs.append({
                "id": "REC-04-LINKER-C-SHORTEN",
                "priority": "MEDIUM",
                "title": "Shorten C-Terminal Linker",
                "observation": (
                    f"C-terminal linker has excessive slack (sigma = {sigma_c:.2f}) "
                    f"with steric clashes detected."
                ),
                "recommendation": (
                    f"Reduce '--linker_c {linker_c_raw}' to "
                    f"'--linker_c {max(3, min_c - 4)}-{max(5, max_c - 4)}'."
                ),
                "param_override": {"linker_c": f"{max(3, min_c - 4)}-{max(5, max_c - 4)}"},
            })

        # Rule 4: Repeated Cartesian Drift Rejections
        if drift_rejections >= 3:
            recs.append({
                "id": "REC-05-SHIFT-HOST-ANCHOR",
                "priority": "HIGH",
                "title": "Shift Host Junction Anchor Residues",
                "observation": (
                    f"{drift_rejections} candidate designs were rejected due to donor Cartesian drift > 0.8 Å. "
                    "The current vessel junction vector is geometrically opposed to the donor active site pocket."
                ),
                "recommendation": (
                    "Shift the host junction attachment point by +/- 1 or +/- 2 residues along the host "
                    "secondary structure. This rotates the peptide exit vector toward the donor pocket."
                ),
                "param_override": {},
            })

        # Rule 5: Asymmetric Reach
        if dist_n is not None and dist_c is not None:
            mode = str(self._get_ctx_val("chimera_mode", "replace")).lower()
            if dist_c > 2.5 * dist_n and dist_c > 25.0 and mode == "replace":
                recs.append({
                    "id": "REC-06-GRAFT-MODE-ASYM-C",
                    "priority": "MEDIUM",
                    "title": "Evaluate N-Terminal Ligation Architecture",
                    "observation": (
                        f"Asymmetric reach detected: N-terminal gap is small ({dist_n:.1f} Å) "
                        f"while C-terminal gap is large ({dist_c:.1f} Å)."
                    ),
                    "recommendation": "Switch architecture mode from '--mode replace' to '--mode ligate_nterm'.",
                    "param_override": {"chimera_mode": "ligate_nterm"},
                })
            elif dist_n > 2.5 * dist_c and dist_n > 25.0 and mode == "replace":
                recs.append({
                    "id": "REC-07-GRAFT-MODE-ASYM-N",
                    "priority": "MEDIUM",
                    "title": "Evaluate C-Terminal Ligation Architecture",
                    "observation": (
                        f"Asymmetric reach detected: C-terminal gap is small ({dist_c:.1f} Å) "
                        f"while N-terminal gap is large ({dist_n:.1f} Å)."
                    ),
                    "recommendation": "Switch architecture mode from '--mode replace' to '--mode ligate_cterm'.",
                    "param_override": {"chimera_mode": "ligate_cterm"},
                })

        # Default fallback if run succeeded and no critical issues found
        if not recs and self.status == "SUCCESS":
            recs.append({
                "id": "REC-00-SUCCESS-EXPANSION",
                "priority": "LOW",
                "title": "Increase Sequence Diversity & Backbone Sampling",
                "observation": "Current run passed all biophysical invariance and geometric strain criteria.",
                "recommendation": (
                    "Expand sampling: Increase '--num_designs 20' and '--mpnn_seqs_per_bb 4' "
                    "to discover higher-affinity binders."
                ),
                "param_override": {"num_designs": 20, "mpnn_seqs_per_bb": 4},
            })

        self.recommendations = recs
        return {
            "status": self.status,
            "analysis": self.analysis,
            "recommendations": self.recommendations,
            "suggested_cli_command": self.generate_next_trial_command(),
        }

    def generate_next_trial_command(self) -> str:
        """Constructs a ready-to-run CLI invocation with the recommended overrides."""
        args = ["python scripts/run_chimera_pipeline.py"]

        for k in ["input_file", "donor_pdb", "vessel_pdb", "donor_id", "vessel_id", "motif_residues"]:
            val = self._get_ctx_val(k)
            if val:
                args.append(f"--{k} {shlex.quote(str(val))}")

        # Check overrides
        overrides = {}
        for r in self.recommendations:
            overrides.update(r.get("param_override", {}))

        mode = overrides.get("chimera_mode", self._get_ctx_val("chimera_mode", "replace"))
        args.append(f"--mode {shlex.quote(str(mode))}")

        rep_range = self._get_ctx_val("vessel_replace_range")
        if rep_range:
            args.append(f"--replace_range {shlex.quote(str(rep_range))}")

        ln = overrides.get("linker_n", self._get_ctx_val("linker_n", "6-10"))
        lc = overrides.get("linker_c", self._get_ctx_val("linker_c", "6-10"))
        args.append(f"--linker_n {shlex.quote(str(ln))}")
        args.append(f"--linker_c {shlex.quote(str(lc))}")

        num_des = overrides.get("num_designs", self._get_ctx_val("num_designs", 10))
        args.append(f"--num_designs {int(num_des)}")

        return " \\\n    ".join(args)

    def generate_markdown(self) -> str:
        """Renders GitHub Flavored Markdown matching specification 5.4."""
        run_name = os.path.basename(str(self._get_ctx_val("out_dir", "outputs/chimera_scaffold")))
        n_info = self.analysis.get("n_terminal", {})
        c_info = self.analysis.get("c_terminal", {})
        drift_info = self.analysis.get("drift_audit", {})

        md = []
        md.append("# AutoScaffold Trial Advisor: Post-Run Optimization & Next-Trial Recommendations")
        md.append("")
        md.append(f"- **Run ID**: `{run_name}`")
        md.append(f"- **AutoScaffold Version**: `v{__version__}`")
        md.append(f"- **Mode**: `{self._get_ctx_val('chimera_mode', 'replace')}`")
        md.append(f"- **Execution Status**: `{self.status}`")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 1. Biophysical Performance & Geometric Summary")
        md.append("")

        n_gap = f"{n_info.get('gap_distance_angstrom', 'N/A')} Å" if n_info.get('gap_distance_angstrom') else "N/A"
        c_gap = f"{c_info.get('gap_distance_angstrom', 'N/A')} Å" if c_info.get('gap_distance_angstrom') else "N/A"

        n_cfg = n_info.get('configured')
        n_slack = n_info.get('slack_ratio')
        n_class = n_info.get('slack_classification')
        md.append(f"- **N-Terminal Junction**: Gap = {n_gap} | Configured = {n_cfg} aa | Slack = {n_slack} ({n_class})")

        c_cfg = c_info.get('configured')
        c_slack = c_info.get('slack_ratio')
        c_class = c_info.get('slack_classification')
        md.append(f"- **C-Terminal Junction**: Gap = {c_gap} | Configured = {c_cfg} aa | Slack = {c_slack} ({c_class})")

        rej = drift_info.get("rejections", 0)
        tot = drift_info.get("total_attempts", 0)
        md.append(f"- **Donor Coordinate Invariance**: {rej} rejections out of {tot} closed-loop attempts.")
        md.append(f"- **Steric Clashes (d < 1.9 Å)**: {self.analysis.get('clashes', 0)} detected.")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 2. Actionable Recommendations for Next Trials")
        md.append("")

        for idx, rec in enumerate(self.recommendations, start=1):
            md.append(f"### 🎯 Recommendation {idx}: {rec['title']} ({rec['priority']} Priority)")
            md.append(f"- **Observation**: {rec['observation']}")
            md.append(f"- **Action**: {rec['recommendation']}")
            md.append("")

        md.append("---")
        md.append("")
        md.append("## 3. Ready-to-Run Next Trial Command")
        md.append("```bash")
        md.append(self.generate_next_trial_command())
        md.append("```")
        md.append("")

        return "\n".join(md)

    def write_artifacts(self, out_dir: Optional[str] = None) -> str:
        """Writes next_trials_recommendations.md and appends to run_methodology.txt."""
        raw_target_dir = out_dir or self._get_ctx_val("out_dir", "outputs/chimera_scaffold")
        target_dir = validate_output_dir(raw_target_dir)
        os.makedirs(target_dir, exist_ok=True)
        md_path = safe_join_path(target_dir, "next_trials_recommendations.md")
        md_text = self.generate_markdown()

        with open(md_path, "w", encoding="utf-8", buffering=65536) as f:
            f.write(md_text)

        # Append to run_methodology.txt if present
        meth_path = safe_join_path(target_dir, "run_methodology.txt")
        try:
            append_text = (
                "\n"
                "================================================================================\n"
                "AUTOSCAFFOLD TRIAL ADVISOR: POST-RUN RECOMMENDATIONS\n"
                "================================================================================\n"
            )
            for r in self.recommendations:
                append_text += f"[*] [{r['priority']}] {r['title']}:\n    {r['recommendation']}\n\n"
            append_text += f"Next Trial Command:\n{self.generate_next_trial_command()}\n"
            with open(meth_path, "a", encoding="utf-8", buffering=65536) as mf:
                mf.write(append_text)
        except Exception:
            pass

        return md_path

    def print_terminal_card(self) -> None:
        """Prints high-visibility terminal card."""
        lines = [
            "\n" + "=" * 70,
            "       AUTOSCAFFOLD TRIAL ADVISOR: NEXT-TRIAL RECOMMENDATIONS",
            "=" * 70,
        ]
        for idx, rec in enumerate(self.recommendations, start=1):
            lines.append(f"[{idx}] {rec['title']} ({rec['priority']} Priority)")
            lines.append(f"    --> {rec['recommendation']}")
        lines.append("\nReady-to-Run Next Trial Command:")
        lines.append(self.generate_next_trial_command())
        lines.append("=" * 70 + "\n")

        output = "\n".join(lines)
        try:
            print(output)
        except UnicodeEncodeError:
            print(output.encode("ascii", errors="replace").decode("ascii"))


def generate_trial_recommendations(
    context: Any,
    summary_data: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    status: str = "SUCCESS",
    error: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convenience functional interface to instantiate and run the Trial Advisor."""
    advisor = TrialAdvisor(context, summary_data=summary_data, status=status, error=error)
    advice = advisor.analyze()
    advisor.write_artifacts(out_dir=output_dir)
    advisor.print_terminal_card()
    return advice


def format_recommendations_markdown(
    context: Any,
    summary_data: Optional[Dict[str, Any]] = None,
    status: str = "SUCCESS",
    error: Optional[Any] = None,
) -> str:
    """Convenience functional interface to produce markdown text."""
    advisor = TrialAdvisor(context, summary_data=summary_data, status=status, error=error)
    advisor.analyze()
    return advisor.generate_markdown()
