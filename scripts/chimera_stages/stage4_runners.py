"""
scripts/chimera_stages/stage4_runners.py
Stage 4: Neural Network Runners and Closed-Loop Generation.

Handles:
- 5-tier Python interpreter and environment resolution (explicit, env vars, conda prefix, roots, host).
- Ephemeral environment runner creation (micromamba run, conda run, mamba run).
- High-visibility dependency alert boxes and CUDA availability warnings.
- Closed-loop backbone generation with in-flight drift auditing, candidate discarding, and garbage collection.
- ProteinMPNN command construction and subprocess dispatch.
- RunnerSpec dataclass and cache management.
"""

import gc
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Resolve scripts directory for helper modules
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = SCRIPTS_DIR
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.pse_pdb_extractor import safe_join_path, validate_output_dir
except ImportError:
    from pse_pdb_extractor import safe_join_path, validate_output_dir

try:
    from .stage1_parsing import _RE_SHELL_INJECTION
    from .stage2_clash import audit_motif_cartesian_drift, _purge_candidate_artifacts
except (ImportError, ValueError):
    from chimera_stages.stage1_parsing import _RE_SHELL_INJECTION
    from chimera_stages.stage2_clash import audit_motif_cartesian_drift, _purge_candidate_artifacts

# In-memory session cache for Python environment probes:
# Key: (tuple(cmd_prefix) or real_bin, tuple(sorted(req_modules))) -> probe_result_dict
_PROBE_CACHE: Dict[Any, Dict[str, Any]] = {}

# In-memory session cache for environment managers and roots:
_CACHED_MANAGERS: Optional[Dict[str, str]] = None
_CACHED_ROOTS: Optional[List[str]] = None


def _sync_caches_to_facade() -> None:
    """Synchronizes global cache variables to scripts.chimera_scaffold if loaded."""
    for mod_name in ("scripts.chimera_scaffold", "chimera_scaffold"):
        cs = sys.modules.get(mod_name)
        if cs is not None:
            cs._CACHED_MANAGERS = _CACHED_MANAGERS
            cs._CACHED_ROOTS = _CACHED_ROOTS


def _get_active_probe_fn():
    """Dynamically resolves probe_python_environment to respect unittest.mock.patch on chimera_scaffold."""
    for mod_name in ("scripts.chimera_scaffold", "chimera_scaffold"):
        cs = sys.modules.get(mod_name)
        if cs is not None and hasattr(cs, "probe_python_environment"):
            fn = getattr(cs, "probe_python_environment")
            if fn is not probe_python_environment:
                return fn
    return probe_python_environment


def _get_active_roots_fn():
    """Dynamically resolves get_all_environment_roots to respect unittest.mock.patch on chimera_scaffold."""
    for mod_name in ("scripts.chimera_scaffold", "chimera_scaffold"):
        cs = sys.modules.get(mod_name)
        if cs is not None and hasattr(cs, "get_all_environment_roots"):
            fn = getattr(cs, "get_all_environment_roots")
            if fn is not get_all_environment_roots:
                return fn
    return get_all_environment_roots


def _get_active_resolve_fn():
    """Dynamically resolves resolve_execution_runner to respect unittest.mock.patch on chimera_scaffold."""
    for mod_name in ("scripts.chimera_scaffold", "chimera_scaffold"):
        cs = sys.modules.get(mod_name)
        if cs is not None and hasattr(cs, "resolve_execution_runner"):
            fn = getattr(cs, "resolve_execution_runner")
            if fn is not resolve_execution_runner:
                return fn
    return resolve_execution_runner


def clear_probe_cache() -> None:
    """Clears the in-memory Python environment probe session cache."""
    _PROBE_CACHE.clear()
    for mod_name in ("scripts.chimera_scaffold", "chimera_scaffold"):
        cs = sys.modules.get(mod_name)
        if cs is not None and hasattr(cs, "_PROBE_CACHE") and cs._PROBE_CACHE is not _PROBE_CACHE:
            cs._PROBE_CACHE.clear()


def clear_environment_cache() -> None:
    """Clears the in-memory cached environment managers and roots."""
    global _CACHED_MANAGERS, _CACHED_ROOTS
    _CACHED_MANAGERS = None
    _CACHED_ROOTS = None
    _sync_caches_to_facade()


@dataclass
class RunnerSpec:
    """
    Encapsulates how an interpreter or tool is executed:
    either directly as an executable binary or via an ephemeral
    environment runner (conda run, micromamba run, mamba run).
    """

    runner_type: str  # 'direct', 'micromamba', 'conda', 'mamba', 'runner', 'custom'
    command_prefix: List[str]  # e.g. ['/path/to/python'] or ['micromamba', 'run', '-n', 'SE3nv', 'python']
    env_name: Optional[str] = None
    env_prefix: Optional[str] = None
    python_binary: Optional[str] = None
    is_valid: bool = False
    cuda_available: bool = False
    torch_version: Optional[str] = None
    cuda_device_name: Optional[str] = None
    missing_modules: List[str] = field(default_factory=list)

    def build_command(self, script_or_args: Union[str, List[str]], *extra_args: str) -> List[str]:
        """Builds the complete subprocess command list, validating against shell operator injection."""
        for part in self.command_prefix:
            if _RE_SHELL_INJECTION.search(str(part)):
                raise ValueError(
                    f"Dangerous shell operator, pipe, or semicolon detected in command prefix: '{part}'"
                )
        cmd = list(self.command_prefix)
        if isinstance(script_or_args, str):
            if _RE_SHELL_INJECTION.search(script_or_args):
                raise ValueError(
                    f"Dangerous shell operator, pipe, or semicolon detected in script argument: '{script_or_args}'"
                )
            cmd.append(script_or_args)
        else:
            for item in script_or_args:
                if _RE_SHELL_INJECTION.search(str(item)):
                    raise ValueError(
                        f"Dangerous shell operator, pipe, or semicolon detected in script arguments: '{item}'"
                    )
            cmd.extend(script_or_args)
        for extra in extra_args:
            if _RE_SHELL_INJECTION.search(str(extra)):
                raise ValueError(
                    f"Dangerous shell operator, pipe, or semicolon detected in extra argument: '{extra}'"
                )
        cmd.extend(extra_args)
        return cmd

    def display_label(self) -> str:
        """Human-readable representation for logs and error boxes."""
        if self.runner_type == "direct":
            return self.python_binary or " ".join(self.command_prefix)
        env_label = self.env_name or self.env_prefix or ""
        if env_label:
            return f"{' '.join(self.command_prefix)} (Env: {env_label})"
        return " ".join(self.command_prefix)


def format_runner_display(runner_or_bin: Any) -> str:
    """Formats a runner or binary target for human-readable display."""
    if runner_or_bin is None:
        return ""
    if hasattr(runner_or_bin, "display_label"):
        return runner_or_bin.display_label()
    if isinstance(runner_or_bin, (list, tuple)):
        return " ".join(str(x) for x in runner_or_bin)
    return str(runner_or_bin).strip()


def display_dependency_alert(
    tool_name: str, python_bin: Any, missing_modules: List[str], suggested_env: str = "SE3nv"
) -> str:
    """Renders a high-visibility terminal alert box with actionable remediation commands."""
    disp_bin = format_runner_display(python_bin)
    missing_str = ", ".join(missing_modules)
    cli_flag = "--python_rfdiff" if "rfdiff" in tool_name.lower() else "--python_mpnn"
    env_var = "PYTHON_RFDIFF" if "rfdiff" in tool_name.lower() else "PYTHON_MPNN"

    alert = f"""
====================================================================================================
 🛑 [AUTOSCAFFOLD ERROR]: MISSING NEURAL MODEL DEPENDENCIES IN RESOLVED PYTHON INTERPRETER
====================================================================================================
 Target Engine:      {tool_name}
 Resolved Python:    {disp_bin}
 Missing Module(s):  {missing_str}

 [REASON]
   {tool_name} requires {missing_str}, which live in a dedicated Conda environment
   (typically '{suggested_env}'). The resolved interpreter does not provide these modules.

 [ACTIONABLE REMEDIATION - SELECT ONE OPTION]

 Option 1: Activate the dedicated Conda environment before launching:
   $ conda activate {suggested_env}
   $ ./chimera_scaffold.sh ...

 Option 2: Provide the explicit Python path via CLI flag:
   $ ./chimera_scaffold.sh {cli_flag} /path/to/conda/envs/{suggested_env}/bin/python ...

 Option 3: Configure persistent path in 'config.env':
   Add the following line to config.env:
   {env_var}=/path/to/conda/envs/{suggested_env}/bin/python

 Option 4: Install the environment automatically if not yet created:
   $ bash install.sh
   or
   $ autoscaffold install
====================================================================================================
"""
    sys.stderr.write(alert + "\n")
    sys.stderr.flush()
    return alert


def display_cuda_warning(python_bin: Any, tool_name: str = "RFdiffusion") -> str:
    """Renders a warning banner when PyTorch is executing on CPU without CUDA acceleration."""
    disp_bin = format_runner_display(python_bin)
    if len(disp_bin) > 73:
        disp_bin_padded = disp_bin[:70] + "..."
    else:
        disp_bin_padded = f"{disp_bin:<73}"
    banner = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║ ⚠️  WARNING: NO GPU / CUDA ACCELERATION DETECTED FOR {tool_name.upper():<20} ║
╠════════════════════════════════════════════════════════════════════════════╣
║ PyTorch reports that CUDA is unavailable in:                              ║
║   {disp_bin_padded}║
║                                                                            ║
║ {tool_name} will execute on CPU. This will be 20x-50x slower than GPU.    ║
║ Expected runtime: ~5-15 minutes per backbone model.                       ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    sys.stderr.write(banner + "\n")
    sys.stderr.flush()
    return banner


def probe_python_environment(
    python_bin: Union[str, List[str], RunnerSpec],
    required_modules: Optional[List[str]] = None,
    check_cuda: bool = True,
    timeout_sec: float = 8.0,
    raise_on_error: bool = False,
    tool_name: str = "Neural Tool",
    suggested_env: str = "SE3nv",
) -> Dict[str, Any]:
    """
    Executes a fast probe subprocess using the target Python interpreter or command runner
    to verify module availability, environment integrity, and hardware acceleration status.
    Supports direct binary paths, command runner lists, and RunnerSpec instances.
    Maintains dual cache indexing for O(1) tuple(cmd_prefix) and real_bin lookups in _PROBE_CACHE.
    """
    req_modules = list(required_modules or [])
    result: Dict[str, Any] = {
        "missing_modules": [],
        "cuda_available": False,
        "torch_version": None,
        "cuda_device_name": None,
    }

    if python_bin is None or python_bin == "" or (isinstance(python_bin, (list, tuple)) and not python_bin):
        result["missing_modules"] = req_modules
        if raise_on_error and req_modules:
            display_dependency_alert(tool_name, str(python_bin), req_modules, suggested_env)
            raise EnvironmentError(f"Interpreter binary not specified for {tool_name}")
        return result

    real_bin = ""
    cmd_prefix: List[str] = []

    if isinstance(python_bin, RunnerSpec):
        cmd_prefix = list(python_bin.command_prefix)
        real_bin = python_bin.python_binary or (cmd_prefix[0] if cmd_prefix else "")
    elif isinstance(python_bin, (list, tuple)):
        cmd_prefix = [str(x) for x in python_bin]
        real_bin = cmd_prefix[0] if cmd_prefix else ""
    else:
        s_val = str(python_bin).strip()
        if os.path.isfile(s_val):
            real_bin = os.path.realpath(s_val)
            cmd_prefix = [real_bin]
        elif " " in s_val and any(s_val.startswith(kw) for kw in ["micromamba ", "conda ", "mamba ", "python "]):
            cmd_prefix = shlex.split(s_val)
            real_bin = cmd_prefix[0] if cmd_prefix else ""
        else:
            real_bin = os.path.realpath(os.path.expanduser(os.path.expandvars(s_val)))
            cmd_prefix = [real_bin]

    sorted_reqs = tuple(sorted(req_modules))
    key_runner = (tuple(cmd_prefix), sorted_reqs) if cmd_prefix else None
    key_bin = (real_bin, sorted_reqs) if real_bin else None

    if len(cmd_prefix) == 1:
        bin_target = cmd_prefix[0]
        if _RE_SHELL_INJECTION.search(bin_target):
            result["missing_modules"] = req_modules
            if raise_on_error:
                display_dependency_alert(tool_name, bin_target, req_modules, suggested_env)
                raise EnvironmentError(
                    f"Dangerous shell operator or metacharacter detected in interpreter binary: '{bin_target}'"
                )
            return result
        if not os.path.isfile(bin_target) or not (os.access(bin_target, os.X_OK) or sys.platform == "win32"):
            result["missing_modules"] = req_modules
            if raise_on_error and req_modules:
                display_dependency_alert(tool_name, bin_target, req_modules, suggested_env)
                raise EnvironmentError(
                    f"Interpreter binary does not exist or is not an executable file: '{bin_target}'"
                )
            return result
    elif not cmd_prefix:
        result["missing_modules"] = req_modules
        if raise_on_error and req_modules:
            display_dependency_alert(tool_name, str(python_bin), req_modules, suggested_env)
            raise EnvironmentError(f"Empty command runner specified for {tool_name}")
        return result
    else:
        for token in cmd_prefix:
            if _RE_SHELL_INJECTION.search(str(token)):
                result["missing_modules"] = req_modules
                if raise_on_error:
                    display_dependency_alert(tool_name, str(python_bin), req_modules, suggested_env)
                    raise EnvironmentError(
                        f"Dangerous shell operator or metacharacter detected in command runner argument: '{token}'"
                    )
                return result

    # Check session probe cache with dual indexing (O(1) runner tuple or binary string)
    cached = None
    if key_runner and key_runner in _PROBE_CACHE:
        cached = _PROBE_CACHE[key_runner]
    elif key_bin and key_bin in _PROBE_CACHE:
        cached = _PROBE_CACHE[key_bin]

    if cached is not None:
        if not check_cuda or cached.get("_cuda_checked", False):
            res_copy = {k: v for k, v in cached.items() if not k.startswith("_")}
            if raise_on_error and res_copy["missing_modules"]:
                disp_label = format_runner_display(python_bin)
                display_dependency_alert(tool_name, disp_label, res_copy["missing_modules"], suggested_env)
                raise EnvironmentError(
                    f"{tool_name} Python interpreter '{disp_label}' lacks required modules: "
                    f"{', '.join(res_copy['missing_modules'])}"
                )
            return res_copy

    probe_code = (
        "import sys, json\n"
        "out = {'missing_modules': [], 'cuda_available': False, 'torch_version': None, 'cuda_device_name': None}\n"
        f"reqs = {json.dumps(req_modules)}\n"
        "for m in reqs:\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception:\n"
        "        out['missing_modules'].append(m)\n"
        "if 'torch' not in out['missing_modules']:\n"
        "    try:\n"
        "        import torch\n"
        "        out['torch_version'] = str(getattr(torch, '__version__', 'unknown'))\n"
        f"        if {bool(check_cuda)}:\n"
        "            out['cuda_available'] = bool(torch.cuda.is_available())\n"
        "            if out['cuda_available'] and torch.cuda.device_count() > 0:\n"
        "                out['cuda_device_name'] = str(torch.cuda.get_device_name(0))\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps(out))\n"
    )

    full_cmd = list(cmd_prefix) + ["-c", probe_code]
    effective_timeout = max(1.0, timeout_sec)

    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=effective_timeout, shell=False)
        if proc.returncode == 0 and proc.stdout:
            lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
            parsed = None
            for ln in reversed(lines):
                if ln.startswith("{") and ln.endswith("}"):
                    try:
                        parsed = json.loads(ln)
                        break
                    except Exception:
                        pass
            if parsed and isinstance(parsed, dict):
                result.update(parsed)
            else:
                result["missing_modules"] = req_modules
        else:
            result["missing_modules"] = req_modules
    except Exception:
        result["missing_modules"] = req_modules

    cached_entry = dict(result)
    cached_entry["_cuda_checked"] = bool(check_cuda)
    if key_runner:
        _PROBE_CACHE[key_runner] = cached_entry
    if key_bin:
        _PROBE_CACHE[key_bin] = cached_entry

    if raise_on_error and result["missing_modules"]:
        disp_label = format_runner_display(python_bin)
        display_dependency_alert(tool_name, disp_label, result["missing_modules"], suggested_env)
        raise EnvironmentError(
            f"{tool_name} Python interpreter '{disp_label}' lacks required modules: "
            f"{', '.join(result['missing_modules'])}"
        )

    return result


def find_environment_managers(use_cache: bool = True) -> Dict[str, str]:
    """
    Discovers available environment managers (micromamba, mamba, conda)
    across system PATH, local directories, and standard installation roots.
    Caches results in module-level _CACHED_MANAGERS after the first scan.
    """
    global _CACHED_MANAGERS
    if use_cache and _CACHED_MANAGERS is not None:
        return dict(_CACHED_MANAGERS)

    managers: Dict[str, str] = {}
    home_dir = os.path.expanduser("~")
    local_envs_dir = os.environ.get("LOCAL_ENVS_DIR", os.path.join(SCRIPT_DIR, "..", "envs"))

    search_paths = {
        "micromamba": [
            shutil.which("micromamba"),
            os.path.join(local_envs_dir, "bin", "micromamba"),
            os.path.join(local_envs_dir, "micromamba"),
            os.path.join(home_dir, ".local", "bin", "micromamba"),
            os.path.join(home_dir, "bin", "micromamba"),
            os.path.join(home_dir, ".micromamba", "bin", "micromamba"),
            os.path.join(home_dir, "micromamba", "micromamba"),
            "/usr/local/bin/micromamba",
            "/opt/conda/bin/micromamba",
        ],
        "mamba": [
            shutil.which("mamba"),
            os.path.join(home_dir, "miniforge3", "bin", "mamba"),
            os.path.join(home_dir, "mambaforge", "bin", "mamba"),
            os.path.join(home_dir, "micromamba", "bin", "mamba"),
            "/opt/conda/bin/mamba",
        ],
        "conda": [
            shutil.which("conda"),
            os.environ.get("CONDA_EXE"),
            (
                os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "conda")
                if os.environ.get("CONDA_PREFIX")
                else None
            ),
            os.path.join(home_dir, "miniconda3", "bin", "conda"),
            os.path.join(home_dir, "miniforge3", "bin", "conda"),
            os.path.join(home_dir, "mambaforge", "bin", "conda"),
            os.path.join(home_dir, "anaconda3", "bin", "conda"),
            "/opt/conda/bin/conda",
            "/usr/local/miniconda3/bin/conda",
            r"C:\miniconda3\Scripts\conda.exe",
            r"C:\miniconda3\condabin\conda.bat",
            r"C:\anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\miniconda3\Scripts\conda.exe",
        ],
    }

    for mgr, paths in search_paths.items():
        for p in paths:
            if not p:
                continue
            p_clean = os.path.realpath(os.path.expanduser(os.path.expandvars(str(p).strip())))
            p_drive, p_rest = os.path.splitdrive(p_clean)
            is_root = (
                (not p_drive and p_clean.rstrip("/\\") == "")
                or (p_drive and p_rest in ("/", "\\", ""))
                or p_clean in ("/", "\\", "//", "\\\\")
            )
            if is_root:
                continue
            if os.path.isfile(p_clean) and (os.access(p_clean, os.X_OK) or sys.platform == "win32"):
                managers[mgr] = p_clean
                break

    if use_cache:
        _CACHED_MANAGERS = dict(managers)
        _sync_caches_to_facade()
    return dict(managers)


def get_all_environment_roots(use_cache: bool = True) -> List[str]:
    """
    Returns ordered list of root directories where conda and micromamba environments reside.
    Canonicalizes paths and excludes root directory escapes.
    Caches results in module-level _CACHED_ROOTS after the first scan.
    """
    global _CACHED_ROOTS
    if use_cache and _CACHED_ROOTS is not None:
        return list(_CACHED_ROOTS)

    home_dir = os.path.expanduser("~")
    local_envs_dir = os.environ.get("LOCAL_ENVS_DIR", os.path.join(SCRIPT_DIR, "..", "envs"))
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX", "")

    roots: List[str] = [
        local_envs_dir,
    ]
    if mamba_root:
        roots.append(os.path.join(mamba_root, "envs"))
        roots.append(mamba_root)

    # Micromamba roots
    roots.extend(
        [
            os.path.join(home_dir, ".micromamba", "envs"),
            os.path.join(home_dir, "micromamba", "envs"),
            os.path.join(home_dir, ".local", "share", "mamba", "envs"),
        ]
    )

    # Conda roots
    if conda_prefix:
        roots.append(os.path.join(conda_prefix, "envs"))
        roots.append(os.path.join(conda_prefix, "..", "envs"))

    roots.extend(
        [
            os.path.join(home_dir, "miniconda3", "envs"),
            os.path.join(home_dir, "miniforge3", "envs"),
            os.path.join(home_dir, "mambaforge", "envs"),
            os.path.join(home_dir, "anaconda3", "envs"),
            os.path.join(home_dir, ".conda", "envs"),
            "/opt/conda/envs",
            "/usr/local/miniconda3/envs",
            "/usr/local/anaconda3/envs",
            r"C:\miniconda3\envs",
            r"C:\anaconda3\envs",
            r"C:\ProgramData\miniconda3\envs",
            r"C:\ProgramData\anaconda3\envs",
        ]
    )

    valid_roots = []
    seen = set()
    for r in roots:
        if not r:
            continue
        clean_r = os.path.realpath(os.path.expanduser(os.path.expandvars(r)))
        p_drive, p_rest = os.path.splitdrive(clean_r)
        is_root = (
            (not p_drive and clean_r.rstrip("/\\") == "")
            or (p_drive and p_rest in ("/", "\\", ""))
            or clean_r in ("/", "\\", "//", "\\\\")
        )
        if is_root:
            continue
        if clean_r not in seen and os.path.isdir(clean_r):
            seen.add(clean_r)
            valid_roots.append(clean_r)

    if use_cache:
        _CACHED_ROOTS = list(valid_roots)
        _sync_caches_to_facade()
    return list(valid_roots)


def build_ephemeral_runner_cmd(
    manager_type: str, manager_bin: str, env_name: Optional[str] = None, env_prefix: Optional[str] = None
) -> List[str]:
    """
    Builds the command prefix for running python inside an ephemeral activated environment.
    e.g. ['micromamba', 'run', '-n', 'SE3nv', 'python']
    or ['conda', 'run', '-p', '/path/to/env', '--no-capture-output', 'python']
    """
    cmd = [manager_bin, "run"]
    if env_prefix:
        cmd.extend(["-p", env_prefix])
    elif env_name:
        cmd.extend(["-n", env_name])

    if manager_type == "conda":
        cmd.append("--no-capture-output")
    cmd.append("python")
    return cmd


def resolve_execution_runner(
    explicit_target: Optional[Union[str, List[str], RunnerSpec]] = None,
    env_var_keys: Optional[List[str]] = None,
    target_env_names: Optional[List[str]] = None,
    required_modules: Optional[List[str]] = None,
    tool_label: str = "Neural Tool",
    strict: bool = False,
) -> RunnerSpec:
    """
    Resolves the execution runner for a neural tool with resilient multi-candidate
    discovery, dual direct/ephemeral probing, and fallback iteration.
    Uses dynamic probe and roots delegation to support testing mocks.
    """
    active_probe = _get_active_probe_fn()
    active_roots = _get_active_roots_fn()

    req_mods = list(required_modules or [])
    managers = find_environment_managers()
    seen_bins = set()
    first_invalid_spec: Optional[RunnerSpec] = None
    cpu_fallback_runners: List[RunnerSpec] = []

    def test_candidate(
        cand_bin: Optional[str] = None,
        env_name: Optional[str] = None,
        env_prefix: Optional[str] = None,
        runner_prefix: Optional[List[str]] = None,
    ) -> Optional[RunnerSpec]:
        nonlocal first_invalid_spec
        if runner_prefix:
            for part in runner_prefix:
                if _RE_SHELL_INJECTION.search(str(part)):
                    return None
            probe = active_probe(
                runner_prefix, required_modules=req_mods, check_cuda=True, raise_on_error=False, tool_name=tool_label
            )
            if len(probe.get("missing_modules", [])) == 0:
                mgr_type = "runner"
                first_arg = str(runner_prefix[0]).lower()
                if "micromamba" in first_arg:
                    mgr_type = "micromamba"
                elif "conda" in first_arg:
                    mgr_type = "conda"
                elif "mamba" in first_arg:
                    mgr_type = "mamba"
                return RunnerSpec(
                    runner_type=mgr_type,
                    command_prefix=list(runner_prefix),
                    env_name=env_name,
                    env_prefix=env_prefix,
                    python_binary=cand_bin,
                    is_valid=True,
                    cuda_available=probe.get("cuda_available", False),
                    torch_version=probe.get("torch_version"),
                    cuda_device_name=probe.get("cuda_device_name"),
                    missing_modules=[],
                )
            return None

        if not cand_bin or _RE_SHELL_INJECTION.search(str(cand_bin)):
            return None
        clean_bin = os.path.realpath(os.path.expanduser(os.path.expandvars(str(cand_bin).strip())))
        if not (os.path.isfile(clean_bin) and (os.access(clean_bin, os.X_OK) or sys.platform == "win32")):
            return None
        if clean_bin in seen_bins:
            return None
        seen_bins.add(clean_bin)

        # Step A: Direct binary probe
        probe_direct = active_probe(
            clean_bin, required_modules=req_mods, check_cuda=True, raise_on_error=False, tool_name=tool_label
        )
        if len(probe_direct.get("missing_modules", [])) == 0:
            return RunnerSpec(
                runner_type="direct",
                command_prefix=[clean_bin],
                env_name=env_name,
                env_prefix=env_prefix,
                python_binary=clean_bin,
                is_valid=True,
                cuda_available=probe_direct.get("cuda_available", False),
                torch_version=probe_direct.get("torch_version"),
                cuda_device_name=probe_direct.get("cuda_device_name"),
                missing_modules=[],
            )

        calc_prefix = env_prefix
        if not calc_prefix:
            p_dir = os.path.dirname(clean_bin)
            gp_dir = os.path.dirname(p_dir)
            if os.path.basename(p_dir).lower() in ("bin", "scripts"):
                calc_prefix = gp_dir
            else:
                calc_prefix = p_dir
        calc_name = env_name or (os.path.basename(calc_prefix) if calc_prefix else None)

        if first_invalid_spec is None:
            first_invalid_spec = RunnerSpec(
                runner_type="direct",
                command_prefix=[clean_bin],
                env_name=calc_name,
                env_prefix=calc_prefix,
                python_binary=clean_bin,
                is_valid=False,
                missing_modules=probe_direct.get("missing_modules", req_mods),
            )

        # Step B: Ephemeral runner probe
        for mgr_key in ["micromamba", "mamba", "conda"]:
            if mgr_key in managers:
                mgr_path = managers[mgr_key]
                if calc_prefix and os.path.isdir(calc_prefix):
                    cmd_p = build_ephemeral_runner_cmd(mgr_key, mgr_path, env_prefix=calc_prefix)
                    spec_p = test_candidate(
                        cand_bin=clean_bin, env_name=calc_name, env_prefix=calc_prefix, runner_prefix=cmd_p
                    )
                    if spec_p:
                        return spec_p
                if calc_name:
                    cmd_n = build_ephemeral_runner_cmd(mgr_key, mgr_path, env_name=calc_name)
                    spec_n = test_candidate(
                        cand_bin=clean_bin, env_name=calc_name, env_prefix=calc_prefix, runner_prefix=cmd_n
                    )
                    if spec_n:
                        return spec_n

        return None

    # Tier 1: Explicit target
    if explicit_target:
        if isinstance(explicit_target, RunnerSpec):
            if explicit_target.is_valid:
                return explicit_target
            spec = test_candidate(runner_prefix=explicit_target.command_prefix)
            if spec:
                return spec
            return explicit_target
        elif isinstance(explicit_target, (list, tuple)):
            if any(_RE_SHELL_INJECTION.search(str(x)) for x in explicit_target):
                return first_invalid_spec or RunnerSpec(
                    runner_type="invalid", command_prefix=[], is_valid=False, missing_modules=req_mods
                )
            spec = test_candidate(runner_prefix=list(explicit_target))
            if spec:
                return spec
            return first_invalid_spec or RunnerSpec(
                runner_type="runner", command_prefix=list(explicit_target), is_valid=False, missing_modules=req_mods
            )
        else:
            s_val = str(explicit_target).strip()
            if _RE_SHELL_INJECTION.search(s_val):
                return first_invalid_spec or RunnerSpec(
                    runner_type="invalid", command_prefix=[], is_valid=False, missing_modules=req_mods
                )
            is_runner_cmd = s_val.startswith(("micromamba ", "conda ", "mamba ")) or (
                " " in s_val and not os.path.isfile(s_val) and " run " in s_val
            )
            if is_runner_cmd:
                spec = test_candidate(runner_prefix=shlex.split(s_val))
                if spec:
                    return spec
                return first_invalid_spec or RunnerSpec(
                    runner_type="runner", command_prefix=shlex.split(s_val), is_valid=False, missing_modules=req_mods
                )
            else:
                clean_exp = os.path.realpath(os.path.expanduser(os.path.expandvars(s_val)))
                if not (os.path.isfile(clean_exp) and (os.access(clean_exp, os.X_OK) or sys.platform == "win32")):
                    return RunnerSpec(
                        runner_type="direct",
                        command_prefix=[clean_exp],
                        python_binary=clean_exp,
                        is_valid=False,
                        missing_modules=req_mods,
                    )
                spec = test_candidate(cand_bin=clean_exp)
                if spec:
                    return spec
                return first_invalid_spec or RunnerSpec(
                    runner_type="direct",
                    command_prefix=[clean_exp],
                    python_binary=clean_exp,
                    is_valid=False,
                    missing_modules=req_mods,
                )

    # Tier 2: Environment variables
    if env_var_keys:
        for var in env_var_keys:
            val = os.environ.get(var)
            if val and str(val).strip():
                clean_val = str(val).strip()
                if _RE_SHELL_INJECTION.search(clean_val):
                    continue
                spec = test_candidate(cand_bin=clean_val)
                if spec:
                    if spec.cuda_available:
                        return spec
                    cpu_fallback_runners.append(spec)

    # Tier 3: Active Conda Prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for rel in ["bin/python", "python.exe", "Scripts/python.exe"]:
            cand = os.path.join(conda_prefix, rel)
            if not os.path.isfile(cand):
                continue
            spec = test_candidate(cand_bin=cand, env_prefix=conda_prefix)
            if spec:
                if spec.cuda_available:
                    return spec
                cpu_fallback_runners.append(spec)
                break

    # Tier 4: Filesystem scan across common roots (Conda + Micromamba)
    all_roots = active_roots()
    for env_name in target_env_names or []:
        clean_name = str(env_name).strip()
        if (
            not clean_name
            or ".." in clean_name
            or "/" in clean_name
            or "\\" in clean_name
            or _RE_SHELL_INJECTION.search(clean_name)
        ):
            continue
        for root in all_roots:
            env_dir = os.path.realpath(os.path.join(root, clean_name))
            if not env_dir.startswith(root) or not os.path.isdir(env_dir):
                continue
            for rel in ["bin/python", "python.exe", "Scripts/python.exe"]:
                cand = os.path.join(env_dir, rel)
                if not os.path.isfile(cand):
                    continue
                spec = test_candidate(cand_bin=cand, env_name=clean_name, env_prefix=env_dir)
                if spec:
                    if spec.cuda_available:
                        return spec
                    cpu_fallback_runners.append(spec)
                    break

    # Tier 5: Host interpreter fallback
    spec = test_candidate(cand_bin=sys.executable)
    if spec:
        if spec.cuda_available:
            return spec
        cpu_fallback_runners.append(spec)

    if cpu_fallback_runners:
        return cpu_fallback_runners[0]

    if first_invalid_spec:
        return first_invalid_spec

    clean_host = os.path.realpath(sys.executable)
    probe_host = active_probe(clean_host, required_modules=req_mods, raise_on_error=False)
    return RunnerSpec(
        runner_type="direct",
        command_prefix=[clean_host],
        python_binary=clean_host,
        is_valid=False,
        missing_modules=probe_host.get("missing_modules", req_mods),
    )


def resolve_python_interpreter(
    explicit_binary: Optional[Union[str, List[str]]] = None,
    env_var_keys: Optional[List[str]] = None,
    conda_env_names: Optional[List[str]] = None,
    required_modules: Optional[List[str]] = None,
    tool_label: str = "Neural Tool",
    strict: bool = False,
) -> Tuple[Union[str, List[str]], bool]:
    """
    Resolves the appropriate Python interpreter binary or runner prefix for a neural modeling tool
    following the 5-tier resolution hierarchy.
    """
    active_resolve = _get_active_resolve_fn()
    runner = active_resolve(
        explicit_target=explicit_binary,
        env_var_keys=env_var_keys,
        target_env_names=conda_env_names,
        required_modules=required_modules,
        tool_label=tool_label,
        strict=strict,
    )
    if runner.runner_type == "direct":
        py_bin = runner.python_binary or (runner.command_prefix[0] if runner.command_prefix else sys.executable)
        return (py_bin, runner.is_valid)
    return (runner.command_prefix, runner.is_valid)


def resolve_neural_python(
    explicit_path: Optional[Union[str, List[str]]] = None,
    env_var_names: Optional[List[str]] = None,
    target_env_names: Optional[List[str]] = None,
    required_modules: Optional[List[str]] = None,
    tool_label: str = "Neural Tool",
    strict: bool = False,
) -> Tuple[Union[str, List[str]], bool]:
    """Alias for resolve_python_interpreter conforming to RFDIFF_ENV_SPEC.md."""
    return resolve_python_interpreter(
        explicit_binary=explicit_path,
        env_var_keys=env_var_names,
        conda_env_names=target_env_names,
        required_modules=required_modules,
        tool_label=tool_label,
        strict=strict,
    )


def run_closed_loop_backbone_generation(
    seed_pdb: str,
    contig_str: str,
    seed_provenance: Dict[Union[int, str], Dict[str, Any]],
    target_passing_designs: int = 3,
    max_attempts: int = 50,
    max_drift_cutoff: float = 0.8,
    mean_drift_cutoff: float = 0.5,
    rfdiff_runner: Optional[Any] = None,
    rfdiff_script: Optional[str] = None,
    work_dir: str = "1_rfdiffusion",
    sampler_fn: Optional[Any] = None,
) -> List[str]:
    """
    Executes an active, iterative trial-and-error generation loop.
    Discards non-viable designs in-flight, stopping only when target_passing_designs
    pass strict Cartesian drift auditing, or max_attempts is reached.
    Safely purges all rejected micro-batch candidate artifacts and manages memory.
    """
    work_dir = validate_output_dir(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    if not seed_pdb or not os.path.isfile(seed_pdb):
        raise FileNotFoundError(f"Seed PDB not found: '{seed_pdb}'")
    real_seed = os.path.realpath(os.path.abspath(seed_pdb))
    p_drive, p_rest = os.path.splitdrive(real_seed)
    if (not p_drive and real_seed.rstrip("/\\") in ("", "/")) or (p_drive and p_rest in ("/", "\\", "")):
        raise ValueError(f"Dangerous or root file specified for seed_pdb: '{seed_pdb}'")

    passing_backbones: List[str] = []
    attempt = 0

    while len(passing_backbones) < target_passing_designs and attempt < max_attempts:
        attempt += 1
        prefix = safe_join_path(work_dir, f"candidate_{attempt}")

        cand_pdb = None
        if sampler_fn is not None:
            cand_pdb = sampler_fn(seed_pdb, contig_str, prefix, attempt)
        else:
            if not rfdiff_runner or not rfdiff_script or not os.path.isfile(rfdiff_script):
                raise FileNotFoundError(f"RFdiffusion script not found: '{rfdiff_script}'")
            cmd = rfdiff_runner.build_command(
                rfdiff_script,
                f"inference.output_prefix={prefix}",
                f"inference.input_pdb={seed_pdb}",
                f"contigmap.contigs=[{contig_str}]",
                "inference.num_designs=1",
                "denoiser.noise_scale_ca=0.0",
                "denoiser.noise_scale_frame=0.0",
            )
            rfdiff_repo = os.path.dirname(os.path.dirname(os.path.abspath(rfdiff_script)))
            cwd_dir = rfdiff_repo if os.path.isdir(rfdiff_repo) else None
            subprocess.run(cmd, check=True, shell=False, cwd=cwd_dir)
            expected_out = f"{prefix}_0.pdb"
            if os.path.isfile(expected_out):
                cand_pdb = expected_out

        if not cand_pdb or not os.path.isfile(cand_pdb):
            _purge_candidate_artifacts(prefix, None)
            continue

        max_drift, mean_drift, drift_details = audit_motif_cartesian_drift(
            candidate_pdb=cand_pdb,
            reference_seed_pdb=seed_pdb,
            seed_provenance=seed_provenance,
        )

        if max_drift > max_drift_cutoff or mean_drift > mean_drift_cutoff:
            print(
                f"[-] [Attempt {attempt}/{max_attempts}] REJECTED: Max drift {max_drift:.3f} Å > {max_drift_cutoff} Å "
                f"(Mean: {mean_drift:.3f} Å). Discarding."
            )
            _purge_candidate_artifacts(prefix, cand_pdb)
            del drift_details
            if attempt % 5 == 0:
                gc.collect()
            continue

        print(
            f"[+] [Attempt {attempt}/{max_attempts}] ACCEPTED: Max drift {max_drift:.3f} Å <= {max_drift_cutoff} Å "
            f"(Mean: {mean_drift:.3f} Å)"
        )
        passing_backbones.append(cand_pdb)
        del drift_details

    if not passing_backbones:
        raise RuntimeError(
            f"Closed-loop scaffolding failed to produce any passing models within {max_attempts} attempts. "
            f"All candidates exceeded max drift cutoff of {max_drift_cutoff} Å."
        )

    gc.collect()
    return passing_backbones
