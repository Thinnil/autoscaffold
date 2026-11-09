#!/usr/bin/env python3
"""
scripts/system_doctor.py
AutoScaffold Automated System Doctor & Preflight Diagnostic Engine.

Performs comprehensive machine-independent introspection:
- Host OS, architecture, kernel, CPU, RAM, and disk space.
- Hardware accelerators and CUDA driver authorization probe (safe subprocess isolation).
- Environment managers (micromamba, conda, mamba) across standard system & user paths.
- Virtual environments matrix (SE3nv, mlfold, colabfold, openmm, pymol).
- Deep learning neural model checkpoints & weights verification.
- Output formats: Colored/ASCII diagnostic cards, actionable remediation steps, and structured JSON.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

_RE_SHELL_INJECTION = re.compile(r"[;&|`><\n\r\$]")

# Resolve repository paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from scripts.version import __version__, get_git_sync_status
except ImportError:
    try:
        from version import __version__, get_git_sync_status
    except ImportError:
        __version__ = "2.2.0"

        def get_git_sync_status(r=None):
            return "Local Installation (v2.2.0)"

_HOST_SYSTEM_CACHE: Optional[Dict[str, Any]] = None
_CUDA_HARDWARE_CACHE: Optional[Dict[str, Any]] = None
_PACKAGE_MANAGERS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_VIRTUAL_ENVS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_MODEL_WEIGHTS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def clear_system_doctor_cache() -> None:
    """Clears all in-memory system doctor inspection caches."""
    global _HOST_SYSTEM_CACHE, _CUDA_HARDWARE_CACHE, _PACKAGE_MANAGERS_CACHE, _VIRTUAL_ENVS_CACHE, _MODEL_WEIGHTS_CACHE
    _HOST_SYSTEM_CACHE = None
    _CUDA_HARDWARE_CACHE = None
    _PACKAGE_MANAGERS_CACHE = None
    _VIRTUAL_ENVS_CACHE = None
    _MODEL_WEIGHTS_CACHE = None


def _is_mocked(target: Any) -> bool:
    return hasattr(target, "assert_called") or hasattr(target, "mock")


def probe_cuda_hardware(use_cache: bool = True) -> Dict[str, Any]:
    """
    Safely tests NVIDIA driver presence and PyTorch CUDA authorization via an isolated subprocess.
    Never throws unhandled exceptions or terminates the parent process if CUDA is absent or unprivileged.
    Caches probed results in memory for O(1) subsequent checks.
    """
    global _CUDA_HARDWARE_CACHE
    if _is_mocked(subprocess.run) or _is_mocked(shutil.which):
        use_cache = False

    if use_cache and _CUDA_HARDWARE_CACHE is not None:
        return json.loads(json.dumps(_CUDA_HARDWARE_CACHE))

    res = {
        "nvidia_smi_present": False,
        "cuda_driver_version": None,
        "gpus": [],
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_device_name": None,
        "cuda_authorized": False,
        "recommended_mode": "CPU_FALLBACK",
    }

    # 1. Probe nvidia-smi
    smi_bin = shutil.which("nvidia-smi")
    if smi_bin:
        res["nvidia_smi_present"] = True
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3.0,
                shell=False,
            )
            if p.returncode == 0 and p.stdout.strip():
                lines = [ln.strip() for ln in p.stdout.strip().splitlines() if ln.strip()]
                for ln in lines:
                    parts = ln.split(",")
                    name = parts[0].strip() if len(parts) > 0 else "NVIDIA GPU"
                    mem = parts[1].strip() if len(parts) > 1 else "Unknown VRAM"
                    res["gpus"].append({"name": name, "memory": mem})
        except Exception:
            pass

    # 2. Subprocess PyTorch CUDA probe
    py_exec = sys.executable or "python"
    probe_code = (
        "import json\n"
        "out = {'cuda_available': False, 'device_count': 0, 'device_name': None}\n"
        "try:\n"
        "    import torch\n"
        "    if torch.cuda.is_available():\n"
        "        out['cuda_available'] = True\n"
        "        out['device_count'] = torch.cuda.device_count()\n"
        "        if out['device_count'] > 0:\n"
        "            out['device_name'] = torch.cuda.get_device_name(0)\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(out))\n"
    )

    try:
        proc = subprocess.run(
            [py_exec, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in reversed(proc.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    res["torch_cuda_available"] = bool(data.get("cuda_available", False))
                    res["torch_cuda_device_count"] = int(data.get("device_count", 0))
                    res["torch_cuda_device_name"] = data.get("device_name")
                    break
    except Exception:
        pass

    if res["torch_cuda_available"] and res["torch_cuda_device_count"] > 0:
        res["cuda_authorized"] = True
        res["recommended_mode"] = "CUDA_ACCELERATED"
    else:
        res["cuda_authorized"] = False
        res["recommended_mode"] = "CPU_FALLBACK"

    if use_cache:
        _CUDA_HARDWARE_CACHE = res
    return res


def inspect_host_system(use_cache: bool = True) -> Dict[str, Any]:
    """Inspects OS, kernel, CPU, RAM, and disk storage with O(1) in-memory caching."""
    global _HOST_SYSTEM_CACHE
    if use_cache and _HOST_SYSTEM_CACHE is not None:
        return json.loads(json.dumps(_HOST_SYSTEM_CACHE))

    system_info = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count() or 1,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "disk_free_gb": None,
    }

    # Disk space in workspace
    try:
        total, used, free = shutil.disk_usage(REPO_ROOT)
        system_info["disk_free_gb"] = round(free / (1024**3), 2)
    except Exception:
        pass

    # RAM inspection
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        system_info["ram_total_gb"] = round(vm.total / (1024**3), 1)
        system_info["ram_available_gb"] = round(vm.available / (1024**3), 1)
    except Exception:
        # Fallback via OS commands or standard libraries
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r", buffering=65536) as f:
                    mem_dict = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            mem_dict[parts[0].strip()] = parts[1].strip()
                    if "MemTotal" in mem_dict:
                        kb = float(mem_dict["MemTotal"].split()[0])
                        system_info["ram_total_gb"] = round(kb / (1024**2), 1)
                    if "MemAvailable" in mem_dict:
                        kb = float(mem_dict["MemAvailable"].split()[0])
                        system_info["ram_available_gb"] = round(kb / (1024**2), 1)
            except Exception:
                pass

    if use_cache:
        _HOST_SYSTEM_CACHE = system_info
    return system_info


def discover_package_managers(use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """Discovers micromamba, mamba, and conda binaries and queries versions with O(1) caching."""
    global _PACKAGE_MANAGERS_CACHE
    if _is_mocked(subprocess.run) or _is_mocked(shutil.which):
        use_cache = False

    if use_cache and _PACKAGE_MANAGERS_CACHE is not None:
        return json.loads(json.dumps(_PACKAGE_MANAGERS_CACHE))
    managers = {}
    home = os.path.expanduser("~")
    local_envs_dir = os.environ.get("LOCAL_ENVS_DIR", os.path.join(REPO_ROOT, "envs"))

    search_targets = {
        "micromamba": [
            shutil.which("micromamba"),
            os.path.join(local_envs_dir, "bin", "micromamba"),
            os.path.join(local_envs_dir, "micromamba"),
            os.path.join(home, ".local", "bin", "micromamba"),
            os.path.join(home, "bin", "micromamba"),
            os.path.join(home, ".micromamba", "bin", "micromamba"),
            os.path.join(home, "micromamba", "micromamba"),
            "/usr/local/bin/micromamba",
            "/opt/conda/bin/micromamba",
        ],
        "mamba": [
            shutil.which("mamba"),
            os.path.join(home, "miniforge3", "bin", "mamba"),
            os.path.join(home, "mambaforge", "bin", "mamba"),
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
            os.path.join(home, "miniconda3", "bin", "conda"),
            os.path.join(home, "miniforge3", "bin", "conda"),
            os.path.join(home, "mambaforge", "bin", "conda"),
            os.path.join(home, "anaconda3", "bin", "conda"),
            "/opt/conda/bin/conda",
            "/usr/local/miniconda3/bin/conda",
            r"C:\miniconda3\Scripts\conda.exe",
            r"C:\miniconda3\condabin\conda.bat",
            r"C:\anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\miniconda3\Scripts\conda.exe",
        ],
    }

    for mgr_name, candidates in search_targets.items():
        found_bin = None
        for cand in candidates:
            if not cand:
                continue
            p = os.path.realpath(os.path.expanduser(str(cand)))
            if os.path.isfile(p) and (os.access(p, os.X_OK) or sys.platform == "win32"):
                found_bin = p
                break
        if found_bin and not _RE_SHELL_INJECTION.search(found_bin):
            ver = None
            try:
                out = subprocess.run(
                    [found_bin, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    shell=False,
                )
                if out.returncode == 0:
                    ver = out.stdout.strip() or out.stderr.strip()
            except Exception:
                pass
            managers[mgr_name] = {
                "installed": True,
                "path": found_bin,
                "version": ver,
            }
        else:
            managers[mgr_name] = {
                "installed": False,
                "path": None,
                "version": None,
            }

    if use_cache:
        _PACKAGE_MANAGERS_CACHE = managers
    return managers


def inspect_virtual_environments(use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """Probes the 5 target pipeline virtual environments with O(1) in-memory caching."""
    global _VIRTUAL_ENVS_CACHE
    if _is_mocked(subprocess.run):
        use_cache = False

    if use_cache and _VIRTUAL_ENVS_CACHE is not None:
        return json.loads(json.dumps(_VIRTUAL_ENVS_CACHE))

    envs_info = {}
    target_envs = {
        "SE3nv (RFdiffusion)": {
            "names": ["SE3nv", "rfdiffusion", "rf_diffusion"],
            "required_modules": ["torch", "hydra"],
        },
        "mlfold (ProteinMPNN)": {
            "names": ["mlfold", "SE3nv", "mpnn", "proteinmpnn"],
            "required_modules": ["torch"],
        },
        "colabfold (AlphaFold2)": {
            "names": ["colabfold", "colabfold_batch", "alphafold"],
            "required_modules": ["colabfold_batch", "jax", "torch"],
        },
        "openmm (Relaxation)": {
            "names": ["colabfold", "openmm", "base"],
            "required_modules": ["openmm"],
        },
        "pymol (Visualization)": {
            "names": ["pymol", "base"],
            "required_modules": ["pymol"],
        },
    }

    home = os.path.expanduser("~")
    conda_base_roots = [
        os.path.join(home, "miniconda3"),
        os.path.join(home, "miniforge3"),
        os.path.join(home, "mambaforge"),
        os.path.join(home, "anaconda3"),
        os.path.join(home, ".conda"),
        "/opt/conda",
        r"C:\miniconda3",
        r"C:\anaconda3",
    ]

    for label, spec in target_envs.items():
        found_py = None
        # Check active environment first
        active_prefix = os.environ.get("CONDA_PREFIX")
        active_env_name = os.environ.get("CONDA_DEFAULT_ENV", "")

        for name in spec["names"]:
            if active_env_name.lower() == name.lower() and active_prefix:
                cand = os.path.join(active_prefix, "bin", "python")
                if not os.path.isfile(cand):
                    cand = os.path.join(active_prefix, "python.exe")
                if os.path.isfile(cand):
                    found_py = cand
                    break

            for base in conda_base_roots:
                for cand in [
                    os.path.join(base, "envs", name, "bin", "python"),
                    os.path.join(base, "envs", name, "python.exe"),
                ]:
                    if os.path.isfile(cand):
                        found_py = cand
                        break
                if found_py:
                    break
            if found_py:
                break

        # Probe interpreter modules if found
        modules_status = {}
        is_ready = False
        torch_ver = None
        if found_py and not _RE_SHELL_INJECTION.search(found_py):
            probe_code = (
                "import json\n"
                f"reqs = {json.dumps(spec['required_modules'])}\n"
                "res = {'missing': [], 'torch_ver': None}\n"
                "for m in reqs:\n"
                "    try:\n"
                "        __import__(m)\n"
                "    except Exception:\n"
                "        res['missing'].append(m)\n"
                "try:\n"
                "    import torch\n"
                "    res['torch_ver'] = torch.__version__\n"
                "except Exception:\n"
                "    pass\n"
                "print(json.dumps(res))\n"
            )
            try:
                proc = subprocess.run(
                    [found_py, "-c", probe_code],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                    shell=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    for line in reversed(proc.stdout.strip().splitlines()):
                        if line.startswith("{") and line.endswith("}"):
                            d = json.loads(line)
                            missing = d.get("missing", [])
                            torch_ver = d.get("torch_ver")
                            for m in spec["required_modules"]:
                                modules_status[m] = (m not in missing)
                            is_ready = len(missing) == 0
                            break
            except Exception:
                pass
        else:
            # Check current sys.executable
            for m in spec["required_modules"]:
                try:
                    __import__(m)
                    modules_status[m] = True
                except Exception:
                    modules_status[m] = False
            is_ready = all(modules_status.values())
            if is_ready:
                found_py = sys.executable

        envs_info[label] = {
            "interpreter": found_py,
            "ready": is_ready,
            "modules": modules_status,
            "torch_version": torch_ver,
        }

    if use_cache:
        _VIRTUAL_ENVS_CACHE = envs_info
    return envs_info


def inspect_model_weights(use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """Verifies existence and file size of required neural checkpoints with O(1) in-memory caching."""
    global _MODEL_WEIGHTS_CACHE
    if use_cache and _MODEL_WEIGHTS_CACHE is not None:
        return json.loads(json.dumps(_MODEL_WEIGHTS_CACHE))

    home = os.path.expanduser("~")
    rfdiff_dir = os.environ.get("RFDIFF_DIR", os.path.join(home, "programs", "RFdiffusion"))
    mpnn_dir = os.environ.get("MPNN_DIR", os.path.join(home, "programs", "ProteinMPNN"))

    weights_info = {}

    # 1. RFdiffusion checkpoints
    rfdiff_candidates = [
        os.path.join(rfdiff_dir, "models", "ActiveSite_v01.pt"),
        os.path.join(rfdiff_dir, "models", "Base_ckpt.pt"),
        os.path.join(rfdiff_dir, "models", "Complex_base_ckpt.pt"),
    ]
    found_rfdiff_ckpts = []
    for ckpt in rfdiff_candidates:
        if os.path.isfile(ckpt):
            try:
                sz_mb = round(os.path.getsize(ckpt) / (1024**2), 1)
                if sz_mb > 0:
                    found_rfdiff_ckpts.append({"name": os.path.basename(ckpt), "size_mb": sz_mb, "path": ckpt})
            except OSError:
                pass

    weights_info["RFdiffusion Weights"] = {
        "status": "READY" if found_rfdiff_ckpts else "NOT_FOUND",
        "checkpoints": found_rfdiff_ckpts,
        "directory": os.path.join(rfdiff_dir, "models"),
    }

    # 2. ProteinMPNN weights
    mpnn_weights_dir = os.path.join(mpnn_dir, "vanilla_model_weights")
    found_mpnn_weights = []
    if os.path.isdir(mpnn_weights_dir):
        for f in os.listdir(mpnn_weights_dir):
            if f.endswith(".pt"):
                fp = os.path.join(mpnn_weights_dir, f)
                try:
                    sz_mb = round(os.path.getsize(fp) / (1024**2), 1)
                    if sz_mb > 0:
                        found_mpnn_weights.append({"name": f, "size_mb": sz_mb})
                except OSError:
                    pass

    weights_info["ProteinMPNN Weights"] = {
        "status": "READY" if found_mpnn_weights else "NOT_FOUND",
        "weights_count": len(found_mpnn_weights),
        "directory": mpnn_weights_dir,
    }

    if use_cache:
        _MODEL_WEIGHTS_CACHE = weights_info
    return weights_info


class SystemDoctor:
    """Diagnostic orchestrator producing cards, remediation, and JSON schemas."""

    def __init__(self, use_cache: bool = True):
        self.host_info = inspect_host_system(use_cache=use_cache)
        self.cuda_info = probe_cuda_hardware(use_cache=use_cache)
        self.managers_info = discover_package_managers(use_cache=use_cache)
        self.envs_info = inspect_virtual_environments(use_cache=use_cache)
        self.weights_info = inspect_model_weights(use_cache=use_cache)
        self.git_status = get_git_sync_status(REPO_ROOT)
        self.launcher_status = self._check_launcher_status()

    def _check_launcher_status(self) -> Dict[str, Any]:
        """Checks if the global 'autoscaffold' command is active in PATH and ~/.local/bin."""
        home = os.path.expanduser("~")
        local_bin = os.path.join(home, ".local", "bin", "autoscaffold")
        sys_bin = "/usr/local/bin/autoscaffold"
        in_path = shutil.which("autoscaffold") is not None
        installed = os.path.exists(local_bin) or os.path.exists(sys_bin)
        path = shutil.which("autoscaffold") or (local_bin if os.path.exists(local_bin) else None)
        return {
            "installed": installed or in_path,
            "in_path": in_path,
            "path": path,
        }

    def evaluate_verdict(self) -> Tuple[str, List[str]]:
        """Computes system operational status and actionable remediation steps."""
        remediations = []
        has_any_manager = any(m["installed"] for m in self.managers_info.values())
        if not has_any_manager:
            remediations.append(
                "No package manager (micromamba, conda, mamba) detected in standard locations.\n"
                "  --> Run './autoscaffold_suite.sh install' or './install.sh' to bootstrap micromamba."
            )

        if not self.cuda_info["cuda_authorized"]:
            remediations.append(
                "No CUDA accelerator driver authorization detected.\n"
                "  --> System will run smoothly in 100% functional CPU FALLBACK MODE.\n"
                "  --> (Optional for GPU speedup): Install NVIDIA proprietary drivers and CUDA toolkit."
            )

        missing_envs = [lbl for lbl, info in self.envs_info.items() if not info["ready"]]
        if missing_envs:
            remediations.append(
                f"Missing or unconfigured environments: {', '.join(missing_envs)}.\n"
                "  --> Run 'bash install.sh' or activate dedicated environment (e.g. 'conda activate SE3nv')."
            )

        if self.weights_info["RFdiffusion Weights"]["status"] != "READY":
            remediations.append(
                "RFdiffusion model checkpoints not found in $RFDIFF_DIR/models/.\n"
                "  --> Run 'bash scripts/download_weights.sh' or './autoscaffold_suite.sh install'."
            )

        # Operational verdict
        if not missing_envs and self.weights_info["RFdiffusion Weights"]["status"] == "READY":
            if self.cuda_info["cuda_authorized"]:
                verdict = "SYSTEM FULLY OPERATIONAL (GPU Acceleration Active)"
            else:
                verdict = "SYSTEM FULLY OPERATIONAL (CPU Fallback Mode Active)"
        else:
            if not self.cuda_info["cuda_authorized"]:
                verdict = "OPERATIONAL WITH WARNINGS (CPU Fallback Mode - Some Dependencies Pending)"
            else:
                verdict = "OPERATIONAL WITH WARNINGS (Some Dependencies Pending)"

        return verdict, remediations

    def to_dict(self) -> Dict[str, Any]:
        """Returns complete serializable diagnostic schema."""
        verdict, remediations = self.evaluate_verdict()
        return {
            "autoscaffold_version": __version__,
            "git_sync_status": self.git_status,
            "launcher_shortcut": self.launcher_status,
            "host_system": self.host_info,
            "accelerators": self.cuda_info,
            "package_managers": self.managers_info,
            "virtual_environments": self.envs_info,
            "model_weights": self.weights_info,
            "verdict": verdict,
            "remediation_steps": remediations,
        }

    def print_terminal_card(self) -> None:
        """Renders clean diagnostic cards matching specification."""
        verdict, remediations = self.evaluate_verdict()

        ram_val = self.host_info['ram_available_gb']
        ram_disp = f"{ram_val} GB Available" if ram_val else "Detected"
        disk_val = self.host_info['disk_free_gb']
        disk_disp = f"{disk_val} GB Free" if disk_val else "Accessible"

        print("======================================================================")
        print(f"       AUTOSCAFFOLD SYSTEM DOCTOR: PREFLIGHT DIAGNOSTICS v{__version__}")
        print("======================================================================")
        host_line = (
            f"[+] Host System: {self.host_info['os_name']} {self.host_info['os_release']} "
            f"({self.host_info['machine']}) | RAM: {ram_disp} | Disk: {disk_disp}"
        )
        print(host_line)
        print(f"[+] Upstream Sync: {self.git_status}")

        if self.launcher_status["in_path"]:
            print(f"[+] CLI Shortcut: Active in PATH ({self.launcher_status['path']})")
        elif self.launcher_status["installed"]:
            print(f"[!] CLI Shortcut: Installed at {self.launcher_status['path']} (Add to PATH in ~/.bashrc)")
        else:
            print("[!] CLI Shortcut: Not installed. Run './install.sh launcher' for global 'autoscaffold' command.")

        if self.cuda_info["cuda_authorized"]:
            dev_name = self.cuda_info["torch_cuda_device_name"] or "NVIDIA GPU"
            cnt = self.cuda_info["torch_cuda_device_count"]
            print(f"[+] Accelerators: CUDA Authorized ({cnt}x {dev_name})")
        else:
            print("[!] Accelerators: No CUDA driver authorization detected.")
            print("    --> System will operate in 100% functional CPU FALLBACK MODE.")

        # Managers
        mgrs = [k for k, v in self.managers_info.items() if v["installed"]]
        if mgrs:
            first = mgrs[0]
            print(f"[+] Package Managers: {first.capitalize()} detected at {self.managers_info[first]['path']}")
        else:
            print("[!] Package Managers: None detected in default search roots.")

        print("\nTarget Pipeline Environments:")
        for label, env_data in self.envs_info.items():
            status_tag = "[READY]" if env_data["ready"] else "[PENDING]"
            details = []
            if env_data.get("torch_version"):
                details.append(f"torch {env_data['torch_version']}")
            for mod, ok in env_data.get("modules", {}).items():
                if ok and mod != "torch":
                    details.append(mod)
            det_str = f" ({', '.join(details)})" if details else ""
            print(f"  [*] {label:<28}: {status_tag:>7}{det_str}")

        print("\nModel Weights & Checkpoints:")
        for label, w_data in self.weights_info.items():
            w_status = "[READY]" if w_data["status"] == "READY" else "[NOT_FOUND]"
            print(f"  [*] {label:<28}: {w_status:>7}")

        print("\n----------------------------------------------------------------------")
        print(f"Verdict: {verdict}")
        print("======================================================================")

        if remediations:
            print("\nActionable Remediation Guidance:")
            for idx, step in enumerate(remediations, start=1):
                print(f"  ({idx}) {step}")
            print("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoScaffold System Doctor: Machine-independent preflight diagnostics and environment inspection."
    )
    parser.add_argument("--doctor", action="store_true", help="Run full diagnostic suite")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--check-only", action="store_true", help="Silent exit code validation (0 = operational)")
    parser.add_argument("--version", "-v", action="version", version=f"AutoScaffold v{__version__}")

    args = parser.parse_args()

    doc = SystemDoctor()
    if args.json:
        print(json.dumps(doc.to_dict(), indent=2))
        return 0

    if not args.check_only:
        doc.print_terminal_card()

    verdict, _ = doc.evaluate_verdict()
    # If system is fully operational or operational with CPU fallback, return 0
    if "FULLY OPERATIONAL" in verdict:
        return 0
    return 0 if not args.check_only else 1


if __name__ == "__main__":
    sys.exit(main())
