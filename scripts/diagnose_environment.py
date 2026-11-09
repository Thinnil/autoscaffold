#!/usr/bin/env python3
"""
scripts/diagnose_environment.py
AutoScaffold Deep Connectivity, Dependency & Subsystem Diagnostic Probe.

Strictly compatible with Python 3.6+ using ONLY standard library modules.
Inspects:
1. Host OS, Kernel, Architecture, and Environment Variables.
2. System & Virtual Environment Python Interpreters.
3. Standard Library Feature Matrix (dataclasses, argparse subparser, subprocess).
4. AutoScaffold Internal Module Importability & Stage Connectivity.
5. Neural Network Dependencies (PyTorch, CUDA, ProteinMPNN, RFdiffusion, OpenMM).
6. Shell Wrapper Summoning & Conda Environment Handshake.
7. Generates a structured OrchestAI Diagnostic Dossier.
"""

import sys
import os
import shutil
import subprocess
import platform
import json
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
STAGES_DIR = os.path.join(SCRIPTS_DIR, "chimera_stages")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_probe_cmd(cmd, timeout=3.0):
    """Executes a subprocess safely with universal Python 3.6+ compatibility."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            shell=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def get_host_info():
    """Gathers OS and system details."""
    info = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "distro": "Unknown",
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "home": os.environ.get("HOME", ""),
        "cwd": os.getcwd(),
        "repo_root": REPO_ROOT,
    }
    if os.path.isfile("/etc/os-release"):
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info["distro"] = line.split("=", 1)[1].strip().strip('"')
                        break
        except Exception:
            pass
    return info


def find_all_python_interpreters():
    """Locates all candidate Python interpreters across system and conda paths."""
    candidates = []
    # 1. Current executable
    if sys.executable and sys.executable not in candidates:
        candidates.append(sys.executable)

    # 2. PATH binaries
    for name in ["python3", "python", "python3.10", "python3.9", "python3.8", "python3.7"]:
        path = shutil.which(name)
        if path and path not in candidates:
            candidates.append(path)

    # 3. Known system binaries
    for sys_path in ["/usr/bin/python3", "/usr/bin/python", "/usr/local/bin/python3", "/usr/local/bin/python"]:
        if os.path.isfile(sys_path) and os.access(sys_path, os.X_OK) and sys_path not in candidates:
            candidates.append(sys_path)

    # 4. Standard Conda & Micromamba installations
    home = os.path.expanduser("~")
    search_roots = [
        os.path.join(REPO_ROOT, "envs"),
        os.path.join(home, ".micromamba", "envs"),
        os.path.join(home, "micromamba", "envs"),
        os.path.join(home, ".local", "share", "mamba", "envs"),
        os.path.join(home, "miniconda3", "envs"),
        os.path.join(home, "miniforge3", "envs"),
        os.path.join(home, "mambaforge", "envs"),
        os.path.join(home, "anaconda3", "envs"),
        "/opt/conda/envs",
        "/usr/local/miniconda3/envs",
    ]
    for env_root in search_roots:
        if os.path.isdir(env_root):
            try:
                for env_dir in os.listdir(env_root):
                    full_env = os.path.join(env_root, env_dir)
                    for py_name in ["bin/python", "bin/python3", "python.exe"]:
                        py_cand = os.path.join(full_env, py_name)
                        if os.path.isfile(py_cand) and os.access(py_cand, os.X_OK) and py_cand not in candidates:
                            candidates.append(py_cand)
            except Exception:
                pass

    return candidates


def inspect_python_interpreter(py_bin):
    """Probes a specific Python interpreter for version, features, and packages."""
    probe_script = """
import sys, json

res = {
    'version': sys.version.split()[0],
    'version_info': list(sys.version_info[:3]),
    'executable': sys.executable,
    'has_dataclasses': False,
    'has_subparsers_required': False,
    'has_numpy': False,
    'numpy_ver': None,
    'has_scipy': False,
    'scipy_ver': None,
    'has_torch': False,
    'torch_ver': None,
    'cuda_available': False,
    'device_count': 0,
    'has_pymol': False,
    'has_openmm': False,
}

try:
    import dataclasses
    res['has_dataclasses'] = True
except Exception:
    pass

try:
    import argparse
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    res['has_subparsers_required'] = True
except Exception:
    pass

try:
    import numpy
    res['has_numpy'] = True
    res['numpy_ver'] = getattr(numpy, '__version__', 'unknown')
except Exception:
    pass

try:
    import scipy
    res['has_scipy'] = True
    res['scipy_ver'] = getattr(scipy, '__version__', 'unknown')
except Exception:
    pass

try:
    import torch
    res['has_torch'] = True
    res['torch_ver'] = getattr(torch, '__version__', 'unknown')
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        res['cuda_available'] = True
        res['device_count'] = torch.cuda.device_count()
except Exception:
    pass

try:
    import pymol
    res['has_pymol'] = True
except Exception:
    pass

try:
    import openmm
    res['has_openmm'] = True
except Exception:
    pass

print(json.dumps(res))
"""
    code, stdout, stderr = run_probe_cmd([py_bin, "-c", probe_script], timeout=5.0)
    if code == 0 and stdout:
        try:
            return json.loads(stdout)
        except Exception:
            pass
    return {
        "executable": py_bin,
        "version": "Error / Unusable",
        "error": stderr or "Failed to execute Python probe",
    }


def audit_autoscaffold_modules(py_bin):
    """Tests importing each AutoScaffold module using the specified Python binary."""
    modules_to_test = [
        ("version", "scripts.version"),
        ("pse_pdb_extractor", "scripts.pse_pdb_extractor"),
        ("stage1_parsing", "chimera_stages.stage1_parsing"),
        ("stage2_clash", "chimera_stages.stage2_clash"),
        ("stage3_geometry", "chimera_stages.stage3_geometry"),
        ("stage4_runners", "chimera_stages.stage4_runners"),
        ("stage5_export", "chimera_stages.stage5_export"),
        ("chimera_stages_init", "chimera_stages"),
        ("chimera_scaffold", "scripts.chimera_scaffold"),
        ("run_chimera_pipeline", "scripts.run_chimera_pipeline"),
        ("discontinuous_motif_scaffold", "scripts.discontinuous_motif_scaffold"),
        ("methodology_reporter", "scripts.methodology_reporter"),
        ("relax_chimera", "scripts.relax_chimera"),
        ("system_doctor", "scripts.system_doctor"),
    ]

    results = {}
    for short_name, mod_path in modules_to_test:
        test_code = (
            "import sys, os\n"
            "repo_root = " + repr(REPO_ROOT) + "\n"
            "scripts_dir = os.path.join(repo_root, 'scripts')\n"
            "stages_dir = os.path.join(scripts_dir, 'chimera_stages')\n"
            "for p in [repo_root, scripts_dir, stages_dir]:\n"
            "    if p not in sys.path: sys.path.insert(0, p)\n"
            "try:\n"
            "    __import__('" + mod_path + "')\n"
            "    print('SUCCESS')\n"
            "except Exception as e:\n"
            "    import traceback\n"
            "    print('ERROR: ' + str(e))\n"
            "    traceback.print_exc()\n"
        )
        code, stdout, stderr = run_probe_cmd([py_bin, "-c", test_code], timeout=4.0)
        if "SUCCESS" in stdout:
            results[short_name] = {"status": "OK", "error": None}
        else:
            err_msg = ""
            for line in (stdout + "\n" + stderr).splitlines():
                if line.startswith("ERROR:") or "Error:" in line or "Exception" in line:
                    err_msg = line.strip()
                    break
            if not err_msg:
                err_msg = stderr.strip() or stdout.strip() or "Import failed"
            results[short_name] = {"status": "FAILED", "error": err_msg}

    return results


def audit_shell_configuration():
    """Inspects shell environment variables and AutoScaffold configuration."""
    cfg = {
        "AUTOSCAFFOLD_HOME": os.environ.get("AUTOSCAFFOLD_HOME", "Not set"),
        "PYTHON_BIN": os.environ.get("PYTHON_BIN", "Not set"),
        "PYTHON_RFDIFF": os.environ.get("PYTHON_RFDIFF", "Not set"),
        "PYTHON_MPNN": os.environ.get("PYTHON_MPNN", "Not set"),
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", "None (Host Base/System)"),
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", "None"),
        "RFDIFF_DIR": os.environ.get("RFDIFF_DIR", os.path.expanduser("~/programs/RFdiffusion")),
        "MPNN_DIR": os.environ.get("MPNN_DIR", os.path.expanduser("~/programs/ProteinMPNN")),
    }

    # Check existence of tool repos and weights
    cfg["rfdiff_repo_exists"] = os.path.isdir(cfg["RFDIFF_DIR"])
    cfg["rfdiff_inference_exists"] = os.path.isfile(os.path.join(cfg["RFDIFF_DIR"], "scripts", "run_inference.py"))
    models_dir = os.path.join(cfg["RFDIFF_DIR"], "models")
    cfg["rfdiff_weights_present"] = os.path.isdir(models_dir) and len(os.listdir(models_dir)) > 0 if os.path.isdir(models_dir) else False

    cfg["mpnn_repo_exists"] = os.path.isdir(cfg["MPNN_DIR"])
    cfg["mpnn_script_exists"] = os.path.isfile(os.path.join(cfg["MPNN_DIR"], "protein_mpnn_run.py"))

    return cfg


def main():
    # Ensure stdout/stderr do not fail on restricted Windows cp1252 or non-UTF8 terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="replace")
        except Exception:
            pass

    print("=" * 80)
    print(" [DIAGNOSTICS] AUTOSCAFFOLD: COMPREHENSIVE DEEP-SYSTEM DIAGNOSTIC PROBE")
    print("=" * 80)

    host = get_host_info()
    print("\n[1] HOST & OPERATING SYSTEM:")
    print("    * OS:           %s (%s %s)" % (host['os'], host['release'], host['architecture']))
    print("    * Distribution: %s" % host['distro'])
    print("    * Active User:  %s" % host['user'])
    print("    * Working Dir:  %s" % host['cwd'])
    print("    * AutoScaffold: %s" % host['repo_root'])

    shell_cfg = audit_shell_configuration()
    print("\n[2] SHELL ENVIRONMENT & SUMMONING STATE:")
    print("    * AUTOSCAFFOLD_HOME: %s" % shell_cfg['AUTOSCAFFOLD_HOME'])
    print("    * Active Conda Env:  %s (%s)" % (shell_cfg['CONDA_DEFAULT_ENV'], shell_cfg['CONDA_PREFIX']))
    print("    * Configured PYTHON_BIN:    %s" % shell_cfg['PYTHON_BIN'])
    print("    * Configured PYTHON_RFDIFF: %s" % shell_cfg['PYTHON_RFDIFF'])
    print("    * Configured PYTHON_MPNN:   %s" % shell_cfg['PYTHON_MPNN'])
    print("    * RFdiffusion Repository:   %s (Exists: %s, Weights: %s)" % (shell_cfg['RFDIFF_DIR'], shell_cfg['rfdiff_repo_exists'], shell_cfg['rfdiff_weights_present']))
    print("    * ProteinMPNN Repository:   %s (Exists: %s, Script: %s)" % (shell_cfg['MPNN_DIR'], shell_cfg['mpnn_repo_exists'], shell_cfg['mpnn_script_exists']))

    # Locate and probe all Python interpreters
    py_cands = find_all_python_interpreters()
    print("\n[3] DETECTED PYTHON INTERPRETERS (%d found):" % len(py_cands))

    probed_pythons = []
    recommended_python = None

    for py in py_cands:
        data = inspect_python_interpreter(py)
        probed_pythons.append(data)
        ver = data.get("version", "Unknown")
        has_dc = "[OK]" if data.get("has_dataclasses") else "[MISSING]"
        has_np = "[OK]" if data.get("has_numpy") else "[MISSING]"
        has_torch = "[OK] (v%s, CUDA=%s)" % (data.get('torch_ver', ''), data.get('cuda_available')) if data.get("has_torch") else "[MISSING]"

        print("\n    ▶ Interpreter: %s" % py)
        print("      * Version:      %s" % ver)
        print("      * dataclasses:  %s" % has_dc)
        print("      * numpy:        %s (v%s)" % (has_np, data.get('numpy_ver', 'none')))
        print("      * PyTorch/CUDA: %s" % has_torch)

        # Evaluate if this python is suitable for AutoScaffold Core
        v_info = data.get("version_info", [])
        if len(v_info) >= 2 and (v_info[0] == 3 and v_info[1] >= 8) and data.get("has_dataclasses") and data.get("has_numpy"):
            if not recommended_python:
                recommended_python = py

    # Audit AutoScaffold internal modules with the current running Python
    curr_py = sys.executable
    print("\n[4] AUTOSCAFFOLD MODULE IMPORT CONNECTIVITY (Running with: %s):" % curr_py)
    mod_results = audit_autoscaffold_modules(curr_py)
    failed_modules = []
    for mod_name, res in mod_results.items():
        if res["status"] == "OK":
            print("    * %-28s : [OK]" % mod_name)
        else:
            print("    * %-28s : [FAIL] -> %s" % (mod_name, res['error']))
            failed_modules.append((mod_name, res["error"]))

    # If another python is available (like a virtualenv), test that one too
    if recommended_python and recommended_python != curr_py:
        print("\n[4b] MODULE IMPORT TEST WITH CANDIDATE VIRTUALENV (%s):" % recommended_python)
        rec_results = audit_autoscaffold_modules(recommended_python)
        for mod_name, res in rec_results.items():
            status_str = "[OK]" if res["status"] == "OK" else "[FAIL] -> %s" % res['error']
            print("    * %-28s : %s" % (mod_name, status_str))

    print("\n" + "=" * 80)
    print(" [ORCHESTAI ASSESSMENT DOSSIER & PROBLEM SUMMARY]")
    print("=" * 80)

    if failed_modules:
        print("\n[!] IDENTIFIED FAILURE POINTS (%d modules failed to load):" % len(failed_modules))
        for mod, err in failed_modules:
            print("    * %s: %s" % (mod, err))

        print("\n[!] ROOT CAUSE ANALYSIS:")
        if any("dataclasses" in err for _, err in failed_modules):
            print("    1. Host Python Version Incompatibility:")
            print("       Current interpreter is Python %s (%s)." % (sys.version.split()[0], curr_py))
            print("       Python 3.6 does NOT include 'dataclasses' in its standard library (added in 3.7+).")
            print("       AutoScaffold's modular pipeline ('chimera_stages') relies on 'dataclasses'.")
        if shell_cfg["CONDA_DEFAULT_ENV"] in ("None", "None (Host Base/System)"):
            print("    2. Missing Virtual Environment Activation:")
            print("       The pipeline was executed in the bare host system shell, not inside an activated")
            print("       virtual environment ('SE3nv', 'mlfold', or Python >= 3.8).")
            print("       When 'activate_env' was called in chimera_scaffold.sh, it found no active conda/micromamba")
            print("       environment and silently fell back to the host system's unaliased Python 3.6.")

        print("\n[!] ACTIONABLE REMEDIATION FOR ORCHESTAI / USER:")
        if recommended_python:
            print("    A. Switch PYTHON_BIN to discovered modern interpreter:")
            print("       export PYTHON_BIN=%s" % recommended_python)
        else:
            print("    A. Build or Activate Dedicated Virtual Environment:")
            print("       AutoScaffold requires Python >= 3.8 with scientific dependencies.")
            print("       Run: ./install.sh (Option 1 or Option 5) to build micromamba environments (SE3nv, mlfold).")
            print("    B. Or install dataclasses backport for Python 3.6 on host:")
            print("       pip3 install --user dataclasses numpy scipy")
            print("    C. Conda Environment Hook Safeguard:")
            print("       chimera_scaffold.sh must explicitly abort or warn if SE3nv activation fails")
            print("       instead of silently proceeding with an incompatible host Python 3.6.")
    else:
        print("\n[OK] All AutoScaffold modules loaded cleanly without syntax or import errors!")

    print("=" * 80)
    print("End of Diagnostic Output. Copy the text above to OrchestAI.\n")


if __name__ == "__main__":
    main()
