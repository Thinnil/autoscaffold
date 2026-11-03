"""
scripts/version.py
Single Source of Truth for AutoScaffold semantic versioning and repository synchronization.
"""

import os
import re
import shutil
import subprocess
from typing import Optional, Tuple

_RE_SHELL_INJECTION = re.compile(r"[;&|`><\n\r\$]")

__version__ = "2.2.0"
__version_info__: Tuple[int, int, int] = (2, 2, 0)
__author__ = "Thinnil / AutoScaffold Team"
__license__ = "MIT"

_GIT_SYNC_CACHE: dict = {}


def clear_git_sync_cache() -> None:
    """Clears the in-memory git synchronization status session cache."""
    _GIT_SYNC_CACHE.clear()


def get_version() -> str:
    """Returns the current semantic version string."""
    return __version__


def get_git_sync_status(repo_dir: Optional[str] = None, refresh: bool = False) -> str:
    """
    Non-blocking query checking if the local git repository is in sync with upstream origin/main.
    Results are cached in-memory for O(1) lookups to eliminate redundant git subprocess spawns.
    Returns status string such as 'Up-to-date with origin/main', 'Update Available', or 'Local Installation'.
    """
    cache_key = os.path.abspath(str(repo_dir).strip()) if repo_dir else "default"
    if not refresh and cache_key in _GIT_SYNC_CACHE:
        return _GIT_SYNC_CACHE[cache_key]

    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        repo_dir = os.path.realpath(os.path.abspath(str(repo_dir).strip()))
        if _RE_SHELL_INJECTION.search(repo_dir):
            res = f"Local Installation (v{__version__})"
            _GIT_SYNC_CACHE[cache_key] = res
            return res

    if not shutil.which("git"):
        res = f"Local Installation (v{__version__})"
        _GIT_SYNC_CACHE[cache_key] = res
        return res

    git_dir = os.path.join(repo_dir, ".git")
    if not os.path.isdir(git_dir):
        res = f"Local Installation (v{__version__})"
        _GIT_SYNC_CACHE[cache_key] = res
        return res

    try:
        # Check local HEAD
        local_res = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            shell=False,
        )
        if local_res.returncode != 0:
            res = f"Local Installation (v{__version__})"
            _GIT_SYNC_CACHE[cache_key] = res
            return res
        local_sha = local_res.stdout.strip()

        # Check origin/main
        remote_res = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "origin/main"],
            capture_output=True,
            text=True,
            timeout=2.0,
            shell=False,
        )
        if remote_res.returncode != 0:
            res = f"Local Installation (v{__version__})"
            _GIT_SYNC_CACHE[cache_key] = res
            return res
        remote_sha = remote_res.stdout.strip()

        if local_sha and remote_sha:
            if local_sha == remote_sha:
                res = "Up-to-date with origin/main"
            else:
                res = "Update Available (run Option [7] to update)"
            _GIT_SYNC_CACHE[cache_key] = res
            return res
    except Exception:
        pass

    res = f"Local Installation (v{__version__})"
    _GIT_SYNC_CACHE[cache_key] = res
    return res
