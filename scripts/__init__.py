"""AutoScaffold scripts package."""
from scripts.version import (
    __version__,
    __version_info__,
    get_version,
    get_git_sync_status,
    clear_git_sync_cache,
)

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "get_git_sync_status",
    "clear_git_sync_cache",
]
