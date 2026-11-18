"""Path resolution utilities for environment-independent filesystem operations.

This module provides functions to locate the project root directory and resolve
relative paths into absolute paths, regardless of the current working directory
from which the script or module is executed.
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The root is discovered by traversing upwards from this file's location
    until an anchor (e.g., .git, configs, or requirements.txt) is found.
    Defaults to two directories up from this file (src/utils -> project root).

    Returns:
        Path: Absolute path to the repository root.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "configs").exists() or (parent / "requirements.txt").exists():
            return parent
    # Fallback: src/utils/path_helpers.py -> parents[2] is root
    return current.parents[2]


PROJECT_ROOT: Path = get_project_root()


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a relative or absolute path against the project root.

    Args:
        path: Path string or Path object. If relative, resolved against PROJECT_ROOT.
              If already absolute, returned as a resolved Path.

    Returns:
        Path: Fully resolved absolute Path.
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()

