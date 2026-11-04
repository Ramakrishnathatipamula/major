"""
Portable dataset path helpers.

Manifests must not embed machine-specific absolute paths. Project-owned files are
stored relative to the project root, while files in the user's Kaggle cache are
stored with a ``~`` prefix. Both forms are resolved back to absolute paths here.
"""

import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KAGGLE_PLANTVILLAGE_HANDLE = "mohitsingh1804/plantvillage"
_PLANTVILLAGE_CACHE_RELATIVE = Path(
    ".cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1/PlantVillage"
)


def to_portable_path(path: Union[str, Path], root: Union[str, Path] = PROJECT_ROOT) -> str:
    """Return a portable string for a path.

    Paths inside the project become project-relative POSIX strings. Paths inside
    the user's home directory become ``~/...``. Other absolute paths are returned
    unchanged because they cannot be represented portably.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()

    resolved = candidate.resolve()
    root_resolved = Path(root).resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        pass

    home_resolved = Path.home().resolve()
    try:
        return "~/" + resolved.relative_to(home_resolved).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_dataset_path(
    path_value: Union[str, Path],
    root: Union[str, Path] = PROJECT_ROOT,
) -> Path:
    """Resolve a manifest path (project-relative, ``~``-based, or absolute)."""
    if path_value is None or not str(path_value).strip():
        raise ValueError("Dataset path is empty")

    candidate = Path(str(path_value)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return candidate.resolve()


def get_plantvillage_root() -> Path:
    """Locate the cached PlantVillage dataset without hardcoding a user path."""
    env_override = os.getenv("PLANTVILLAGE_ROOT")
    if env_override:
        return Path(env_override).expanduser()

    versions_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / KAGGLE_PLANTVILLAGE_HANDLE / "versions"
    if versions_dir.exists():
        candidates = sorted(versions_dir.glob("*/PlantVillage"), reverse=True)
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return Path.home() / _PLANTVILLAGE_CACHE_RELATIVE
