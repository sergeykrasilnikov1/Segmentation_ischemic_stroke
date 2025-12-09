"""Utility functions for brain stroke segmentation."""

import cv2
import numpy as np
from pathlib import Path


def extract_red_mask_from_path(mask_path: Path | str | None, width: int, height: int) -> np.ndarray | None:
    """
    Read a color overlay image and extract the red-filled lesion as a binary mask.

    Args:
        mask_path: Path to the overlay image
        width: Target width for the mask
        height: Target height for the mask

    Returns:
        Binary mask array or None if path is missing/unreadable
    """
    if mask_path is None:
        return None
    if isinstance(mask_path, str):
        mask_path = Path(mask_path)
    if not mask_path.exists():
        return None

    overlay = cv2.imread(str(mask_path))
    if overlay is None:
        return None

    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def get_git_commit_id() -> str:
    """Get current git commit ID."""
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return "unknown"

