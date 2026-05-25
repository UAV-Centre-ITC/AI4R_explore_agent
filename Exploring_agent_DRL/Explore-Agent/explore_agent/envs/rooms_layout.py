"""Rooms map and local layout loading for the exploration assignment."""

from pathlib import Path

import numpy as np


DEFAULT_ROOMS_WALLS = np.array([
    # upper block
    [300, 80, 1070, 80],
    [300, 80, 300, 420],
    [1070, 80, 1070, 330],
    [300, 250, 690, 250],
    [860, 250, 860, 420],
    [300, 420, 690, 420],
    [840, 420, 1070, 420],

    # lower-left and central block
    [120, 420, 300, 420],
    [120, 420, 120, 900],
    [120, 900, 620, 900],
    [620, 600, 620, 900],
    [120, 600, 190, 600],
    [415, 600, 620, 600],
    [620, 600, 1068, 600],
    [300, 600, 300, 900],

    # right block
    [1070, 330, 1400, 330],
    [1400, 330, 1400, 690],
    [1070, 690, 1400, 690],
    [1070, 330, 1070, 420],
    [1070, 600, 1070, 690],
], dtype=float)


DEFAULT_ROOMS_CHECKPOINTS = np.array([
    [445, 87, 445, 243],
    [640, 90, 640, 245],
    [690, 250, 853, 250],
    [866, 263, 1068, 88],
    [390, 260, 390, 416],
    [570, 260, 570, 420],
    [301, 429, 301, 594],
    [450, 430, 450, 590],
    [640, 430, 640, 590],
    [890, 430, 890, 590],
    [690, 420, 840, 420],
    [1070, 427, 1070, 591],
    [1080, 405, 1399, 405],
    [1078, 616, 1395, 616],
    [190, 600, 300, 600],
    [123, 770, 298, 770],
    [309, 729, 500, 893],
    [301, 599, 412, 599],
    [1064, 331, 867, 331],
    [419, 603, 615, 746],
], dtype=float)


def _candidate_local_layout_paths(override_path=None):
    if override_path:
        return [Path(override_path).expanduser()]
    return [
        Path.cwd() / ".local_tools" / "rooms_layout_edit.npz",
        Path.cwd().parent / ".local_tools" / "rooms_layout_edit.npz",
        Path(__file__).resolve().parents[4] / ".local_tools" / "rooms_layout_edit.npz",
    ]


def _validate_segments(name, array, layout_path):
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Invalid {name} array in {layout_path}: expected shape (N, 4), got {array.shape}")


def load_rooms_layout(override_path=""):
    """Return walls, checkpoints, and the source path for the rooms map.

    The committed defaults are used unless a local editor file exists. Local
    editor files live under .local_tools/ and are intentionally not committed.
    """
    for layout_path in _candidate_local_layout_paths(override_path):
        if not layout_path.exists():
            continue
        with np.load(layout_path) as data:
            walls = np.asarray(data["walls"], dtype=float)
            checkpoints = np.asarray(data["goals"], dtype=float)
        _validate_segments("walls", walls, layout_path)
        _validate_segments("goals", checkpoints, layout_path)
        return walls, checkpoints, str(layout_path)

    if override_path:
        raise FileNotFoundError(f"Rooms layout file not found: {override_path}")
    return DEFAULT_ROOMS_WALLS.copy(), DEFAULT_ROOMS_CHECKPOINTS.copy(), None
