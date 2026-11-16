"""Shared body, foot and hand landmark definitions."""

from pathlib import Path

import numpy as np
from hoi.runtime import runtime_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMPLX = (
    Path(runtime_settings()["lhm_root"])
    / "pretrained_models"
    / "human_model_files"
    / "smplx"
    / "SMPLX_NEUTRAL.npz"
)

BODY = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
WORLD_HUBER_SCALE = 0.02

FOOT_VERTEX = {
    17: ("LBigToe", 10),
    18: ("LSmallToe", 10),
    19: ("LHeel", 10),
    20: ("RBigToe", 11),
    21: ("RSmallToe", 11),
    22: ("RHeel", 11),
}

HAND_OFFSET = 91

HAND_BASE = [37, 25, 28, 34, 31]

TIP_VID = [5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022]


def canonical_offsets(smplx_path: Path = SMPLX):
    """Current fit_motion canonical toe/heel/fingertip offsets, patched hand order."""
    d = np.load(Path(smplx_path), allow_pickle=True)
    v = np.asarray(d["v_template"], np.float64)
    jr = d["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else np.asarray(jr, np.float64)
    J = jr @ v
    from smplx.vertex_ids import vertex_ids

    ids = vertex_ids["smplx"]
    import torch

    feet = []
    for coco, (name, joint) in FOOT_VERTEX.items():
        feet.append(
            (int(coco), int(joint), torch.tensor(v[ids[name]] - J[joint], dtype=torch.float32))
        )
    tips = []
    for side in range(2):
        for f in range(5):
            distal = HAND_BASE[f] + side * 15 + 2
            tips.append(
                (
                    HAND_OFFSET + side * 21 + 4 * f + 4,
                    int(distal),
                    torch.tensor(v[TIP_VID[side * 5 + f]] - J[distal], dtype=torch.float32),
                )
            )
    return feet, tips
