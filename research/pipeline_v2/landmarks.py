"""Single source of truth for landmark definitions and screening constants.

Every stage (proportions, motion, face) must import its COCO<->SMPL-X mapping,
canonical surface offsets and screening thresholds from here, so the three stages
cannot silently disagree about what a "shoulder" or a "valid observation" is.

Landmark classes
  - joint        : COCO keypoint corresponds to an SMPL-X joint centre
  - surface      : COCO keypoint is a body-surface landmark (acromion, malleolus,
                   toe/heel, fingertip); the model point is a joint plus a fixed
                   canonical offset taken from the SMPL-X rest mesh
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SMPLX_NEUTRAL = PROJECT / "lhmpp" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"

# COCO wholebody 133 layout
BODY_SLICE = slice(0, 17)
FOOT_SLICE = slice(17, 23)
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)

# COCO index -> SMPL-X joint index (joint centres)
JOINT_MAP = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}

# COCO index -> (SMPL-X joint, surface name for the offset table)
SURFACE_MAP = {
    17: (10, "LBigToe"), 18: (10, "LSmallToe"), 19: (10, "LHeel"),
    20: (11, "RBigToe"), 21: (11, "RSmallToe"), 22: (11, "RHeel"),
}

# COCO left-hand block: wrist + 5 fingers x (MCP, PIP, DIP, TIP); right block offset +21
HAND_BLOCK = 91
HAND_FINGER_JOINTS = {  # finger -> SMPL-X joint offsets from the wrist-adjacent base
    0: 25, 1: 28, 2: 31, 3: 34, 4: 37,
}
FINGERTIP_VERTICES = [5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022]

# Screening constants, shared by every stage
SCORE_MIN = 0.5
RANSAC_GATE_PX = 10.0
MIN_VIEWS = 3
MIN_RAY_ANGLE_DEG = 1.5


def group_ranges():
    return {"body": (0, 17), "feet": (17, 23), "face": (23, 91),
            "left_hand": (91, 112), "right_hand": (112, 133)}


def load_offsets():
    """Canonical surface offsets (SMPL-X rest mesh), keyed by surface name."""
    import numpy as np
    from smplx.vertex_ids import vertex_ids
    d = np.load(SMPLX_NEUTRAL, allow_pickle=True)
    v = np.asarray(d["v_template"], np.float64)
    jr = d["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else np.asarray(jr, np.float64)
    joints = jr @ v
    off = {}
    for j, name in SURFACE_MAP.values():
        vid = vertex_ids["smplx"][name]
        off[name] = v[vid] - joints[j]
    return off
