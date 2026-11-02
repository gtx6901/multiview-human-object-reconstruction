#!/usr/bin/env python
"""Narrow libigl compatibility for the author's `signed_distance` call.

libigl 2.6.x returns `(S, I, C, N)` (normals fourth).  The author's
`gen_data/gen_weight_volume.py` unpacks only `(dists, face_id, closest_pts)`.
This wrapper preserves the real signed distances, face indices and closest
points unchanged and only drops the trailing normals value.  It does not
compute or fabricate a distance field.

Version evidence is recorded by resume_weight_volume.py at run time
(`igl.__version__` and the observed tuple length).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Tuple

import igl

_original: Callable[..., Any] = igl.signed_distance
last_arity: Optional[int] = None
last_version: Optional[str] = getattr(igl, "__version__", None)


def signed_distance(*args: Any, **kwargs: Any):
    """Call the real libigl signed_distance and return its first three outputs.

    libigl 2.6.3 returns four values: signed distance, face index, closest point,
    vertex/face normals.  Three-value builds are passed through unchanged.
    """
    global last_arity
    result = _original(*args, **kwargs)
    if not isinstance(result, (tuple, list)):
        raise TypeError(
            "libigl.signed_distance returned a non-tuple; cannot adapt arity: "
            f"{type(result)}"
        )
    last_arity = len(result)
    if last_arity == 3:
        return tuple(result)
    if last_arity == 4:
        # S, I, C are the author's expected values; N is discarded only.
        return result[0], result[1], result[2]
    raise ValueError(
        f"unsupported libigl.signed_distance output arity {last_arity}; "
        "expected 3 or 4 (this adapter does not fabricate outputs)"
    )


def install() -> None:
    """Optional process-local patch for callers that cannot use the wrapper."""
    igl.signed_distance = signed_distance


def evidence() -> dict:
    try:
        sig = str(inspect.signature(_original))
    except Exception:
        sig = "unavailable"
    return {
        "libigl_version": getattr(igl, "__version__", None),
        "signed_distance_signature": sig,
        "observed_output_arity": last_arity,
        "adapted": "returns first three (S,I,C); trailing normals dropped",
        "source": "resume_weight_volume.py replaces only the call name with "
                  "libigl_compat.signed_distance; no monkeypatch required",
    }
