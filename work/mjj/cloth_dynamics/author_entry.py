#!/usr/bin/env python
"""Thin launcher for the cloned author modules.

All path/model/expression/extension adaptation lives in runtime_backend.py.
This avoids the old bug where `chdir(author)` made the config data_dir
`../cloth_dynamics/...` resolve under cloth_dynamics/cloth_dynamics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime_backend import run_author_module  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True,
                    choices=["gen_data.gen_pos_maps", "main_template"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--author", default=str(HERE / "author"))
    a = ap.parse_args()
    return int(run_author_module(a.module, Path(a.config), Path(a.author)))


if __name__ == "__main__":
    raise SystemExit(main())
