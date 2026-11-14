"""Run the shared reconstruction pipeline from a dataset configuration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hoi.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
