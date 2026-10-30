"""Portable entry point for the EXP-004 checker shipped inside solution/tests."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "solution/tests/test_launch_trim.py"),
                   run_name="__main__")
