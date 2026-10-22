"""Run the checkout's measured demo and persist its JSON receipt."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "demos" / "demo.py"),
                   run_name="__main__")
