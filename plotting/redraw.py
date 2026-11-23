"""Redraw the three figures using only the data in this package."""
import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copyfile


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here.parent / "redrawn")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output.resolve()
    archive = here.parent

    def run(script, *args):
        subprocess.run([sys.executable, str(here/script), *map(str, args)], check=True)

    run("plot_synthetic_gain.py", "--archive", archive, "--output", dest/"synthetic_gain.pdf")
    run("plot_robot_gain.py", "--archive", archive, "--output", dest/"robot_gain_distribution.pdf")
    with TemporaryDirectory() as temp:
        run("plot_robot_replay.py", "--archive", archive, "--out", temp)
        copyfile(Path(temp)/"robot_replay_window.pdf", dest/"robot_replay.pdf")


if __name__ == "__main__":
    main()
