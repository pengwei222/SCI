"""Count same-candidate certificates meeting each normalized pose-trace tolerance.

Only the archived effective gaps are used, including the first-order fallback.
The two quadratic methods are matched by dataset, initialization, and epoch.
"""
import argparse
import csv
import math
import sys
from pathlib import Path


DEFAULT_DATA = (Path(__file__).resolve().parents[1] /
                "expanded/certificate_extension/results/same_point_tightness.csv")


def read_candidates(path, eta):
    candidates = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["eta"]) != eta:
                continue
            key = tuple(int(row[k]) for k in ("dataset", "seed", "epoch"))
            rule = row["rule"]
            fw = float(row["fw_gap"])
            gap = float(row["effective_gap"])
            group = {"True": "boundary", "False": "interior"}[row["reference_boundary"]]
            candidate = candidates.setdefault(key, {"dataset": key[0], "group": group, "fw": fw})
            if rule not in ("cheap", "vertex") or rule in candidate:
                raise ValueError(f"Unexpected or repeated rule {rule} at {key}")
            if fw != candidate["fw"] or group != candidate["group"]:
                raise ValueError(f"Mismatched first-order gap or group at {key}")
            if not (math.isfinite(fw) and 0 <= gap <= fw):
                raise ValueError(f"Invalid effective gap at {key}")
            candidate[rule] = gap
    if not candidates:
        raise ValueError(f"No saved candidates at eta={eta}")
    if any("cheap" not in c or "vertex" not in c for c in candidates.values()):
        raise ValueError(f"Unmatched candidates at eta={eta}")
    return list(candidates.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--eta", type=float, default=.001)
    args = parser.parse_args()
    candidates = read_candidates(args.data, args.eta)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("eta", "dataset", "reference_group", "tolerance", "n",
                     "fw", "cheap", "vertex", "vertex_additional_to_fw"))
    for dataset in ("all", *sorted({c["dataset"] for c in candidates})):
        for group in ("all", "interior", "boundary"):
            selected = [c for c in candidates if (dataset == "all" or c["dataset"] == dataset)
                        and (group == "all" or c["group"] == group)]
            for tol in (1e-3, 1e-4, 1e-5):
                counts = [sum(c[m] <= tol for c in selected) for m in ("fw", "cheap", "vertex")]
                additional = sum(c["vertex"] <= tol < c["fw"] for c in selected)
                writer.writerow((args.eta, dataset, group, tol, len(selected), *counts, additional))


if __name__ == "__main__":
    main()
