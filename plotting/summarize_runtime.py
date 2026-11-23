"""Summarize the archived end-to-end comparisons on the same robot inputs."""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


root = Path(__file__).resolve().parents[1] / "expanded/results"
samples = defaultdict(dict)
for panel in ("recorded_existing", "recorded_additional"):
    with (root / panel / "paired.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["tolerance"]) != 1e-3:
                continue
            group = samples[int(row["dataset"]), row["contrast"]]
            case = row["case_id"]
            if case in group or row["paired_success"] != "True":
                raise ValueError(case)
            group[case] = row

out = csv.writer(sys.stdout, lineterminator="\n")
contrasts = ("matrix", "newton", "complete", "certificate")
out.writerow(("dataset", "inputs", *contrasts, "inputs_saving_checks",
              "complete_faster_inputs", "median_complete_ms"))
for dataset, n in zip((2, 3, 4, 5), (364, 882, 271, 453)):
    full = samples[dataset, "complete"]
    if len(full) != n or any(samples[dataset, c].keys() != full.keys() for c in contrasts):
        raise ValueError(f"Mismatched input identities for dataset {dataset}")
    ratios = [median(float(r["paired_speedup"]) for r in samples[dataset, c].values())
              for c in contrasts]
    saved = sum(float(r["checks_saved"]) > 0 for r in samples[dataset, "certificate"].values())
    faster = sum(float(r["paired_speedup"]) > 1 for r in full.values())
    runtime = 1000 * median(float(r["end_to_end_seconds"]) for r in full.values())
    out.writerow((dataset, n, *ratios, saved, faster, runtime))
