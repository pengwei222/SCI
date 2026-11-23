"""Recompute the added timing, resolution, and stopping summaries from archived CSVs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def read_csv(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def report(root):
    synthetic = list(read_csv(root / "synthetic/data/certificate_results.csv"))
    setup = list(read_csv(root / "synthetic/data/instances.csv"))
    costs = {
        field: 1000 * median(float(row[field]) for row in records)
        for records, fields in [
            (synthetic, ["vertex_total_seconds", "cheap_total_seconds",
                         "vertex_scale_seconds", "cheap_scale_seconds"]),
            (setup, ["common_preprocessing_seconds", "lift_setup_seconds"]),
        ]
        for field in fields
    }
    ratio = median(float(row["vertex_total_seconds"]) / float(row["cheap_total_seconds"])
                   for row in synthetic)
    cert = root / "expanded/certificate_extension/results"
    refs = list(read_csv(cert / "reference_weights.csv"))
    singles = list(read_csv(cert / "same_point_tightness.csv"))
    resolution = {}
    for eta in [0.001, 0.0001]:
        group = [r for r in singles if r["rule"] == "cheap" and float(r["eta"]) == eta
                 and r["reference_boundary"] == "False"]
        resolution[str(eta)] = sum(float(r["objective_gap_upper"]) - float(r["objective_gap_lower"])
                                   <= max(float(r["objective_gap_lower"]), 1e-16) / 20 for r in group)
    attempts = defaultdict(list)
    finite = defaultdict(list)
    faster_stop = 0
    end_to_end = defaultdict(list)
    for panel in ["recorded_existing", "recorded_additional"]:
        panel_path = root / "expanded/results" / panel
        for row in read_csv(panel_path / "timing.csv"):
            if row["method"] == "newton-cheap" and float(row["tolerance"]) == 0.001:
                a = int(row["certificate_attempts"])
                attempts[row["case_id"]].append(a)
                finite[row["case_id"]].append(a - int(row["certificate_failures"]))
        for row in read_csv(panel_path / "paired.csv"):
            if row["contrast"] == "certificate" and float(row["tolerance"]) == 0.001:
                faster_stop += float(row["checks_saved"]) > 0
                end_to_end[row["dataset"]].append(float(row["end_to_end_seconds"]))
    return {
        "synthetic_median_milliseconds": costs,
        "synthetic_median_paired_vertex_over_inexpensive_time": ratio,
        "recorded_references_meeting_1e-10": sum(float(r["reference_gap"]) <= 1e-10 for r in refs),
        "recorded_interior_references_resolved": resolution,
        "recorded_zero_over_zero_evaluations": sum(float(r["fw_gap"]) == float(r["effective_gap"]) == 0
                                                    for r in singles),
        "recorded_vertex_fallbacks": sum(r["certificate_failed"] == "True" and r["rule"] == "vertex"
                                         for r in singles),
        "online_inputs": len(attempts),
        "online_inputs_entering_quadratic_trigger": sum(median(a) > 0 for a in attempts.values()),
        "online_inputs_with_computed_quadratic_bound": sum(median(a) > 0 for a in finite.values()),
        "online_inputs_saving_checks": faster_stop,
        "online_median_milliseconds_by_dataset": {ds: 1000 * median(t) for ds,t in end_to_end.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted experiment root containing synthetic/ and expanded/")
    print(json.dumps(report(parser.parse_args().root), indent=2))
