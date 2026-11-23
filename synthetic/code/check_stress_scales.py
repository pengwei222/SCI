"""Recheck stored stress vertex scales independently using exact original blocks."""
from csv import DictReader, DictWriter
from fractions import Fraction as F
from pathlib import Path
import json

import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'


def positive_semidefinite(x):
    """Independent exact rational PSD test, including singular remainders."""
    n = len(x)
    for _ in range(n):
        if any(x[j][j] < 0 for j in range(len(x))):
            return False
        pivot = next((j for j in range(len(x)) if x[j][j] > 0), None)
        if pivot is None:
            return all(a == 0 for row in x for a in row)
        keep = [j for j in range(len(x)) if j != pivot]
        x = [[x[i][j] - x[i][pivot] * x[pivot][j] / x[pivot][pivot]
              for j in keep] for i in keep]
    return True


def check_difference(factors, weights, b, other):
    """Take the Schur complement of the original rationalized arrowhead blocks.

    Scale auxiliary coordinates by sqrt((1-a)/a), to obtain rational blocks
    and keep the large top-left and correction terms algebraically separate.
    """
    d = len(factors[0][0][1])
    top = [[F(0) for _ in range(d)] for _ in range(d)]
    scaled = b - 1
    if scaled <= 0:
        raise ValueError("This regression suite expects b>1.")
    for j, terms in enumerate(factors):
        coefficient = b * weights[j] - other[j]
        for a, z in terms:
            if a == 0:
                first, correction = scaled, F(0)
            elif a == 1:
                first, correction = coefficient, F(0)
            else:
                first = scaled / (1 - a)
                diagonal = (1 - a) / a * (scaled * a + coefficient * (1 - a))
                if diagonal < 0:
                    return False
                if diagonal == 0 and any(z):
                    return False
                correction = scaled**2 / diagonal if diagonal else F(0)
            for l in range(d):
                for m in range(l + 1):
                    zz = z[l] * z[m]
                    top[l][m] += first * zz - correction * zz
    return positive_semidefinite([[top[max(i, j)][min(i, j)] for j in range(d)] for i in range(d)])


def unit(size, index):
    return tuple(int(j == index) for j in range(size))


def check_input(factors, weights, b, i):
    return check_difference(factors, weights, b, unit(len(weights), i))


def confirm():
    with (DATA/'stress_vertex_audit.csv').open() as stream:
        audit = list(DictReader(stream))
    assert len(audit) == 128
    factors = {}
    with np.load(DATA/'stress_factor_inputs.npz') as frozen:
        for i in range(32):
            factors[i] = [
                [(F(float(a)), tuple(F(float(t)) for t in z)) for a, z in zip(aa, zz.T)]
                for aa, zz in zip(frozen[f'a_{i}'], frozen[f'Z_{i}'])]
    out = []
    for row in audit:
        c = [F(x) for x in json.loads(row['weights'])]
        total = sum(c)
        c = [x / total for x in c]
        assert min(c) >= 0 and sum(c) == 1
        factor = factors[int(row['case'])]
        b = F(float.fromhex(row['b_hex']))
        assert b == F(float(row['b']))
        valid = all(check_input(factor, c, b, i) for i in range(len(c)))
        assert valid, (row['case'], row['mode'], row['eta'])
        if row['legacy_status'] == 'success':
            old = F(float(row['legacy_b']))
            invalid_vertices = [i + 1 for i in range(len(c)) if not check_input(factor, c, old, i)]
            old_check = 'fails_on_reproduced_center' if invalid_vertices else 'passes_on_reproduced_center'
        else:
            invalid_vertices, old_check = [], 'reported_failure'
        out.append(dict(case=row['case'], mode=row['mode'], eta=row['eta'],
                        verification='exact_all_vertices', vertex_count=len(c),
                        legacy_check=old_check, legacy_invalid_vertices=json.dumps(invalid_vertices)))
    with (DATA/'stress_vertex_verification.csv').open('w', newline='') as stream:
        writer = DictWriter(stream, fieldnames=out[0])
        writer.writeheader()
        writer.writerows(out)
    print('Exact all-vertex verification:', len(out), 'cases,', sum(int(x['vertex_count']) for x in out), 'LMIs')
    for key in sorted({r['legacy_check'] for r in out}):
        print(key, sum(r['legacy_check'] == key for r in out))
    return out


if __name__ == '__main__':
    confirm()
