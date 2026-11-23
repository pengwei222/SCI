"""A stable spectral proposal and an exact scale check for factored SCI inputs."""
from fractions import Fraction
from math import isfinite, lcm, nextafter, inf

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular


def rational_weights(values):
    """Exact normalized interpretation of a feasible floating-point center."""
    weights = tuple(Fraction(float(value)) for value in values)
    total = sum(weights)
    if min(weights) < 0 or total <= 0:
        raise ValueError("A nonnegative center with positive sum is required.")
    return tuple(value / total for value in weights)


def upward_float(value):
    candidate = float(value)
    if not isfinite(candidate):
        raise OverflowError("Scale is outside floating-point range.")
    if Fraction(candidate) < value:
        candidate = nextafter(candidate, inf)
    return candidate


def positive_definite(matrix):
    """Exact symmetric, fraction-free elimination on an integer multiple."""
    size = len(matrix)
    denominator = lcm(*(x.denominator for row in matrix for x in row))
    a = [[int(denominator * x) for x in row] for row in matrix]
    previous = 1
    for k in range(size):
        pivot = a[k][k]
        if pivot <= 0:
            return False
        for i in range(k + 1, size):
            for j in range(i, size):
                quotient, remainder = divmod(pivot * a[i][j] - a[i][k] * a[k][j], previous)
                if remainder:
                    raise ArithmeticError("Nonexact division in symmetric elimination.")
                a[i][j] = a[j][i] = quotient
        previous = pivot
    return True


class FactoredScaleCheck:
    """Treat binary input factors Z and split eigenvalues a as exact rationals."""
    def __init__(self, model):
        self.d = model.d
        self.s = model.s
        self.terms = []
        for Z, eigenvalues in zip(model.Zs, model.aas):
            entries = []
            for z, alpha in zip(Z.T, eigenvalues):
                a = Fraction(float(alpha))
                if not 0 <= a <= 1:
                    raise ValueError("Invalid split eigenvalue.")
                z = [Fraction(float(t)) for t in z]
                outer = [[z[i] * z[j] for j in range(i + 1)] for i in range(self.d)]
                entries.append((a, outer))
            self.terms.append(entries)

    def _check_difference(self, positive, negative, b):
        """Test strict positivity of b M(c)-M(e_vertex) by its Schur complement."""
        if b <= 1:
            return False
        k = b - 1
        lower = [[Fraction(0) for _ in range(i + 1)] for i in range(self.d)]
        for i, terms in enumerate(self.terms):
            v = b * positive[i] - negative[i]
            for a, outer in terms:
                if a == 0:
                    coefficient = k
                elif a == 1:
                    coefficient = v
                else:
                    diagonal = k * a + (1 - a) * v
                    if diagonal <= 0:
                        return False
                    coefficient = k * v / diagonal
                for row_index, row in enumerate(outer):
                    for j, element in enumerate(row):
                        lower[row_index][j] += coefficient * element
        matrix = [[lower[max(i, j)][min(i, j)] for j in range(self.d)]
                  for i in range(self.d)]
        return positive_definite(matrix)

    def _vertices(self):
        for vertex in range(self.s):
            yield tuple(int(i == vertex) for i in range(self.s))

    def verify(self, c, b):
        return all(self._check_difference(c, vertex, b) for vertex in self._vertices())

    def verify_lower(self, c, a):
        if not 0 < a < 1:
            return False
        return all(self._check_difference(vertex, c, 1 / a) for vertex in self._vertices())



def stable_spectral_proposal(model, c, both=False):
    """Schur-eliminate the center before forming the normalized vertex change."""
    J, derivatives, _ = model.info(c, 1)
    chol = cholesky(J, lower=True, check_finite=False)
    dd = []
    mixed = []
    for ci, Z, aa, derivative in zip(c, model.Zs, model.aas, derivatives):
        left = solve_triangular(chol, derivative, lower=True, check_finite=False)
        dd.append(solve_triangular(chol, left.T, lower=True, check_finite=False).T)
        mask = (aa > 0) & (aa < 1)
        a = aa[mask]
        den = a + (1 - a) * ci
        z = solve_triangular(chol, Z[:, mask], lower=True, check_finite=False)
        cross = -z * (np.sqrt(a * (1 - a)) / den**1.5)
        mixed.append((cross, (1 - a) / den))
    n = model.d + sum(len(diagonal) for _, diagonal in mixed)
    b = 1.
    a = 1.
    for vertex in range(model.s):
        direction = -np.array(c, copy=True)
        direction[vertex] += 1
        normalized = np.zeros((n, n))
        block = np.einsum('i,ijk->jk', direction, dd)
        normalized[:model.d, :model.d] = (block + block.T) / 2
        start = model.d
        for weight, (cross, diagonal) in zip(direction, mixed):
            end = start + len(diagonal)
            normalized[:model.d, start:end] = weight * cross
            normalized[start:end, :model.d] = (weight * cross).T
            normalized[start:end, start:end] = np.diag(weight * diagonal)
            start = end
        top = eigh(normalized, eigvals_only=True, subset_by_index=[n - 1, n - 1],
                   check_finite=False)[0]
        b = max(b, 1. + top)
        if both:
            bottom = eigh(normalized, eigvals_only=True, subset_by_index=[0, 0],
                          check_finite=False)[0]
            a = min(a, 1. + bottom)
    return a, b


def verified_scales(model, checker, values, both=False):
    c = rational_weights(values)
    floats = np.array([float(x) for x in c])
    try:
        lower_proposal, proposal = stable_spectral_proposal(model, floats, both)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        lower_proposal, proposal = 0., float('nan')
    upper = float(proposal * (1 + 1e-9))
    if isfinite(upper) and checker.verify(c, Fraction(upper)):
        status = "vertex_exact"
    elif min(c) > 0:
        upper = upward_float(1 / min(c))
        status = "analytic_interior_fallback"
    else:
        # Strict comparisons eventually hold because every vertex lift is SPD.
        upper = max(2., float(proposal) if isfinite(proposal) else 2.)
        while isfinite(upper) and not checker.verify(c, Fraction(upper)):
            upper *= 2
        if not isfinite(upper):
            raise ArithmeticError("No finite verified upper scale was established.")
        status = "vertex_exact_doubled"
    lower = 0.
    if both:
        lower = (lower_proposal / (1 + 1e-9)
                 if isfinite(lower_proposal) and lower_proposal > 0 else 2.**-52)
        while lower > 0 and not checker.verify_lower(c, Fraction(float(lower))):
            lower *= .5
        if lower == 0:
            raise ArithmeticError("No positive representable lower scale was verified.")
    return lower, upper, {"scale_status": status, "spectral_proposal": proposal}
