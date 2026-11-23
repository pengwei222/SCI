"""Exact-arithmetic a posteriori SCI certificate checker.

Inputs are rational matrices; Python floats are interpreted as their EXACT
binary values, not rounded or projected to a PSD cone. The checker uses
SymPy Rational arithmetic. It is a validation implementation, not an efficient
large-scale optimizer and not a proof-assistant formalization.

A logdet objective gap is certified without evaluating a logarithm: the
constant f(c) cancels, and its first and second derivatives are rational.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal, localcontext, ROUND_CEILING
from typing import Sequence, Any
import math
import sympy as sp


def rational(x: Any) -> sp.Rational:
    if isinstance(x, sp.Rational):
        return x
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError('Nonfinite input.')
        n, d = x.as_integer_ratio()
        return sp.Rational(n, d)
    return sp.Rational(x)


def matrix(x: Any) -> sp.Matrix:
    x = sp.Matrix(x)
    return x.applyfunc(rational)


def scalar(x: sp.Matrix) -> sp.Rational:
    return x[0, 0]


def check_symmetric(x: sp.Matrix) -> None:
    if x.rows != x.cols or x != x.T:
        raise ValueError('An exactly symmetric square matrix is required.')


def psd_factor(x: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """Prove x >= 0 and factor x=U diag(d) U.T, with rational positive d.

    Uses diagonal pivoted rank-one elimination. If all residual diagonal
    entries are zero, PSD requires the WHOLE residual to be zero.
    No eigenvalue or rank tolerance is used.
    """
    check_symmetric(x)
    residual = x.copy()
    cols, vals = [], []
    for _ in range(x.rows):
        if any(residual[j, j] < 0 for j in range(x.rows)):
            raise ValueError('Input is not positive semidefinite.')
        pivots = [j for j in range(x.rows) if residual[j, j] > 0]
        if not pivots:
            if residual != sp.zeros(x.rows):
                raise ValueError('Zero diagonal with nonzero off-diagonal: not PSD.')
            break
        j = pivots[0]
        d = residual[j, j]
        u = residual[:, j] / d
        cols.append(u)
        vals.append(d)
        residual = residual - d * u * u.T
    if residual != sp.zeros(x.rows):
        raise AssertionError('Exact PSD elimination did not finish.')
    U = sp.Matrix.hstack(*cols) if cols else sp.zeros(x.rows, 0)
    D = sp.diag(*vals) if vals else sp.zeros(0, 0)
    if U * D * U.T != x:
        raise AssertionError('Invalid exact PSD factorization.')
    return U, D


def is_spd(x: sp.Matrix) -> bool:
    """Exact Sylvester/LDL test, without eigenvalues or square roots."""
    if x.rows != x.cols or x != x.T:
        return False
    s = x.copy()
    for k in range(x.rows):
        p = s[k, k]
        if p <= 0:
            return False
        for i in range(k+1, x.rows):
            for j in range(i, x.rows):
                z = s[i, j] - s[i, k] * s[k, j] / p
                s[i, j] = z
                s[j, i] = z
    return True


def kernel_basis_and_complement(x: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    n = x.rows
    cols = x.nullspace()
    N = sp.Matrix.hstack(*cols) if cols else sp.zeros(n, 0)
    V = N.copy()
    Rcols = []
    for j in range(n):
        e = sp.eye(n)[:, j]
        candidate = V.row_join(e)
        if candidate.rank() > V.cols:
            V = candidate
            Rcols.append(e)
    if V.cols != n:
        raise AssertionError('Unable to complete basis.')
    R = sp.Matrix.hstack(*Rcols) if Rcols else sp.zeros(n, 0)
    return N, R


@dataclass
class RationalBranch:
    A: sp.Matrix
    B: sp.Matrix
    # Stable branch: J(t)=J0+t L(Ared+t Cred)^(-1)L.T.
    J0: sp.Matrix
    L: sp.Matrix
    Ared: sp.Matrix
    Cred: sp.Matrix
    # Rational affine lift: J(t)=R0+t R1-F(D0+t D1)^(-1)F.T.
    R0: sp.Matrix
    R1: sp.Matrix
    F: sp.Matrix
    D0: sp.Matrix
    D1: sp.Matrix

    @classmethod
    def build(cls, A: Any, B: Any) -> 'RationalBranch':
        A, B = matrix(A), matrix(B)
        if A.shape != B.shape:
            raise ValueError('Covariance shapes differ.')
        psd_factor(A); psd_factor(B)
        if not is_spd(A+B):
            raise ValueError('A+B must be positive definite.')
        d = A.rows
        # A-kernel elimination for a cancellation-free information oracle.
        N, R = kernel_basis_and_complement(A)
        if N.cols:
            B0 = N.T * B * N
            assert is_spd(B0)
            B0inv = B0.inv()
            J0 = N * B0inv * N.T
            L = R - N * B0inv * N.T * B * R
            Cred = R.T*B*R - R.T*B*N*B0inv*N.T*B*R
        else:
            J0 = sp.zeros(d)
            L = R
            Cred = R.T*B*R
        Ared = R.T*A*R
        if R.cols:
            assert is_spd(Ared)
            psd_factor(Cred)
        # B-kernel elimination for the rational, square-root-free lift.
        NB, RB = kernel_basis_and_complement(B)
        if NB.cols:
            A0 = NB.T*A*NB
            assert is_spd(A0)
            A0inv = A0.inv()
            R1 = NB*A0inv*NB.T
            LB = RB - NB*A0inv*NB.T*A*RB
            C = RB.T*A*RB - RB.T*A*NB*A0inv*NB.T*A*RB
        else:
            R1 = sp.zeros(d)
            LB = RB
            C = RB.T*A*RB
        if RB.cols:
            Bred = RB.T*B*RB
            assert is_spd(Bred)
            Binv = Bred.inv()
            U, DD = psd_factor(C)
            R0 = LB*Binv*LB.T
            F = LB*Binv*U
            D0 = U.T*Binv*U
            D1 = DD.inv() if DD.rows else sp.zeros(0,0)
            if DD.rows:
                assert is_spd(D0)
        else:
            R0 = sp.zeros(d)
            F = sp.zeros(d,0)
            D0 = D1 = sp.zeros(0,0)
        return cls(A,B,J0,L,Ared,Cred,R0,R1,F,D0,D1)

    def evaluate(self, t: Any) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        t = rational(t)
        if not 0 <= t <= 1:
            raise ValueError('Branch weight outside [0,1].')
        if not self.Ared.rows:
            return self.J0, sp.zeros(self.A.rows), sp.zeros(self.A.rows)
        R = (self.Ared+t*self.Cred).inv()
        C, L = self.Cred, self.L
        J = self.J0 + t*L*R*L.T
        D = L*(R-t*R*C*R)*L.T
        F = L*(-2*R*C*R+2*t*R*C*R*C*R)*L.T
        return J, D, F


class RationalSCI:
    def __init__(self, A: Sequence[Any], B: Sequence[Any], W: Any | None = None):
        if len(A) != len(B) or len(A) < 2:
            raise ValueError('At least two matching branches are required.')
        self.branches = [RationalBranch.build(a,b) for a,b in zip(A,B)]
        self.s = len(A)
        self.d = self.branches[0].A.rows
        if any(q.A.shape != (self.d,self.d) for q in self.branches):
            raise ValueError('Different state dimensions.')
        self.W = sp.eye(self.d) if W is None else matrix(W)
        psd_factor(self.W)
        if self.W.shape != (self.d,self.d):
            raise ValueError('Wrong objective weight size.')
        n = self.d + sum(q.D0.rows for q in self.branches)
        self.M0 = sp.zeros(n)
        self.Mi = [sp.zeros(n) for _ in range(self.s)]
        p = self.d
        for i,q in enumerate(self.branches):
            self.M0[:self.d,:self.d] += q.R0
            self.Mi[i][:self.d,:self.d] += q.R1
            r = q.D0.rows
            if r:
                self.M0[:self.d,p:p+r] = q.F
                self.M0[p:p+r,:self.d] = q.F.T
                self.M0[p:p+r,p:p+r] = q.D0
                self.Mi[i][p:p+r,p:p+r] = q.D1
                p += r
        self.vertices = [self.M0+m for m in self.Mi]
        if not all(is_spd(v) for v in self.vertices):
            raise AssertionError('Exact vertex lift is not SPD.')
        self.n = n

    def weight(self, w: Any) -> sp.Matrix:
        w = matrix(w)
        if w.shape != (self.s,1) or any(x < 0 for x in w) or sum(w) != 1:
            raise ValueError('Weights must be exactly on the simplex.')
        return w

    def lift(self, w: Any) -> sp.Matrix:
        w = self.weight(w)
        return self.M0 + sum((wi*mi for wi,mi in zip(w,self.Mi)), sp.zeros(self.n))

    def oracle(self, w: Any, mode: str = 'logdet'):
        w = self.weight(w)
        vals = [q.evaluate(t) for q,t in zip(self.branches,w)]
        J = sum((q[0] for q in vals),sp.zeros(self.d))
        D, F = [q[1] for q in vals], [q[2] for q in vals]
        P = J.inv()
        g, H = sp.zeros(self.s,1), sp.zeros(self.s)
        if mode not in ('trace','logdet'):
            raise ValueError('mode must be trace or logdet')
        for i in range(self.s):
            if mode == 'logdet':
                g[i] = -sp.trace(P*D[i])
            else:
                g[i] = -sp.trace(self.W*P*D[i]*P)
            for j in range(i,self.s):
                if mode == 'logdet':
                    h = sp.trace(P*D[i]*P*D[j])
                    if i == j:
                        h -= sp.trace(P*F[i])
                else:
                    h = (sp.trace(self.W*P*D[i]*P*D[j]*P)
                         +sp.trace(self.W*P*D[j]*P*D[i]*P))
                    if i == j:
                        h -= sp.trace(self.W*P*F[i]*P)
                H[i,j] = H[j,i] = h
        return J,P,g,H

    def verified_upper_scale(self, c: Any, proposal: Any | None = None):
        """Prove V_i <= b M(c) with exact arithmetic.

        A proposal may come from an ordinary eigensolver, but is not trusted.
        Strict LDL acceptance is used. Doubling terminates for any b>0.
        An interior weight also provides b=1/min(c) analytically.
        """
        c = self.weight(c)
        if proposal is None and all(x > 0 for x in c):
            return 1/min(c), 'simplex-interior inequality'
        Mc = self.lift(c)
        b = max(sp.Integer(1), rational(2 if proposal is None else proposal))
        for _ in range(256):
            if all(is_spd(b*Mc-V) for V in self.vertices):
                return b, 'exact rational LDL at every vertex'
            b *= 2
        raise ArithmeticError('Scale verification failed; no certificate issued.')

    def certificate(self, c: Any, z: Any, mode: str='logdet',
                    b_proposal: Any | None=None) -> dict:
        """Return an EXACT rational bound f(c)-min f <= gap.

        No determinant/logarithm evaluation is needed, even for logdet.
        The maximization with the center anchor preserves the first-order
        certificate as a fallback. A feasible z is a lower-QP anchor, not
        assumed to be a minimizer.
        """
        c,z = self.weight(c),self.weight(z)
        b, method = self.verified_upper_scale(c,b_proposal)
        _,_,g,H = self.oracle(c,mode)
        Q = H/b
        v = z-c
        q = scalar(g.T*v)+scalar(v.T*Q*v)/2
        r = g+Q*v
        delta = scalar(r.T*z)-min(r)
        dual = q-delta
        first_dual = min(g)-scalar(g.T*c)
        dual = max(dual, first_dual)
        gap = -dual
        if gap < 0 or delta < 0:
            raise AssertionError('Negative exact gap: checker failure.')
        def frac(x):
            return {'numerator': str(sp.numer(x)), 'denominator': str(sp.denom(x))}
        return {
            'mode':mode, 'arithmetic':'exact rational',
            'input_convention':'integers/rationals exact; floats exact binary values',
            'gap':frac(gap), 'gap_upper_decimal':upper_decimal(gap),
            'first_order_gap':frac(-first_dual),
            'lower_qp_gap':frac(delta), 'b':frac(b),
            'scale_proof':method,
            'w':[str(x) for x in c], 'lower_anchor':[str(x) for x in z],
            'needs_logarithm_evaluation':False,
        }


def upper_decimal(x: Any, precision: int = 35) -> str:
    x = rational(x)
    with localcontext() as ctx:
        ctx.prec = precision
        ctx.rounding = ROUND_CEILING
        return str(Decimal(int(sp.numer(x))) / Decimal(int(sp.denom(x))))


def quantize_simplex(w: Sequence[float], bits: int=32) -> sp.Matrix:
    """Largest-remainder dyadic rounding, preserving exact feasibility."""
    if bits < 1 or any(not math.isfinite(float(x)) for x in w):
        raise ValueError('Invalid quantization request.')
    values = [rational(max(0.0,float(x))) for x in w]
    total = sum(values)
    if total <= 0:
        raise ValueError('Zero simplex vector.')
    den = 2**bits
    scaled = [x/total*den for x in values]
    nums = [int(sp.floor(x)) for x in scaled]
    rem = den-sum(nums)
    order = sorted(range(len(values)),key=lambda i:scaled[i]-nums[i],reverse=True)
    if not 0 <= rem <= len(values):
        raise ArithmeticError('Invalid exact rounding.')
    for i in order[:rem]:
        nums[i] += 1
    return sp.Matrix([sp.Rational(n,den) for n in nums])
