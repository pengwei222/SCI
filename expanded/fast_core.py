"""BLAS-based SCI oracle. Same float64 SCI model and objective as original_core.

Supported scope: A_i positive definite, B_i positive semidefinite; diagonal trace
weights set by the original constructor, including pose-only objectives; logdet.
No low-rank truncation, reduced precision, changed tolerance or noise rescaling.
Cached arrays are internal read-only-by-convention; do not mutate returned P.
"""
import numpy as np
import original_core as original


class FastSCI(original.SCI):
    """Drop-in oracle with batched products and exact-weight-point reuse.

    Raw quantities are cached before normalization because the parent constructor
    computes the uniform objective before setting self.scale.
    """
    def oracle(self, w, hessian=True):
        w = np.asarray(w, dtype=float)
        if not hasattr(self, '_zt'):
            self._zt = self.Z.swapaxes(-1, -2)
            self._wd = np.diag(self.W).copy()
            self._active = np.flatnonzero(self._wd > 0)
            self._sqrtwd = np.sqrt(self._wd[self._active])
            self._cache = None
        c = self._cache
        if c is None or not np.array_equal(w, c['w']):
            den = 1 + w[:, None]*self.beta
            Ji = (self.Z * (w[:, None]/den)[:, None, :]) @ self._zt
            P = original.inv(Ji.sum(axis=0))
            PZ = P @ self.Z
            if self.mode == 'trace':
                f = float(self._wd @ np.diag(P))
                C = PZ[:, self._active, :] * self._sqrtwd[None, :, None]
                diag = np.einsum('sik,sik->sk', C, C)
            else:
                sign, f = np.linalg.slogdet(P)
                if sign <= 0:
                    raise ArithmeticError('Invalid covariance')
                diag = np.einsum('sik,sik->sk', self.Z, PZ)
            g = -np.sum(diag/den**2, axis=1)
            c = dict(w=w.copy(), den=den, Ji=Ji, P=P, f=f, g=g,
                     diag=diag, H=None)
            self._cache = c
        if hessian and c['H'] is None:
            Zd = self.Z / c['den'][:, None, :]
            D = Zd @ Zd.swapaxes(-1, -2)
            LP = np.linalg.cholesky(c['P'])
            if self.mode == 'trace':
                # sqrt(W) has only pose columns, so nuisance-zero columns need
                # not be formed in the Hessian Gram factor.
                C = c['P'][:, self._active] * self._sqrtwd[None, :]
                Y = (LP.T @ D) @ C
                factor = 2.
            else:
                Y = (LP.T @ D) @ LP
                factor = 1.
            Y = Y.reshape(self.s, -1)
            H = factor*(Y @ Y.T)
            H.flat[::self.s+1] += np.sum(
                (2*self.beta/c['den']**3)*c['diag'], axis=1)
            c['H'] = original.sym(H)
        return (float(c['f']/self.scale), c['g']/self.scale,
                None if not hessian else c['H']/self.scale, c['P'])

    def information(self, w):
        w = np.asarray(w, dtype=float)
        c = getattr(self, '_cache', None)
        if c is not None and np.array_equal(w, c['w']):
            return c['Ji']
        den = 1 + w[:, None]*self.beta
        return (self.Z*(w[:, None]/den)[:, None, :]) @ self.Z.swapaxes(-1,-2)

    def fuse(self, w, means):
        self.oracle(w, False)
        c = self._cache
        x = c['P'] @ np.einsum('sij,sj->i', c['Ji'], means)
        return x, c['P']
