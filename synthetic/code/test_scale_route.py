"""Ordinary public routes, both scales, and explicit fallback regression."""
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark_core import SCI, certificate
from check_stress_scales import check_difference, check_input, unit
from verified_scale import rational_weights

data = Path(__file__).resolve().parents[1] / 'data'


with np.load(data/'stress_factor_inputs.npz') as frozen:
    for case in (3, 7, 15):
        p = SCI(frozen[f'Z_{case}'], frozen[f'a_{case}'])
        factor = [[(F(float(a)), tuple(F(float(t)) for t in z)) for a,z in zip(aa, zz.T)]
                  for aa,zz in zip(p.aas, p.Zs)]
        for values in (np.ones(p.s) / p.s, np.eye(p.s)[0]):
            center = rational_weights(values)
            a, b = p.scales(values, both=True)
            for vertex in range(p.s):
                assert check_input(factor, center, F(b), vertex)
                assert check_difference(factor, unit(p.s,vertex), 1/F(float(a)), center)
            for mode in ('trace', 'logdet'):
                out = certificate(p, values, mode)
                assert out['scale_status'] == 'vertex_exact'
                assert all(check_input(factor, center, F(out['b']), i) for i in range(p.s))
    # Simulate an unsafe spectral proposal independently of machine BLAS.
    with patch('verified_scale.stable_spectral_proposal', return_value=(0.5,1.)):
        values = np.arange(1, p.s+1, dtype=float)
        values /= np.sum(values)
        out = certificate(p, values, 'trace')
        center = rational_weights(values)
        assert out['scale_status'] == 'analytic_interior_fallback'
        assert F(out['b']) >= 1/min(center)
        boundary = np.eye(p.s)[0]
        _, b, details = p.scales(boundary, diagnostics=True)
        assert details['scale_status'] == 'vertex_exact_doubled'
        assert all(check_input(factor, rational_weights(boundary), F(b), i) for i in range(p.s))
print('Public upper/lower scales, both certificates, interior and boundary fallbacks passed.')
