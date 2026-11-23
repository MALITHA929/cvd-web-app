"""
CVD simulation matrices used by adjoint_multires_interp.py, farup.py,
adjoint_recolour_results.py and adjoint_filter_dataset.py.

This file is imported as `from cvd_matrices import CVD_MATRICES` but is not
included in the public repo (https://github.com/tomgillooly/cvd_vit) -- only
the *use* of CVD_MATRICES[severity][cvd_type] is visible in the source, not
its construction. This reconstructs it from the paper's own description
(Sec. 3.1: "we use the simulation matrices M as defined in [15]", where [15]
is Machado, Oliveira & Fernandes, IEEE TVCG 2009) using the `colour-science`
package, which is already pinned in requirements.txt and implements exactly
that model (colour.matrix_cvd_Machado2009).

Severity <-> cone-response-shift mapping, per Sec. 3.4 of the paper:
    0.4  <->  8nm  (mild)
    0.6  <-> 12nm  (moderate)
    1.0  <-> 20nm  (high)
These map directly to the `severity` argument (domain [0, 1]) of
colour.matrix_cvd_Machado2009, which is itself built from Machado et al.'s
pre-tabulated matrices at 8/12/20nm-equivalent shift values.

CVD_MATRICES[severity][cvd_type] -> 3x3 numpy.ndarray (float32), such that
for an RGB image tensor of shape (..., 3), the CVD-simulated RGB is
`image @ CVD_MATRICES[severity][cvd_type].T` (this matches how
adjoint_multires_interp.py calls it via
torch.einsum('ijkl,mj->imkl', image, cvd_sim_matrix)).
"""

import numpy as np

try:
    import colour
except ImportError as e:
    raise ImportError(
        "cvd_matrices.py needs the `colour-science` package "
        "(pip install colour-science, already pinned in requirements.txt)."
    ) from e

# Paper's cvd_type strings -> colour-science's `deficiency` strings
_DEFICIENCY_NAME = {
    'protan': 'Protanomaly',
    'deutan': 'Deuteranomaly',
    'tritan': 'Tritanomaly',
}

# Severities used throughout the repo (argparse choices in
# adjoint_multires_interp.py: --severity {0.4, 0.6, 1.0})
_SEVERITIES = [0.4, 0.6, 1.0]


def _build_cvd_matrices():
    matrices = {}
    for severity in _SEVERITIES:
        matrices[severity] = {}
        for cvd_type, deficiency in _DEFICIENCY_NAME.items():
            m = colour.matrix_cvd_Machado2009(deficiency, severity)
            matrices[severity][cvd_type] = np.asarray(m, dtype=np.float32)
    return matrices


CVD_MATRICES = _build_cvd_matrices()


if __name__ == '__main__':
    # Quick sanity check / demo
    for severity, by_type in CVD_MATRICES.items():
        for cvd_type, m in by_type.items():
            print(f'severity={severity:<4} cvd_type={cvd_type:<7} matrix=\n{m}\n')
