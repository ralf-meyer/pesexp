import pytest
import numpy as np
from pesexp.hessians.hessian_approximations import (BFGSHessian,
                                                    MurtaghSargentHessian,
                                                    PsBHessian,
                                                    BofillHessian)


@pytest.mark.parametrize('hess', [BFGSHessian, MurtaghSargentHessian,
                                  PsBHessian, BofillHessian])
def test_scaling_invariance(hess):
    """The updates should be invariant to various kinds of scaling."""
    H = hess([[2.0, -0.5],
              [-0.5, 3.0]])
    H_ref = H.copy()

    dx = np.ones(2)
    dg = np.ones(2)
    H_ref.update(dx, dg)
    # Rescale both dx and dg
    H.update(1e-3*dx, 1e-3*dg)
    np.testing.assert_allclose(H, H_ref)

    # Transform whole matrix to different units. In this example
    # from Hartree/Bohr**2 to eV/Ang**2 (approximately).
    Hartree = 27.21
    Bohr = 0.53

    H = hess([[2.0, -0.5],
              [-0.5, 3.0]])
    H_ref = H.copy()

    dx = np.ones(2)
    dg = np.ones(2)
    # Update and then transform
    H_ref.update(dx, dg)
    H_ref *= Hartree/Bohr**2
    # Transform and then apply transformed updates
    H *= Hartree/Bohr**2
    H.update(dx*Bohr, dg*Hartree/Bohr)
    np.testing.assert_allclose(H, H_ref)
