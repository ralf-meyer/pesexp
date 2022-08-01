import pytest
import numpy as np
from pesexp.hessians.hessian_approximations import (
    BFGSHessian,
    MurtaghSargentHessian,
    PsBHessian,
    BofillHessian,
)


@pytest.mark.parametrize(
    "hess", [BFGSHessian, MurtaghSargentHessian, PsBHessian, BofillHessian]
)
def test_scaling_invariance(hess):
    """The updates should be invariant to various kinds of scaling."""
    H = hess([[2.0, -0.5], [-0.5, 3.0]])
    H_ref = H.copy()

    dx = np.ones(2)
    dg = np.ones(2)
    H_ref.update(dx, dg)
    # Rescale both dx and dg
    H.update(1e-3 * dx, 1e-3 * dg)
    np.testing.assert_allclose(H, H_ref)

    # Transform whole matrix to different units. In this example
    # from Hartree/Bohr**2 to eV/Ang**2 (approximately).
    Hartree = 27.21
    Bohr = 0.53

    H = hess([[2.0, -0.5], [-0.5, 3.0]])
    H_ref = H.copy()

    dx = np.ones(2)
    dg = np.ones(2)
    # Update and then transform
    H_ref.update(dx, dg)
    H_ref *= Hartree / Bohr**2
    # Transform and then apply transformed updates
    H *= Hartree / Bohr**2
    H.update(dx * Bohr, dg * Hartree / Bohr)
    np.testing.assert_allclose(H, H_ref)


@pytest.mark.parametrize("hess", [PsBHessian, BofillHessian])
def test_perfect_update(hess):
    # Test what happens if the "correct" Hessian is updated using exact values from a
    # quadratic function: f(x, y) = x**2 - y**2
    H = hess([[2.0, 0.0], [0.0, -2.0]])

    x0 = np.array([1.0, 1.0])
    g0 = np.array([2 * x0[0], -2 * x0[1]])
    x1 = np.array([0.0, 0.0])
    g1 = np.array([2 * x1[0], -2 * x1[1]])
    dx = x1 - x0
    dg = g1 - g0

    H.update(dx, dg)
    np.testing.assert_allclose(H, [[2.0, 0.0], [0.0, -2.0]])


@pytest.mark.skip(
    "Test fails because the prediction of the small eigenvalue worsens after the update"
)
def test_Bofill_Hessian():
    # f(x, y, z) = x**2 - y**2
    # Initialize close to correct Hessian
    H = BofillHessian([[1.95, 0.0, 0.0], [0.0, -1.98, 0.0], [0.0, 0.0, 1e-5]])
    vals_prior = np.linalg.eigh(H)[0]

    x0 = np.array([1.0, 1.0, 1.0])
    g0 = np.array([2 * x0[0], -2 * x0[1], 0.0])
    x1 = np.array([0.0, 0.0, 0.0])
    g1 = np.array([2 * x1[0], -2 * x1[1], 0.0])
    dx = x1 - x0
    dg = g1 - g0

    H.update(dx, dg)
    vals_after = np.linalg.eigh(H)[0]
    # Assert that the estimate of all eigenvalues improved:
    # (Note the eigenvalues are sorted by magnitude in np.linalg.eigh)
    vals_ref = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_array_less(
        np.abs(vals_after - vals_ref), np.abs(vals_prior - vals_ref)
    )
