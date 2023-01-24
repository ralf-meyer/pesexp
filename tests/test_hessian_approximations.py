import pytest
import numpy as np
from pesexp.hessians.hessian_approximations import (
    BFGSHessian,
    TSBFGSHessian,
    MurtaghSargentHessian,
    PowellHessian,
    BofillHessian,
    ModifiedBofillHessian,
    FarkasSchlegelHessian,
)


@pytest.mark.parametrize(
    "hess",
    [
        BFGSHessian,
        TSBFGSHessian,
        MurtaghSargentHessian,
        PowellHessian,
        BofillHessian,
        ModifiedBofillHessian,
        FarkasSchlegelHessian,
    ],
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


@pytest.mark.parametrize(
    "hess",
    [BFGSHessian, MurtaghSargentHessian],
)
@pytest.mark.parametrize(
    "h0",
    [[[2.0, -0.5], [-0.5, 3.0]], [[1.0, -2.5], [-2.5, 3.0]]],
)
def test_anisotropic_scaling_invariance(hess, h0):
    # Starting Hessian is positive definite
    H0 = hess(h0)
    H_ref = H0.copy()

    dx = np.ones(2)
    dg = np.ones(2)
    H_ref.update(dx, dg)

    scale = np.array([2.14, 10.25])
    dz = dx * scale
    dE_dz = dg / scale

    scale_mat = np.outer(scale, scale)
    H = H0.copy() / scale_mat
    H.update(dz, dE_dz)
    H *= scale_mat
    np.testing.assert_allclose(H, H_ref)


@pytest.mark.parametrize(
    "hess",
    [
        BFGSHessian,
        TSBFGSHessian,
        MurtaghSargentHessian,
        PowellHessian,
        BofillHessian,
        ModifiedBofillHessian,
        FarkasSchlegelHessian,
    ],
)
def test_quasi_newton_condition(hess):
    H = hess([[2.0, -0.5], [-0.5, 3.0]])

    dx = np.array([0.1, -2.3])
    dg = np.array([0.8, 0.3])

    deltaH = H.deltaH(dx, dg)
    np.testing.assert_allclose(deltaH @ dx, dg - H @ dx)


def test_powell_numerical_stability(alpha=1e-9):
    # Test small updates on the following function:
    # f(x, y) = x**2 - 1.5 * y**2
    H = PowellHessian([[1.999, 0.0], [0.0, -3.0]])

    x0 = np.array([1.0, 0.0])
    g0 = np.array([2.0 * x0[0], -3.0 * x0[1]])
    dx = alpha * np.linalg.pinv(H) @ g0

    x1 = x0 + dx
    g1 = np.array([2.0 * x1[0], -3.0 * x1[1]])
    dg = g1 - g0

    H.update(dx, dg)
    vals, _ = np.linalg.eigh(H)
    np.testing.assert_allclose(vals, [-3.0, 2.0])


@pytest.mark.parametrize("hess", [PowellHessian, BofillHessian])
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


def test_Bofill_Hessian():
    """This test was introduced to inspect why the Bofill update sometimes leads to
    bad estimates of the Hessian eigenspectrum. Changing the initial value of the
    z entry shows that this problem occurs when large steps are taken, thus
    motivating a step size control algorithm."""
    # f(x, y, z) = x**2 - y**2 + 0.001 * z**2
    # Initialize close to correct Hessian
    H = BofillHessian([[1.95, 0.0, 0.0], [0.0, -1.98, 0.0], [0.0, 0.0, 1.8e-3]])
    vals_prior = np.linalg.eigh(H)[0]

    x0 = np.array([1.0, 1.0, 1.0])
    g0 = np.array([2 * x0[0], -2 * x0[1], 0.002 * x0[2]])
    # Quasi Newton step
    x1 = x0 - np.linalg.inv(H).dot(g0)
    g1 = np.array([2 * x1[0], -2 * x1[1], 0.002 * x1[2]])
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
