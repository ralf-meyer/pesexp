import ase.io
import ase.build
import ase.calculators
import ase.atoms
import warnings
import numpy as np
from pesexp.geometry.coordinate_systems import ApproximateNormalCoordinates
from pesexp.hessians.hessian_guesses import SchlegelHessian


def test_transformations(atol=1e-10):
    atoms = ase.build.molecule("H2O")
    anc = ApproximateNormalCoordinates(atoms)
    xyzs = atoms.get_positions()

    assert anc.B.shape == (3, 9)
    assert anc.BTinv.shape == (3, 9)
    # Definition 1 in https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    np.testing.assert_allclose(
        anc.B.T @ anc.BTinv @ anc.B.T - anc.B.T, np.zeros((9, 3)), atol=atol
    )
    _, s, _ = np.linalg.svd(anc.BTinv)
    np.testing.assert_allclose(s, np.ones_like(s), atol=atol)

    q = anc.to_internals(xyzs)
    np.testing.assert_allclose(q, np.zeros_like(q), atol=atol)
    # Test that backtransform works
    np.testing.assert_allclose(anc.to_cartesians(np.zeros(3), xyzs), xyzs)

    atoms.calc = ase.calculators.emt.EMT()
    f_cart = atoms.get_forces()
    f_int = anc.force_to_internals(xyzs, f_cart)
    assert f_int.shape == (3,)
    assert abs(np.linalg.norm(f_cart) - np.linalg.norm(f_int)) < atol
    np.testing.assert_allclose(anc.force_to_cartesians(xyzs, f_int), f_cart, atol=atol)

    H_cart = SchlegelHessian().build(atoms)
    H_int = anc.hessian_to_internals(xyzs, H_cart)
    assert H_int.shape == (3, 3)
    np.testing.assert_allclose(
        anc.hessian_to_cartesians(xyzs, H_int), H_cart, atol=atol
    )
    eigs_cart = np.linalg.eigh(H_cart)[0]
    eigs_int = np.linalg.eigh(H_int)[0]

    np.testing.assert_allclose(eigs_int, eigs_cart[abs(eigs_cart) > 1e-6])

    # Take single NR step:
    dx = (np.linalg.pinv(H_cart) @ f_cart.flatten()).reshape(-1, 3)
    # And shift center of mass
    xyzs_new = xyzs + dx  # + np.array([1.0, 2.0, 3.0])
    atoms.set_positions(xyzs_new)

    q = anc.to_internals(xyzs_new)
    # Since the direction of q is arbitrary, divide by the sign of the first component
    np.testing.assert_allclose(
        q / np.sign(q[0]), [0.1116441, -0.1121717, 0.0], atol=1e-6
    )
    dq = anc.diff_internals(xyzs_new, xyzs)
    np.testing.assert_allclose(anc.to_cartesians(dq, xyzs), xyzs_new, atol=atol)

    f_cart = atoms.get_forces()
    f_int = anc.force_to_internals(xyzs_new, f_cart)
    assert f_int.shape == (3,)
    assert abs(np.linalg.norm(f_cart) - np.linalg.norm(f_int)) < atol
    np.testing.assert_allclose(
        anc.force_to_cartesians(xyzs_new, f_int), f_cart, atol=atol
    )


def test_hessian_weighting(atol=1e-6):
    k1 = 10.0
    k2 = 100.0

    x0, y0 = 0.0, 0.0
    xyzs_0 = np.array([[x0, y0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    atoms = ase.atoms.Atoms(["H"] * 3, xyzs_0)

    def fun(x, y):
        return k1 / 2 * (x + y) ** 2 + k2 / 2 * (x - y) ** 2

    def grad(x, y):
        return np.array([k1 * (x + y) + k2 * (x - y), k1 * (x + y) - k2 * (x - y)])

    H = np.zeros((3 * len(atoms), 3 * len(atoms)))
    H[:2, :2] = [[k1 + k2, k1 - k2], [k1 - k2, k1 + k2]]

    with warnings.catch_warnings(record=True) as w:
        anc = ApproximateNormalCoordinates(atoms, H=H, weighted=True)
        # Expecting a warning about the number of coordinates
        assert "ApproximateNormalCoordinates found" in str(w[0].message)

    H_int = anc.hessian_to_internals(xyzs_0, H, grad_cart=grad(x0, y0))
    np.testing.assert_allclose(H_int, np.eye(2), atol=atol)

    dq1 = np.array([1.0, 0.0])
    xyzs_1 = anc.to_cartesians(dq1, xyzs_0)
    x1, y1 = xyzs_1[0, :2]
    assert abs(fun(x1, y1) - 0.5) < atol

    dq2 = np.array([0.0, 1.0])
    xyzs_2 = anc.to_cartesians(dq2, xyzs_0)
    x2, y2 = xyzs_2[0, :2]
    assert abs(fun(x2, y2) - 0.5) < atol
