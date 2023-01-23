import os
import pytest
import ase.atoms
import ase.build
import ase.calculators
import ase.units
import numpy as np
import numdifftools as nd
import geometric.internal
from utils import g2_molecules, xtb_installed
from pesexp.calculators import _openbabel_methods, get_calculator
from pesexp.hessians.hessian_guesses import (
    filter_hessian,
    get_hessian_guess,
    numerical_hessian,
    TrivialGuessHessian,
    SchlegelHessian,
    FischerAlmloefHessian,
    LindhHessian,
)


def num_diff_tools_hessian(atoms, step=None, method="central", use_forces=True):
    x0 = atoms.get_positions()

    if method == "forward":
        order = 1
    else:
        order = 2

    if use_forces:

        def fun(x):
            atoms.set_positions(x.reshape(-1, 3))
            return -atoms.get_forces().flatten()

        H = nd.Jacobian(fun, step=step, method=method, order=order)(x0.flatten())
    else:

        def fun(x):
            atoms.set_positions(x.reshape(-1, 3))
            return atoms.get_potential_energy()

        H = nd.Hessian(fun, step=step, method=method, order=order)(x0.flatten())
    return 0.5 * (H + H.T)


@pytest.mark.parametrize("method", _openbabel_methods)
@pytest.mark.parametrize("system", ["H2", "H2O", "C3H8"])
@pytest.mark.parametrize("diff_method", ["central", "forward"])
def test_numerical_hessian(method, system, diff_method):
    # check if openbabel version > 3.0. This is necessary as
    # OBForceField.GetGradient is not public for prior versions.
    pytest.importorskip("openbabel", minversion="3.0")
    atoms = ase.build.molecule(system)
    atoms.calc = get_calculator(method)
    if method == "mmff94" and system == "H2":
        # MMFF94 does not have parameters for H2 and is
        # therefore expected to fail.
        with pytest.raises(RuntimeError):
            atoms.get_potential_energy()
    else:
        x0 = atoms.get_positions()
        H = numerical_hessian(atoms, symmetrize=False, method=diff_method)
        np.testing.assert_allclose(atoms.get_positions(), x0)
        if diff_method == "central":  # Forward difference are not symmetric
            np.testing.assert_allclose(H, H.T, atol=1e-8)
        H_ref = num_diff_tools_hessian(atoms, step=1e-5, method=diff_method)
        # Symmetrize
        H = 0.5 * (H + H.T)
        np.testing.assert_allclose(H, H_ref, atol=1e-4)


@pytest.mark.parametrize("system", ["H2", "H2O", "C3H8"])
@pytest.mark.parametrize("diff_method", ["central", "forward"])
def test_numerical_hessian_EMT(system, diff_method):
    atoms = ase.build.molecule(system)
    atoms.calc = ase.calculators.emt.EMT()

    x0 = atoms.get_positions()
    H = numerical_hessian(atoms, symmetrize=False, method=diff_method)
    np.testing.assert_allclose(atoms.get_positions(), x0)
    if diff_method == "central":  # Forward difference are not symmetric
        np.testing.assert_allclose(H, H.T, atol=1e-8)
    H_ref = num_diff_tools_hessian(atoms, step=1e-5, method=diff_method)
    # Symmetrize
    H = 0.5 * (H + H.T)
    np.testing.assert_allclose(H, H_ref, atol=1e-4)

    # vals, vecs = np.linalg.eigh(H)
    # if len(atoms) == 2:  # Linear molecule
    #     assert np.nonzero(abs(vals) > 1e-6) == 3 * len(atoms) - 5
    # else:
    #     assert np.nonzero(abs(vals) > 1e-6) == 3 * len(atoms) - 6


@xtb_installed
@pytest.mark.parametrize("system", ["H2", "LiF"])
def test_xtb_hessian(system):
    """TODO: For some reason I can not get this test to pass for any
    non-dimer molecules. Maybe due to the way the singlepoint calculations are
    restarted in xtb internal hessian calculation. RM 2022/02/22
    """
    atoms = ase.build.molecule(system)
    atoms.rotate(15, (1, 0, 0))
    x0 = atoms.get_positions()
    H = get_hessian_guess(atoms, "xtb")
    np.testing.assert_allclose(atoms.get_positions(), x0)
    np.testing.assert_allclose(H, H.T, atol=1e-8)
    xtb_ase_calc = pytest.importorskip("xtb.ase.calculator")
    atoms.calc = xtb_ase_calc.XTB(method="GFN2-xTB", accuracy=0.3)
    H_ref = numerical_hessian(atoms, step=0.005 * ase.units.Bohr)
    eig, _ = np.linalg.eigh(H)
    eig_ref, _ = np.linalg.eigh(H_ref)
    np.testing.assert_allclose(eig, eig_ref, atol=1e-2)
    np.testing.assert_allclose(H, H_ref, atol=1e-2)


@pytest.mark.skip(
    "Skipping Hessian test for Fe CO as it fails for both uff "
    "(exploding energies) and xtb"
)
@pytest.mark.parametrize("method", ["uff", "xtb"])
def test_Fe_CO_6(method):
    """TODO: Fails for both uff (exploding energies) and xtb
    (see test_xtb_hessian)
    """
    r_FeC = 2.3
    r_FeO = 2.3 + 1.1
    atoms = ase.atoms.Atoms(
        ["Fe"] + ["C", "O"] * 6,
        positions=[
            [0.0, 0.0, 0.0],
            [r_FeC, 0.0, 0.0],
            [r_FeO, 0.0, 0.0],
            [0.0, r_FeC, 0.0],
            [0.0, r_FeO, 0.0],
            [-r_FeC, 0.0, 0.0],
            [-r_FeO, 0.0, 0.0],
            [0.0, -r_FeC, 0.0],
            [0.0, -r_FeO, 0.0],
            [0.0, 0.0, r_FeC],
            [0.0, 0.0, r_FeO],
            [0.0, 0.0, -r_FeC],
            [0.0, 0.0, -r_FeO],
        ],
        charges=[2] + [0, 0] * 6,
    )
    x0 = atoms.get_positions()
    atoms.calc = get_calculator(method)
    H = get_hessian_guess(atoms, method)
    np.testing.assert_allclose(atoms.get_positions(), x0)
    np.testing.assert_allclose(H, H.T, atol=1e-8)
    H_ref = num_diff_tools_hessian(atoms, step=1e-5)
    np.testing.assert_allclose(H, H_ref, atol=1e-4)


@pytest.mark.parametrize("system", ["H2", "F2", "H2O", "CO2"])
def test_schlegel_vs_geometric(tmpdir, system):
    """Only tests simple systems since geomeTRIC does not use the actual
    Schlegel rules for dihedrals and impropers."""
    atoms = ase.build.molecule(system)
    tmp_file = os.path.join(tmpdir, "tmp.xyz")
    ase.io.write(tmp_file, atoms, plain=True)
    mol = geometric.molecule.Molecule(tmp_file)
    coords_ref = geometric.internal.PrimitiveInternalCoordinates(mol, connect=True)
    xyzs = atoms.get_positions()
    # The following only works in the newest version of geometric:
    # H_ref = coords_ref.calcHessCart(
    #     xyzs/ase.units.Bohr, np.zeros(len(coords_ref.Internals)),
    #     coords_ref.guess_hessian(xyzs/ase.units.Bohr))
    # for now calcHessCart is replaced with a simplified function
    # that neglects the gradient in internal coordinates:
    Bmat = coords_ref.wilsonB(xyzs / ase.units.Bohr)
    H_ref = (
        np.einsum(
            "ai,ab,bj->ij", Bmat, coords_ref.guess_hessian(xyzs / ase.units.Bohr), Bmat
        )
        * ase.units.Hartree
        / ase.units.Bohr**2
    )
    H = SchlegelHessian(threshold=1.2, h_trans=0.0).build(atoms)
    np.testing.assert_allclose(H, H_ref, atol=1e-10)


@pytest.mark.parametrize("method", ["trivial", "schlegel", "fischer_almloef", "lindh"])
@pytest.mark.parametrize("name", g2_molecules.keys())
def test_internal_coordinate_based_hessians(name, method, atol=1e-10):
    atoms = g2_molecules[name]["atoms"]
    h_trans = 5.0
    h_rot = 0.0
    if method == "trivial":
        H = TrivialGuessHessian(h_trans=h_trans, h_rot=h_rot).build(atoms)
    elif method == "schlegel":
        H = SchlegelHessian(h_trans=h_trans, h_rot=h_rot).build(atoms)
    elif method == "fischer_almloef":
        H = FischerAlmloefHessian(h_trans=h_trans, h_rot=h_rot).build(atoms)
    elif method == "lindh":
        H = LindhHessian(h_trans=h_trans, h_rot=h_rot).build(atoms)
    vals, _ = np.linalg.eigh(H)
    # Assert that there are exactly 3 eigenvalues corresponding to translation.
    assert np.count_nonzero(np.abs(vals - h_trans) < atol) == 3
    # Assert that there are 3 eigenvalues corresponding to rotation (or 2 in
    # the case of a linear molecule).
    indices = np.nonzero(np.abs(vals - h_rot) < atol)[0]
    # List of g2 linear molecules:
    linear = ["C2H2", "CO2", "CS2", "NCCN", "N2O", "HCN", "OCS", "CCH"]
    if len(atoms) == 2 or name in linear:
        # Lindh Hessians do not include linear bend coordinates and, therefore,
        # have more than two zero eigenvalues for linear molecules.
        if not (method == "lindh" and name in linear):
            assert len(indices) == 2
    else:
        assert len(indices) == 3
    # Set these indices to finite value for further checks
    vals[indices] = 0.1
    # Assert that the remaining eigenvalues are positive
    np.testing.assert_array_less(np.zeros_like(vals), vals)


def test_filter_hessian():
    H = np.diag([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    H = filter_hessian(H, thresh=1.1e-5)
    np.testing.assert_allclose(H, np.diag([1.1e-5, 1.1e-5, 1.0, 2.0, 3.0, 4.0]))

    # Build random matrix with eigenvalue above 0.1
    A = np.array(
        [
            [0.7432, 0.4965, 0.2700, 0.0742, -0.0800, -0.1814],
            [0.4965, 1.0133, 0.5708, 0.1900, -0.1071, -0.2977],
            [0.2700, 0.5708, 0.9332, 0.3893, -0.0277, -0.2830],
            [0.0742, 0.1900, 0.3893, 0.7155, 0.2134, -0.0696],
            [-0.0800, -0.1071, -0.0277, 0.2134, 0.6736, 0.4129],
            [-0.1814, -0.2977, -0.2830, -0.0696, 0.4129, 1.2388],
        ]
    )
    A_ref = A.copy()
    A = filter_hessian(A)
    # Test that it remained unaltered
    np.testing.assert_allclose(A, A_ref)
    # Increase the threshold to 1.0
    A = filter_hessian(A, thresh=1.0)
    vals, _ = np.linalg.eigh(A)
    # Test that the smaller eigenvalues have been filtered correctly
    np.testing.assert_allclose(vals, [1.0, 1.0, 1.0, 1.0, 1.27649882, 2.20586986])
