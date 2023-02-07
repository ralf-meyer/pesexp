import pytest
import ase.io
import ase.atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.neb import NEB
import numpy as np
from pesexp.mep.utils import pRFO_guess_from_neb
from pesexp.optimizers.optimizers import PRFO
from pesexp.optimizers.convergence import terachem_convergence
from pesexp.geometry.coordinate_systems import ApproximateNormalCoordinates
from pesexp.geometry.primitives import Dihedral


def test_pRFO_workflow(resource_path_root):
    images = ase.io.read(resource_path_root / "mep" / "Fe_(NH3)_5+N2O.traj", index=":")
    imax = 4  # Highest image on this trajectory
    atoms, H = pRFO_guess_from_neb(images, remove_translation_and_rotation=True)
    # Assert that the guess for the transition state is close to the highest image
    np.testing.assert_allclose(
        atoms.get_positions(), images[imax].get_positions(), atol=1e-1
    )
    # Test the properties of the newly constructed Hessian
    vals, vecs = np.linalg.eigh(H)
    # Assert that there is only one negative eigenvalue (neglecting values
    # close to zero)
    assert np.count_nonzero(vals < -1e-6) == 1
    # Assert that the Hessian has 6 close to zero eigenvalues (translation and rotation)
    assert np.count_nonzero(abs(vals) < 1e-6) == 6
    # Assert that the smallest eigenvalue corresponds to the bond breaking
    # direction, i.e. it is approximately parallel to the tangent at image imax.
    tangent = images[imax - 1].get_positions() - images[imax + 1].get_positions()
    tangent = tangent.flatten() / np.linalg.norm(tangent)
    assert np.abs(np.dot(vecs[:, 0], tangent)) > 0.95

    # Assert that the curvature is approximately correct:
    ts = images[4]
    neighbor1 = images[3]
    neighbor2 = images[5]
    h1 = np.linalg.norm(neighbor1.get_positions() - ts.get_positions())
    h2 = np.linalg.norm(neighbor2.get_positions() - ts.get_positions())
    e1 = neighbor1.get_potential_energy()
    e2 = neighbor2.get_potential_energy()
    curv = (
        2
        * (h1 * e2 + h2 * e1 - (h1 + h2) * ts.get_potential_energy())
        / (h1 * h2 * (h1 + h2))
    )
    # Very loose check that the relative error is smaller than 0.5
    assert abs(vals[0] / curv - 1) < 0.5


def test_pRFO_curvature():
    images = []
    x = np.linspace(-1, 1, 6)
    for xi in x:
        a = ase.atoms.Atoms(["H"], positions=[[xi, 0.0, 0.0]])
        a.calc = SinglePointCalculator(
            a, energy=-(xi**2), forces=[[2 * xi, 0.0, 0.0]]
        )
        images.append(a)

    atoms, H = pRFO_guess_from_neb(images, remove_translation_and_rotation=False)

    # Maximum should be at x = 0
    np.testing.assert_allclose(atoms.get_positions(), [[0.0, 0.0, 0.0]], atol=1e-4)
    # Curvature of -x**2 should be -2
    vals, _ = np.linalg.eigh(H)
    assert abs(vals[0] - -2.0) < 1e-4


def test_pRFO_workflow_ethane():
    xtb_ase_calc = pytest.importorskip("xtb.ase.calculator")
    initial = ase.build.molecule("C2H6")

    final = initial.copy()
    final.positions[2:5] = initial.positions[[3, 4, 2]]

    images = [initial] + [initial.copy() for _ in range(9)] + [final]
    for im in images:
        im.calc = xtb_ase_calc.XTB(method="GFN2-xTB")
    neb = NEB(images)
    neb.interpolate("idpp")

    atoms, H = pRFO_guess_from_neb(images, remove_translation_and_rotation=True)
    atoms.calc = xtb_ase_calc.XTB(method="GFN2-xTB")
    coord_set = ApproximateNormalCoordinates(atoms, H=H, threshold=1e-8)

    opt = terachem_convergence(PRFO)(
        atoms, coordinate_set=coord_set, H0=H, maxstep=0.05
    )
    opt.run(steps=100)

    assert opt.converged()
    # Assert that the two methly groups have aligned
    assert abs(Dihedral(2, 0, 1, 6).value(atoms.get_positions())) < 1e-2
    assert abs(Dihedral(3, 0, 1, 5).value(atoms.get_positions())) < 1e-2
    assert abs(Dihedral(4, 0, 1, 7).value(atoms.get_positions())) < 1e-2
