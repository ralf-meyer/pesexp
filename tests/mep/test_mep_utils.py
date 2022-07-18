import ase.io
import ase.atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from pesexp.mep.utils import pRFO_guess_from_neb


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
