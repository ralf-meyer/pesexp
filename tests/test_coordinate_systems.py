import pytest
import ase.io
import numpy as np
import geometric.internal
from utils import g2_molecules
from rmsd import kabsch_rmsd
from pesexp.geometry.primitives import (Distance, Angle,
                                        LinearAngle, Dihedral, Improper)
from pesexp.geometry.coordinate_systems import (InternalCoordinates,
                                                DelocalizedCoordinates,
                                                get_coordinate_system)


@pytest.mark.parametrize('name', g2_molecules.keys())
def test_redundant_internals(name):
    if name == 'Si2H6':
        # Skip Si2H6 because of a 0 != 2 pi error
        return
    atoms = g2_molecules[name]['atoms']
    mol = g2_molecules[name]['mol']
    coords_ref = geometric.internal.PrimitiveInternalCoordinates(
        mol, connect=True)

    primitives = []
    for ic in coords_ref.Internals:
        if isinstance(ic, geometric.internal.Distance):
            primitives.append(Distance(ic.a, ic.b))
        elif isinstance(ic, geometric.internal.Angle):
            primitives.append(Angle(ic.a, ic.b, ic.c))
        elif isinstance(ic, geometric.internal.Dihedral):
            primitives.append(Dihedral(ic.a, ic.b, ic.c, ic.d))
        elif isinstance(ic, geometric.internal.OutOfPlane):
            primitives.append(Dihedral(ic.a, ic.b, ic.c, ic.d))
        elif isinstance(ic, geometric.internal.LinearAngle):
            primitives.append(LinearAngle(ic.a, ic.b, ic.c, ic.axis))
        else:
            raise NotImplementedError(f'Internal {type(ic)} not implemented')

    coords = InternalCoordinates(primitives, save_failures=False)

    xyzs = atoms.get_positions()
    # Test transformation to internal coordinates
    np.testing.assert_allclose(coords.to_internals(xyzs),
                               coords_ref.calculate(xyzs),
                               atol=1e-8)

    B = coords.B(xyzs)
    np.testing.assert_allclose(B, coords_ref.wilsonB(xyzs), atol=1e-8)
    # Assert that B matrix contains no translations
    n_q = coords.size()
    np.testing.assert_allclose(np.sum(B.reshape((n_q, -1, 3)), axis=1),
                               np.zeros((n_q, 3)), atol=1e-8)

    np.testing.assert_allclose(coords.Ginv(xyzs),
                               coords_ref.GInverse(xyzs),
                               atol=1e-8)

    # Test transformation back to cartesian
    # Try to reconstruct the geometry from a distorted reference
    np.random.seed(4321)
    xyzs_dist = xyzs + 0.1*np.random.randn(*xyzs.shape)
    np.testing.assert_allclose(coords.to_internals(xyzs_dist),
                               coords_ref.calculate(xyzs_dist),
                               atol=1e-8)

    dq_ref = coords_ref.calcDiff(xyzs, xyzs_dist)
    dq = coords.diff_internals(xyzs, xyzs_dist)
    np.testing.assert_allclose(dq, dq_ref, atol=1e-8)
    xyzs2_ref = coords_ref.newCartesian(
        xyzs_dist.flatten(), dq.flatten()).reshape(-1, 3)

    if coords_ref.bork:  # Meaning that the transformation failed:
        with pytest.raises(RuntimeError):
            coords.to_cartesians(dq, xyzs_dist, maxstep=np.inf)
        return
    xyzs2 = coords.to_cartesians(dq, xyzs_dist, maxstep=0.1)

    assert kabsch_rmsd(xyzs2, xyzs2_ref, translate=True) < 1e-5
    # Test that final geometry is close to original
    assert kabsch_rmsd(xyzs2, xyzs, translate=True) < 1e-5


@pytest.mark.parametrize('name', g2_molecules.keys())
def test_delocalized_internals(name):
    if name == 'Si2H6':
        # Skip Si2H6 because of a 0 != 2 pi error
        return
    atoms = g2_molecules[name]['atoms']
    mol = g2_molecules[name]['mol']
    coords_ref = geometric.internal.DelocalizedInternalCoordinates(
        mol, connect=True, build=True)

    primitives = []
    for ic in coords_ref.Prims.Internals:
        if isinstance(ic, geometric.internal.Distance):
            primitives.append(Distance(ic.a, ic.b))
        elif isinstance(ic, geometric.internal.Angle):
            primitives.append(Angle(ic.a, ic.b, ic.c))
        elif isinstance(ic, geometric.internal.Dihedral):
            primitives.append(Dihedral(ic.a, ic.b, ic.c, ic.d))
        elif isinstance(ic, geometric.internal.OutOfPlane):
            primitives.append(Dihedral(ic.a, ic.b, ic.c, ic.d))
        elif isinstance(ic, geometric.internal.LinearAngle):
            primitives.append(LinearAngle(ic.a, ic.b, ic.c, ic.axis))
        else:
            raise NotImplementedError(f'Internal {type(ic)} not implemented')

    xyzs = atoms.get_positions()
    coords = DelocalizedCoordinates(primitives, xyzs, threshold=1e-6)

    # Obtaining the same eigenvectors is almost impossible due to possible
    # degeneracies. TODO: figure out way to test this nevertheless.

    # From here on out geomeTRIC is forced to use our results.
    coords_ref.Vecs = coords.U

    q = coords.to_internals(xyzs)
    q_ref = coords_ref.calculate(xyzs)
    np.testing.assert_allclose(q, q_ref, atol=1e-8)

    B = coords.B(xyzs)
    B_ref = coords_ref.wilsonB(xyzs)
    np.testing.assert_allclose(B, B_ref, atol=1e-8)
    # Assert that B matrix contains no translations
    n_q = coords.size()
    np.testing.assert_allclose(np.sum(B.reshape((n_q, -1, 3)), axis=1),
                               np.zeros((n_q, 3)), atol=1e-8)

    # Try to reconstruct the geometry from a distorted reference
    np.random.seed(4321)
    xyzs_dist = xyzs + 0.1*np.random.randn(*xyzs.shape)

    dq = coords.diff_internals(xyzs, xyzs_dist)
    dq_ref = coords_ref.calcDiff(xyzs, xyzs_dist)
    np.testing.assert_allclose(dq, dq_ref, atol=1e-8)

    xyzs2 = coords.to_cartesians(dq, xyzs_dist, maxstep=0.1)
    # Test that final geometry is close to original
    assert kabsch_rmsd(xyzs2, xyzs, translate=True) < 1e-5


def test_impossible_triangles():
    """The idea of this test case is to request a back transformation from
    geometrically 'impossible' (i.e. contradicting) set of redundant
    coordinates."""
    # First example is a equilateral triangle where a inconsistent set of side
    # lengths and angles is used.
    xyzs = np.array([[0., 0., 4.],
                     [3., 0., 0.],
                     [-3., 0., 0.]])

    primitives = [Distance(0, 1), Distance(0, 2), Distance(1, 2),
                  Angle(1, 0, 2)]
    coord_set = InternalCoordinates(primitives)
    q = coord_set.to_internals(xyzs)
    # Change the side lengths (0, 1) and (0, 2) from 5 to 4 Angstrom while
    # keeping the angle (1, 0, 2) fixed.
    q_new = q.copy()
    q_new[:2] = 4.
    # This results in [4., 4., 6., 1.28700222]
    dq = q_new - q
    xyzs = coord_set.to_cartesians(dq, xyzs)
    q_final = coord_set.to_internals(xyzs)
    # Assert that the backtransformation resulted in a compromise of length
    # and angle changes that are as close as possible to q_new.
    q_ref = [4.08, 4.08, 5.89, 1.61]
    np.testing.assert_allclose(q_final, q_ref, atol=0.01)

    # Second example: three angles that do not add up to 180
    primitives = [Angle(1, 0, 2), Angle(0, 1, 2), Angle(0, 2, 1)]
    coord_set = InternalCoordinates(primitives)
    q = coord_set.to_internals(xyzs)
    # Increase all angles by 1
    dq = np.ones(q.shape)
    xyzs = coord_set.to_cartesians(dq, xyzs)
    q_final = coord_set.to_internals(xyzs)
    # Assert that the geometry has not changed since it is not possible to
    # realize the requested change
    np.testing.assert_allclose(q_final, q)


def test_difficult_backtransformations():
    """Collection of backtransformations that have previously failed"""
    dq = np.array([
        -2.55649951e-01, -7.00629446e-02, -3.06523919e-01, -1.10446877e-01,
        9.11339677e-03,  2.82363504e-01,  2.43707743e-02,  7.89718523e-02,
        2.45630077e-02,  8.33115396e-02,  5.73663475e-03,  3.65728286e-02,
        1.35996523e-02, -9.16465357e-02,  9.49289202e-02,  1.40607213e-02,
        -1.68959287e-01,  1.73961566e-01,  1.84477339e-02, -9.56975507e-02,
        9.24851631e-02, -1.77046877e-01,  1.72087667e-01, -6.94935808e-01,
        -2.08768675e-02,  9.93074135e-01, -2.97252849e-02,  6.95489833e-01,
        1.00788510e-02, -1.00000000e+00,  8.95796387e-03,  1.58401215e-03,
        -2.31595649e-03, -7.48378180e-04, -1.86763552e-03, -1.23072431e-01])
    xyzs_ref = np.array([
        [-1.86318924e-02, -2.00432626e-02, -2.65778210e-01],
        [4.50528678e-02,  2.00410681e+00, -1.25055298e-01],
        [8.82009675e-02,  2.92135490e+00,  3.47073266e-01],
        [2.07799671e+00,  4.85485219e-02, -1.54557178e-02],
        [2.80140937e+00,  8.30984741e-02,  5.40339397e-01],
        [1.36360916e-03, -2.02203919e+00, -1.23924003e-01],
        [5.32957615e-03, -2.93975477e+00,  3.51035266e-01],
        [-2.09749862e+00,  4.94112974e-03, -1.30389161e-02],
        [-2.81815924e+00,  7.05630460e-03,  5.46231436e-01],
        [-1.97089471e-02, -1.98612123e-02,  1.48973506e+00],
        [-1.91075577e-02, -1.88338775e-02,  2.64031391e+00],
        [-2.25656805e-02, -2.35345004e-02, -2.12318401e+00],
        [-2.36811609e-02, -2.50393166e-02, -3.27843517e+00]])

    primitives = [Distance(0, 1), Distance(0, 3), Distance(0, 5),
                  Distance(0, 7), Distance(0, 9), Distance(0, 11),
                  Distance(1, 2), Distance(3, 4), Distance(5, 6),
                  Distance(7, 8), Distance(9, 10), Distance(11, 12),
                  Angle(1, 0, 7), Angle(1, 0, 9), Angle(1, 0, 11),
                  Angle(3, 0, 5), Angle(3, 0, 9), Angle(3, 0, 11),
                  Angle(5, 0, 7), Angle(5, 0, 9), Angle(5, 0, 11),
                  Angle(7, 0, 9), Angle(7, 0, 11),
                  LinearAngle(0, 1, 2, axis=0), LinearAngle(0, 1, 2, axis=1),
                  LinearAngle(0, 3, 4, axis=0), LinearAngle(0, 3, 4, axis=1),
                  LinearAngle(0, 5, 6, axis=0), LinearAngle(0, 5, 6, axis=1),
                  LinearAngle(0, 7, 8, axis=0), LinearAngle(0, 7, 8, axis=1),
                  LinearAngle(0, 9, 10, axis=0), LinearAngle(0, 9, 10, axis=1),
                  LinearAngle(0, 11, 12, axis=0),
                  LinearAngle(0, 11, 12, axis=1), Improper(0, 1, 3, 5)]
    coord_set = InternalCoordinates(primitives)
    # Check that the backtransformation does not fail
    # Only works after slightly decreasing the step size, fails if scaling
    # factor is omitted
    _ = coord_set.to_cartesians(0.95*dq, xyzs_ref)


def test_misc_6_failure(resource_path_root):
    """For this structure the default threshold used in DLCs build function
    yields a redundant set of coordinates."""
    atoms = ase.io.read(resource_path_root / 'previous_failures'
                        / 'co_ii_misc_6_s_2.xyz')
    # To get the correct number use kwarg: coord_kwargs=dict(threshold=1e-7)
    coord_sys = get_coordinate_system(atoms, 'dlc')
    # Two redundant coordinates, therefore 3N - 4
    assert coord_sys.size() == 3*len(atoms) - 4
