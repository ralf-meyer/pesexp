import numpy as np
import ase.atoms
from pesexp.calculators import (CerjanMillerSurface,
                                AdamsSurface,
                                MuellerBrownSurface)


def test_cerjan_miller_surface():
    atoms = ase.atoms.Atoms(positions=[[0., 0., 0.]])
    atoms.calc = CerjanMillerSurface()
    # Test minimum at (0, 0)
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f))

    # Test transition states
    atoms.set_positions([[1.0, 0., 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f))

    atoms.set_positions([[-1.0, 0., 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f))


def test_adams_surface():
    atoms = ase.atoms.Atoms(positions=[[0., 0., 0.]])
    atoms.calc = AdamsSurface()
    # Test minimum at (0, 0)
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f))

    # Test maximum at approximately (3.8239, -4.4096)
    atoms.set_positions([[3.8239, -4.4096, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test transition state at approximately (2.2410, 0.4419)
    atoms.set_positions([[2.2410, 0.4419, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test transition state at approximately (-0.1985,-2.2793)
    atoms.set_positions([[-0.1985, -2.2793, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)


def test_mueller_brown_surface():
    atoms = ase.atoms.Atoms(positions=[[0., 0., 0.]])
    atoms.calc = MuellerBrownSurface()

    # Test minimum at approximately (-0.55822363,  1.44172584)
    atoms.set_positions([[-0.55822363,  1.44172584, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test minimum at approximately (-0.05001083,  0.4666941)
    atoms.set_positions([[-0.05001083,  0.4666941, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test minimum at approximately (0.6234994,  0.02803776)
    atoms.set_positions([[0.6234994,  0.02803776, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test transition state at approximately (-0.8220, 0.6243)
    atoms.set_positions([[-0.8220, 0.6243, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test transition state at approximately (0.2125, 0.2930)
    atoms.set_positions([[0.2125, 0.2930, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)
