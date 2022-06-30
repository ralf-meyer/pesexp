import numpy as np
import ase.atoms
import ase.units
from pesexp.calculators import (CerjanMillerSurface,
                                AdamsSurface,
                                MuellerBrownSurface,
                                LEPSPotential,
                                TeraChem)


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


def test_LEPSPotential():
    atoms = ase.atoms.Atoms(positions=[[0., 0., 0.]])
    atoms.calc = LEPSPotential()

    # Test that the two endpoints used in the reference are close
    # to minima (Note not actual minima):
    atoms.set_positions([[0.742, 4., 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    atoms.set_positions([[4., 0.742, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)

    # Test approximate transition state
    atoms.set_positions([[1.15, 0.862, 0.]])
    f = atoms.get_forces()
    np.testing.assert_allclose(f, np.zeros_like(f), atol=1e-2)


def test_TeraChem_read_results(resource_path_root):
    calc = TeraChem(label=str(resource_path_root / 'io/h2'))
    calc.read_results()

    np.testing.assert_allclose(calc.results['energy'],
                               -1.1754839056 * ase.units.Hartree)

    forces_ref = -np.array([[0., 0., 1.], [0., 0., -1.]])
    forces_ref *= 0.0000042754 * ase.units.Hartree/ase.units.Bohr
    np.testing.assert_allclose(calc.results['forces'],
                               forces_ref)


def test_TeraChem_write_input(resource_path_root, tmpdir):
    atoms = ase.atoms.Atoms(symbols=['H', 'H'])
    calc = TeraChem(label='h2', directory=tmpdir, method='b3lyp')
    calc.write_input(atoms)

    def lines_to_dict(lines):
        params = {}
        for line in lines:
            # Skip comments and 'end'
            if not (line.startswith('#') or 'end' in line):
                key, val = line.split()
                params[key] = val
        return params

    with open(tmpdir / 'h2.inp', 'r') as fin:
        lines = fin.readlines()
    params = lines_to_dict(lines)

    with open(resource_path_root / 'io/h2.inp', 'r') as fin:
        lines_ref = fin.readlines()
    params_ref = lines_to_dict(lines_ref)

    assert params == params_ref
