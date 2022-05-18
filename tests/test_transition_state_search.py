import pytest
import numpy as np
import ase.atoms
import ase.build
import ase.calculators
from pesexp.geometry.connectivity import get_primitives
from pesexp.geometry.primitives import (Distance, Angle, Improper,
                                        Dihedral, Cartesian)
from pesexp.geometry.coordinate_systems import (InternalCoordinates,
                                                DelocalizedCoordinates)
from pesexp.optimizers import RFO, PRFO
from pesexp.hessians.hessian_guesses import numerical_hessian
from pesexp.calculators import (CerjanMillerSurface,
                                AdamsSurface,
                                MuellerBrownSurface)


@pytest.mark.parametrize('optimizer,mu', [(RFO, 1), (PRFO, 0)])
def test_transition_state_cerjan_miller_surface(optimizer, mu):
    atoms = ase.atoms.Atoms(positions=np.array([[0.05, 0.05, 0.]]))
    atoms.calc = CerjanMillerSurface()
    coord_set = InternalCoordinates([Cartesian(0, axis=0),
                                     Cartesian(0, axis=1)])
    H = numerical_hessian(atoms)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.005, steps=100)
    assert opt.converged()
    q = coord_set.to_internals(atoms.get_positions())
    # q[0] should be close to 1
    assert abs(abs(q[0]) - 1.0) < 1e-2
    # q[1] should be close to 0
    assert abs(q[1]) < 1e-2


@pytest.mark.parametrize('optimizer,mu', [(RFO, 1), (PRFO, 0)])
def test_transition_state_adams_surface(optimizer, mu, atol=1e-2):
    # Should end up at transition state (2.2410, 0.4419)
    atoms = ase.atoms.Atoms(positions=np.array([[-0.05, 0.05, 0.]]))
    atoms.calc = AdamsSurface()
    coord_set = InternalCoordinates([Cartesian(0, axis=0),
                                     Cartesian(0, axis=1)])
    H = numerical_hessian(atoms)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.005, steps=100)
    assert opt.converged()
    q = coord_set.to_internals(atoms.get_positions())
    np.testing.assert_allclose(q, (2.2410, 0.4419), atol)

    # Should end up at transition state (-0.1985, -2.2793)
    atoms.set_positions(np.array([[0.05, -0.05, 0.]]))
    H = numerical_hessian(atoms)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.005, steps=100)
    assert opt.converged()
    q = coord_set.to_internals(atoms.get_positions())
    np.testing.assert_allclose(q, (-0.1985, -2.2793), atol)


@pytest.mark.parametrize('optimizer,mu', [(RFO, 1), (PRFO, 0)])
def test_transition_state_mueller_brown_surface(optimizer, mu, atol=1e-2):
    # Should end up at transition state (-0.8220, 0.6243)
    atoms = ase.atoms.Atoms(positions=np.array([[-0.1, 0.5, 0.]]))
    atoms.calc = MuellerBrownSurface()
    coord_set = InternalCoordinates([Cartesian(0, axis=0),
                                     Cartesian(0, axis=1)])
    H = numerical_hessian(atoms)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.005, steps=100)
    assert opt.converged()
    q = coord_set.to_internals(atoms.get_positions())
    np.testing.assert_allclose(q, (-0.8220, 0.6243), atol)

    # Start from other side
    atoms.set_positions(np.array([[-1,  0.7, 0.]]))
    H = numerical_hessian(atoms)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.005, steps=100)
    assert opt.converged()
    q = coord_set.to_internals(atoms.get_positions())
    np.testing.assert_allclose(q, (-0.8220, 0.6243), atol)


@pytest.mark.parametrize('optimizer,mu', [(RFO, 1), (PRFO, 0)])
def test_transition_state_ammonia(optimizer, mu):
    atoms = ase.build.molecule('NH3')
    XTB = pytest.importorskip('xtb.ase.calculator.XTB')
    atoms.calc = XTB(method='GFN2-xTB')

    # The plane spanned by the H atoms is at z=-.27
    atoms.positions[0, :] = [0., 0., -0.1]

    primitives = [Distance(0, 1), Distance(0, 2), Distance(0, 3),
                  Angle(1, 0, 2), Angle(2, 0, 3), Improper(0, 1, 2, 3)]
    H = numerical_hessian(atoms)
    coord_set = InternalCoordinates(primitives)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.001, steps=100)
    assert opt.converged()
    # Assert that the geometry is close to planar
    assert np.abs(primitives[-1].value(atoms.get_positions())) < 1e-2


@pytest.mark.parametrize('optimizer,mu', [(RFO, 1), (PRFO, 0)])
def test_transition_state_ethane(optimizer, mu):
    atoms = ase.build.molecule('C2H6')
    XTB = pytest.importorskip('xtb.ase.calculator.XTB')
    atoms.calc = XTB(method='GFN2-xTB')

    # Rotate one methyl group by 30 degrees about the z axis
    rot_mat = np.array([[np.sqrt(3)/2, -0.5, 0.],
                        [0.5, np.sqrt(3)/2, 0.],
                        [0., 0., 1.]])
    atoms.positions[2:5] = atoms.positions[2:5].dot(rot_mat)

    xyzs = atoms.get_positions()
    primitives = get_primitives(atoms)
    H = numerical_hessian(atoms)
    coord_set = DelocalizedCoordinates(primitives, xyzs)
    opt = optimizer(atoms, coordinate_set=coord_set, H0=H, mu=mu, maxstep=0.05)
    opt.run(fmax=0.005, steps=100)

    assert opt.converged()
    # Assert that the two methly groups have aligned
    assert abs(Dihedral(3, 0, 1, 6).value(atoms.get_positions())) < 1e-2
    assert abs(Dihedral(2, 0, 1, 7).value(atoms.get_positions())) < 1e-2
    assert abs(Dihedral(4, 0, 1, 5).value(atoms.get_positions())) < 1e-2
