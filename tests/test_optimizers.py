import pytest
import numpy as np
import ase.atoms
import ase.build
import ase.calculators.emt
import ase.optimize
from pesexp.calculators import OpenbabelFF
from pesexp.geometry.connectivity import get_primitives
from pesexp.geometry.primitives import Distance
from pesexp.geometry.coordinate_systems import (CartesianCoordinates,
                                                DelocalizedCoordinates,
                                                InternalCoordinates,
                                                ApproximateNormalCoordinates)
from pesexp.optimizers import BFGS, LBFGS, RFO
from pesexp.optimizers.convergence import ConvergenceMixin, TerachemConvergence
from rmsd import kabsch_rmsd


@pytest.mark.parametrize('optimizer', [BFGS, LBFGS, RFO])
def test_optimizers_on_H2(optimizer):
    """Separate test case because H2 is not supported by MMFF94"""
    atoms = ase.atoms.Atoms(['H', 'H'], positions=[[0., 0., 0.],
                                                   [.5, .5, .5]])
    atoms.calc = ase.calculators.emt.EMT()

    # Reference calculation in Cartesian coordinates
    atoms_ref = atoms.copy()
    atoms_ref.calc = ase.calculators.emt.EMT()
    opt_ref = ase.optimize.BFGS(atoms_ref)
    opt_ref.run(fmax=0.01, steps=100)
    xyzs_ref = atoms_ref.get_positions()
    r_ref = np.linalg.norm(xyzs_ref[0] - xyzs_ref[1])

    coord_set = InternalCoordinates([Distance(0, 1)])
    opt = optimizer(atoms, coord_set)
    opt.run(fmax=0.01, steps=100)
    assert opt.converged()
    # Test that the final bond length correct
    xyzs = atoms.get_positions()
    r = np.linalg.norm(xyzs[0] - xyzs[1])
    assert abs(r - r_ref) < 1e-3


@pytest.mark.parametrize('optimizer', [BFGS, LBFGS, RFO])
def test_max_step_internal(optimizer, maxstep=0.05, atol=1e-8):
    atoms = ase.atoms.Atoms(['H', 'H'], positions=[[0., 0., 0.],
                                                   [1.5, 0., 0.]])
    atoms.calc = ase.calculators.emt.EMT()
    coord_set = InternalCoordinates([Distance(0, 1)])
    # LBFGS raises value error is maxstep>1.0
    opt = optimizer(atoms, coord_set, maxstep=1.0, maxstep_internal=maxstep)

    previous_positions = atoms.get_positions()
    step_lengths = []
    for _ in opt.irun(fmax=1e-4, steps=300):
        pos = atoms.get_positions()
        dq = coord_set.diff_internals(pos, previous_positions)
        step_lengths.append(np.max(np.abs(dq)))
        previous_positions = pos
    step_lengths = np.array(step_lengths)
    # Assert all steps are smaller than maxstep (within numerical accuracy)
    assert all(step_lengths < maxstep + atol)
    # Assert that at least one step was close to the maxlength
    # (otherwise this would be a bad test)
    assert any(abs(step_lengths - maxstep) < 1e-3)
    assert opt.converged()


@pytest.mark.parametrize('optimizer', [BFGS, LBFGS, RFO])
def test_max_step_cartesian(optimizer, maxstep=0.05, atol=1e-8):
    atoms = ase.atoms.Atoms(['H', 'H'], positions=[[0., 0., 0.],
                                                   [1.5, 0., 0.]])
    atoms.calc = ase.calculators.emt.EMT()
    coord_set = InternalCoordinates([Distance(0, 1)])
    opt = optimizer(atoms, coord_set, maxstep=maxstep, maxstep_internal=0.5)

    previous_positions = atoms.get_positions()
    step_lengths = []
    for _ in opt.irun(fmax=1e-4, steps=300):
        pos = atoms.get_positions()
        step_lengths.append(np.max(np.abs(pos - previous_positions)))
        previous_positions = pos
    step_lengths = np.array(step_lengths)
    # Assert all steps are smaller than maxstep (within numerical accuracy)
    assert all(step_lengths < maxstep + atol)
    # Assert that at least one step was close to the maxlength
    # (otherwise this would be a bad test)
    assert any(abs(step_lengths - maxstep) < 1e-3)
    assert opt.converged()


@pytest.mark.parametrize('mixin', [ConvergenceMixin,
                                   TerachemConvergence])
def test_convergence_criteria(mixin):
    atoms = ase.atoms.Atoms(['H', 'H'], positions=[[0., 0., 0.],
                                                   [.5, .5, .5]])
    atoms.calc = ase.calculators.emt.EMT()

    class TestOptimizer(mixin, BFGS):
        pass

    opt = TestOptimizer(atoms)
    # Check that both run and irun work
    for _ in opt.irun(steps=2):
        pass
    opt.run(steps=100)
    assert opt.converged()

    # Assert that the fmax kwarg was properly removed
    with pytest.raises(TypeError):
        opt.run(fmax=0.05)
    with pytest.raises(TypeError):
        opt.irun(fmax=0.05)


@pytest.mark.parametrize('optimizer', [BFGS, LBFGS, RFO])
@pytest.mark.parametrize('mol', ['H2O', 'NH3', 'CH4', 'C2H4', 'C2H6',
                                 'C6H6', 'butadiene', 'bicyclobutane'])
@pytest.mark.parametrize('coord_set', ['cart', 'internal', 'dlc', 'anc'])
def test_optimizers_on_organic_molecules(optimizer, mol, coord_set):
    # check if openbabel version > 3.0. This is necessary as
    # OBForceField.GetGradient is not public for prior versions.
    pytest.importorskip('openbabel', minversion='3.0')

    atoms = ase.build.molecule(mol)
    # slightly distort the molecules
    xyzs = atoms.get_positions()
    rng = np.random.default_rng(1234)
    xyzs += rng.normal(scale=0.05, size=xyzs.shape)
    atoms.set_positions(xyzs)
    atoms.calc = OpenbabelFF(ff='MMFF94')

    # Reference calculation in Cartesian coordinates
    atoms_ref = atoms.copy()
    atoms_ref.calc = OpenbabelFF(ff='MMFF94')

    primitives = get_primitives(atoms)
    if coord_set == 'cart':
        coord_set = CartesianCoordinates(atoms)
    elif coord_set == 'internal':
        coord_set = InternalCoordinates(primitives)
    elif coord_set == 'dlc':
        coord_set = DelocalizedCoordinates(primitives, xyzs)
    elif coord_set == 'anc':
        coord_set = ApproximateNormalCoordinates(atoms)

    opt_ref = ase.optimize.BFGS(atoms_ref)
    opt = optimizer(atoms, coord_set)

    opt_ref.run(fmax=0.001, steps=100)
    opt.run(fmax=0.001, steps=100)
    assert opt.converged()

    assert kabsch_rmsd(atoms.get_positions(),
                       atoms_ref.get_positions(),
                       translate=True) < 1e-2


@pytest.mark.parametrize('optimizer', [BFGS, LBFGS, RFO])
@pytest.mark.parametrize('ligand', ['water'])
@pytest.mark.parametrize('coord_set', ['cart', 'internal', 'dlc', 'anc'])
def test_optimizers_on_homoleptic_TMCs(resource_path_root, optimizer, ligand,
                                       coord_set):
    """TODO: For now only works on water since UFF does not give reasonable
    results for the other ligands."""
    # check if openbabel version > 3.0. This is necessary as
    # OBForceField.GetGradient is not public for prior versions.
    pytest.importorskip('openbabel', minversion='3.0')

    atoms = ase.io.read(
        resource_path_root / f'homoleptic_octahedrals/Co_II_{ligand}.xyz')
    xyzs = atoms.get_positions()
    atoms.calc = OpenbabelFF(ff='UFF')

    # Reference calculation in Cartesian coordinates
    atoms_ref = atoms.copy()
    atoms_ref.calc = OpenbabelFF(ff='UFF')

    primitives = get_primitives(atoms)
    if coord_set == 'cart':
        coord_set = CartesianCoordinates(atoms)
    elif coord_set == 'internal':
        coord_set = InternalCoordinates(primitives)
    elif coord_set == 'dlc':
        coord_set = DelocalizedCoordinates(primitives, xyzs)
    elif coord_set == 'anc':
        coord_set = ApproximateNormalCoordinates(atoms)

    opt_ref = ase.optimize.BFGS(atoms_ref)
    opt_ref.run(fmax=0.001, steps=100)

    opt = optimizer(atoms, coord_set)
    opt.run(fmax=0.001, steps=100)
    assert opt.converged()

    assert kabsch_rmsd(atoms.get_positions(),
                       atoms_ref.get_positions(),
                       translate=True) < 1e-2
