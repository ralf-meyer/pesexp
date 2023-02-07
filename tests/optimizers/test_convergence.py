import pytest
import ase.atoms
import ase.calculators.emt
from pesexp.optimizers import BFGS
from pesexp.optimizers.convergence import (
    custom_convergence,
    terachem_convergence,
    baker_convergence,
)


@pytest.mark.parametrize(
    "decorator", [custom_convergence, terachem_convergence, baker_convergence]
)
def test_convergence_criteria(decorator):
    atoms = ase.atoms.Atoms(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    atoms.calc = ase.calculators.emt.EMT()

    opt = decorator(BFGS)(atoms)
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
