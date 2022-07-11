import pytest
import ase.atoms
import ase.calculators.emt
from pesexp.optimizers import BFGS
from pesexp.optimizers.convergence import ConvergenceMixin, TerachemConvergence


@pytest.mark.parametrize("mixin", [ConvergenceMixin, TerachemConvergence])
def test_convergence_criteria(mixin):
    atoms = ase.atoms.Atoms(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
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
