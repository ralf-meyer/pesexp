import ase.atoms
import ase.calculators.emt
from pesexp.optimizers import BFGS
from pesexp.optimizers.linesearch import backtracking


def test_convergence_criteria():
    """This test uses a bad guess for the initial hessian to provoke large step
    sizes and tests that the optimization converges if backtracking line search
    is used."""
    atoms = ase.atoms.Atoms(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    atoms.calc = ase.calculators.emt.EMT()

    opt = backtracking(BFGS)(atoms, H0=1.0)
    opt.run(steps=100)
    assert opt.converged()
