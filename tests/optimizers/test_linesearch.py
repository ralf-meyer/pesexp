import ase.atoms
import ase.calculators.emt
import numpy as np
from pesexp.calculators.calculators import ThreeDCalculator
from pesexp.optimizers import BFGS, PRFO
from pesexp.optimizers.linesearch import backtracking, banerjee_step_control
from pesexp.geometry.primitives import Cartesian
from pesexp.geometry.coordinate_systems import InternalCoordinates


def test_backtracking():
    """This test uses a bad guess for the initial hessian to provoke large step
    sizes and tests that the optimization converges if backtracking line search
    is used."""
    atoms = ase.atoms.Atoms(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    atoms.calc = ase.calculators.emt.EMT()

    opt = backtracking(BFGS)(atoms, H0=1.0)
    opt.run(steps=100)
    assert opt.converged()


def test_banerjee_step_control():
    class DummyCalc(ThreeDCalculator):
        def energy(self, x, y, z):
            return x**4 - y**2 + 0.001 * z**2

        def gradient(self, x, y, z):
            return 4 * x**3, -2 * y, 0.002 * z

    atoms = ase.atoms.Atoms(positions=np.array([[1.0, 1.0, 1.0]]))
    atoms.calc = DummyCalc()
    coord_set = InternalCoordinates(
        [Cartesian(0, axis=0), Cartesian(0, axis=1), Cartesian(0, axis=2)]
    )

    # Starting from exact Hessian
    H0 = np.array([[12.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.002]])
    opt = banerjee_step_control(PRFO)(
        atoms, coordinate_set=coord_set, H0=H0, maxstep=0.05
    )
    opt.run(fmax=0.001, steps=100)
    assert opt.converged()
