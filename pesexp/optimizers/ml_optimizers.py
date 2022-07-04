from pesexp.hessians.hessian_approximations import BFGSHessian
from pesexp.optimizers.optimizers import InternalCoordinatesOptimizer
from pesexp.optimizers import BFGS


class GPROptimizer(InternalCoordinatesOptimizer):
    hessian_approx = BFGSHessian

    def initialize(self):
        return super().initialize()
        # Initialize GPR

    def internal_step(self, f):
        ml_atoms = self.atoms.copy()
        ml_atoms.calc = self.ml_calc
        opt = BFGS(ml_atoms, coordinate_set=self.coord_set)
        opt.run(0.01)
        return self.coord_set.diff_internals(
            ml_atoms.get_positions(), self.atoms.get_positions()
        )

    def update(self, r, f, r0, f0):
        super().update(r, f, r0, f0)
        # Determine hyperparameters

        # Fit GPR
