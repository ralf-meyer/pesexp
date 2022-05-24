import numpy as np
from pesexp.optimizers.optimizers import InternalCoordinatesOptimizer
from pesexp.ml_calculators.gpr_calculator import GPRCalculator
from pesexp.ml_calculators.kernels import RBFKernel


class RestrictedVarianceOptimizer(InternalCoordinatesOptimizer):

    def initialize(self):
        if np.size(self.H0) == 1:
            H0 = np.eye(self.coord_set.size()) * self.H0
        else:
            H0 = self.coord_set.hessian_to_internals(
                self.atoms.get_positions(), self.H0)
        self.H = self.hessian_approx(H0)
        self.r0 = None
        self.f0 = None
        self.e0 = None

        # GPR initialization:
        kernel = RBFKernel()
        self.ml_calculator = GPRCalculator(kernel=kernel, normalize_y='max+10')
        self._hyperparameter_from_hessian()

    def _hyperparameter_from_hessian(self):
        self.ml_calculator.kernel.length_scale = 1.0

    def internal_step(self, f):
        return -f

    def update(self, r, f, r0, f0):
        if r0 is None or f0 is None:  # No update on the first iteration
            return

        dr = self.coord_set.diff_internals(r, r0).flatten()

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dg = -(f - f0)
        self.H.update(dr, dg)

        # Update GPRCalculator
        self.ml_calculator.add_data(self.atoms)
        # Update kernel hyperparameter
        self._hyperparameter_from_hessian()
        # Refit GPRCalculator
        self.ml_calculator.fit()
