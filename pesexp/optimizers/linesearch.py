import logging
import numpy as np

logger = logging.getLogger(__name__)


def backtracking(opt, c1=1e-4):
    """Creates a new subclass of the provided optimizer that enforces the
    sufficient decrease condition (first Wolfe condition) with parameter c1"""

    class BacktrackingOptimizer(opt):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.xyz_prev = self.atoms.get_positions()
            self.e_prev = np.inf
            self.f_prev = np.zeros_like(self.xyz_prev)
            self.step_prev = np.zeros_like(self.xyz_prev)

        def step(self, f=None):
            if f is None:
                f = self.atoms.get_forces()
            e = self.atoms.get_potential_energy()

            # Check sufficient decrease condition:
            # Scaling parameter alpha and step direction p are combined in
            # self.step_prev. The minus sign is due to the use of force instead
            # of gradient. np.vdot is used to flattten the arrays f_prev and step_prev
            if e < self.e_prev - self.c1 * np.vdot(self.f_prev, self.step_prev):
                # Last step was accepted: save current values
                self.e_prev = e
                self.f_prev = f
                self.xyz_prev = self.atoms.get_positions()
                # Take unscaled step:
                super().step()
                # Save step
                self.step_prev = self.atoms.get_positions() - self.xyz_prev
            else:  # Retake last step with smaller alpha
                logger.info(
                    f"Rejecting step {self.nsteps} - delta E = {e - self.e_prev:.6f}"
                )
                self.step_prev = 0.5 * self.step_prev
                self.atoms.set_positions(self.xyz_prev + self.step_prev)

    BacktrackingOptimizer.c1 = c1
    return BacktrackingOptimizer
