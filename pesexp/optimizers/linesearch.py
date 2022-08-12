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

        def step(self, f=None):
            if f is None:
                f = self.atoms.get_forces()
            e = self.atoms.get_potential_energy()

            step_prev = self.atoms.get_positions() - self.xyz_prev
            # Check sufficient decrease condition:
            # Scaling parameter alpha and step direction p are combined in
            # step_prev. The minus sign is due to the use of force instead
            # of gradient. np.vdot is used to flattten the arrays f_prev and step_prev
            if e > self.e_prev - self.c1 * np.vdot(self.f_prev, step_prev):
                # Retake last step with smaller alpha
                logger.info(
                    f"Rejecting step {self.nsteps} - delta E = {e - self.e_prev:.6f}"
                )
                step_internal = self.coord_set.diff_internals(
                    self.atoms.get_positions(), self.xyz_prev
                )
                self.atoms.set_positions(
                    self.coord_set.to_cartesians(0.5 * step_internal, self.xyz_prev)
                )
                return
            # Last step was accepted: save current values
            self.e_prev = e
            self.f_prev = f
            self.xyz_prev = self.atoms.get_positions()
            # Take unscaled step:
            super().step()

    BacktrackingOptimizer.c1 = c1
    return BacktrackingOptimizer


def banerjee_step_control(opt, threshold=0.3, min_reduction=0.9):
    class BanerjeeStepControlledOptimizer(opt):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.xyz_prev = self.atoms.get_positions()
            self.e_prev = None
            self.f_prev = np.zeros_like(self.xyz_prev)

        def step(self, f=None):
            if f is None:
                f = self.atoms.get_forces()
            e = self.atoms.get_potential_energy()

            # In the first step e_prev is None and we do not need to check any other
            # conditions
            if self.e_prev is not None:
                actual_delta_e = e - self.e_prev
                f_prev_internal = self.coord_set.force_to_internals(
                    self.xyz_prev, self.f_prev
                )
                step_internal = self.coord_set.diff_internals(
                    self.atoms.get_positions(), self.xyz_prev
                )
                # Eq. (7) in Banerjee et al. (but in internal coordinates)
                model_delta_e = (
                    np.dot(step_internal, -f_prev_internal)
                    + 0.5
                    * np.dot(
                        step_internal,
                        np.dot(self.H, step_internal),
                    )
                ) / (1.0 + np.dot(step_internal, step_internal))
                # Note that this is different from Banerjee et al since we are dividing
                # by the actual delta E.
                delta = np.abs(actual_delta_e - model_delta_e) / np.abs(actual_delta_e)
                logger.debug(
                    f"Step {self.nsteps}: delta E, "
                    f"actual {actual_delta_e:.5f}, "
                    f"model {model_delta_e:.5f} - "
                    f"relative error of delta E = {delta:.2f}"
                )
                norm_x = np.linalg.norm(step_internal)
                # If relative error of model predicted energy change exceeds the
                # treshold (and the previous step was larger than 1e-4)
                if delta > self.threshold and norm_x > 1e-4:
                    # Retake last step with smaller step length
                    logger.info(
                        f"Rejecting step {self.nsteps} - "
                        f"relative error of delta E = {delta:.2f} > threshold"
                    )
                    u = step_internal / norm_x
                    g = np.dot(-f_prev_internal, u)
                    h = np.dot(u, np.dot(self.H, u))
                    M = 6 * np.abs(actual_delta_e - model_delta_e) / norm_x**3
                    logger.debug(f"Parameters: g = {g:.3f}, h = {h:.3f}, M = {M:.3f}")
                    t_vec = np.linspace(0, norm_x, 101)
                    inds = np.where(
                        M * t_vec**2 / 6
                        <= self.threshold * np.abs(g + 0.5 * h * t_vec)
                    )
                    t = t_vec[inds[0][-1]]
                    # Ensure that there is always some reduction:
                    t = max(1e-4, min(t, self.min_reduction * norm_x))
                    logger.debug(
                        f"Rescaling last step by {t/norm_x:.3f}, "
                        f"new norm_x = {t:.4f} "
                        f"original norm_x = {norm_x:.4f}"
                    )
                    self.atoms.set_positions(
                        self.coord_set.to_cartesians(t * u, self.xyz_prev)
                    )
                    return
            # Last step was accepted: save current values
            self.e_prev = e
            self.f_prev = f
            self.xyz_prev = self.atoms.get_positions()
            # Take unscaled step:
            super().step()
            # Save step
            self.step_prev = self.atoms.get_positions() - self.xyz_prev

    BanerjeeStepControlledOptimizer.threshold = threshold
    BanerjeeStepControlledOptimizer.min_reduction = min_reduction
    return BanerjeeStepControlledOptimizer
