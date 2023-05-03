from abc import abstractmethod
import logging
import numpy as np
import ase.optimize
import ase.units
from scipy.optimize import brentq
from pesexp.geometry.coordinate_systems import CartesianCoordinates
from pesexp.hessians.hessian_approximations import (
    HessianApproximation,
    BFGSHessian,
    BofillHessian,
)
from pesexp.utils.exceptions import ConvergenceError

logger = logging.getLogger(__name__)


class InternalCoordinatesOptimizer(ase.optimize.optimize.Optimizer):
    # Use abstract base class as placeholder here
    hessian_approx = HessianApproximation
    defaults = {
        **ase.optimize.optimize.Optimizer.defaults,
        "maxstep_internal": 1.0,
        "H0": 70.0,
    }

    def __init__(
        self,
        atoms,
        coordinate_set=None,
        restart=None,
        logfile="-",
        trajectory=None,
        master=None,
        H0=None,
        maxstep=None,
        maxstep_internal=None,
    ):
        if coordinate_set is None:
            self.coord_set = CartesianCoordinates(atoms)
        else:
            self.coord_set = coordinate_set
        if H0 is None:
            self.H0 = self.defaults["H0"]
        else:
            self.H0 = H0

        if maxstep is None:
            self.maxstep = self.defaults["maxstep"]
        else:
            self.maxstep = maxstep
        if maxstep_internal is None:
            self.maxstep_internal = self.defaults["maxstep_internal"]
        else:
            self.maxstep_internal = maxstep_internal
        ase.optimize.optimize.Optimizer.__init__(
            self, atoms, restart, logfile, trajectory, master
        )

    def initialize(self):
        if np.size(self.H0) == 1:
            H0 = np.eye(self.coord_set.size()) * self.H0
        else:
            H0 = self.coord_set.hessian_to_internals(
                self.atoms.get_positions(), self.H0
            )
        self.H = self.hessian_approx(H0)
        self.r0 = None
        self.f0 = None
        self.e0 = None

    def irun(self, fmax=0.05, steps=None):
        """call Dynamics.irun and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        for converged in ase.optimize.optimize.Dynamics.irun(self):
            # let the user inspect the step and change things before logging
            # and predicting the next step
            yield converged
            if not converged:
                self.predict_next_step()

    def run(self, fmax=0.05, steps=None):
        for converged in InternalCoordinatesOptimizer.irun(
            self, fmax=fmax, steps=steps
        ):
            pass
        return converged

    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()

        r = self.atoms.get_positions()
        e = self.atoms.get_potential_energy()

        self.atoms.set_positions(r + self.next_step)
        self.r0 = r.copy()
        self.f0 = f.copy()
        self.e0 = e
        self.dump(
            (
                self.coord_set,
                self.H,
                self.r0,
                self.f0,
                self.e0,
                self.maxstep,
                self.maxstep_internal,
            )
        )

    def predict_next_step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()
        r = self.atoms.get_positions()
        # Transform forces to internal coordinates
        f_int = self.coord_set.force_to_internals(r, f.flatten())
        self.update(r, f, self.r0, self.f0)

        # step in internals
        dq = self.internal_step(f_int)
        # Rescale
        maxsteplength_internal = np.max(np.abs(dq))
        if maxsteplength_internal > self.maxstep_internal:
            dq *= self.maxstep_internal / maxsteplength_internal

        # Transform to Cartesians
        dr = self.coord_set.to_cartesians(dq, r) - r
        # Rescale
        maxsteplength = np.max(np.sqrt(np.sum(dr**2, axis=1)))
        if maxsteplength > self.maxstep:
            dr *= self.maxstep / maxsteplength
        self.next_step = dr

    @abstractmethod
    def internal_step(self, f):
        """this needs to be implemented by subclasses"""

    def read(self):
        (
            self.coord_set,
            self.H,
            self.r0,
            self.f0,
            self.e0,
            self.maxstep,
            self.maxstep_internal,
        ) = self.load()

    def update(self, r, f, r0, f0):
        if r0 is None or f0 is None:  # No update on the first iteration
            return

        dr = self.coord_set.diff_internals(r, r0).flatten()

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        logger.debug(f"Step {self.nsteps}: Updating the Hessian approximation.")
        dg = -(
            self.coord_set.force_to_internals(r, f.flatten())
            - self.coord_set.force_to_internals(r0, f0.flatten())
        )
        # dg = -(f - f0)
        self.H.update(dr, dg)


class NewtonRaphson(InternalCoordinatesOptimizer):
    """Can not be instantiated because of the missing Hessian approximation
    that must be added in subclasses."""

    def internal_step(self, f):
        omega, V = np.linalg.eigh(self.H)
        # Only take step in the direction of non-zero eigenvalues
        non_zero = np.abs(omega) > 1e-6
        logger.debug(
            "NewtonRaphson step: first 3 eigenvalues are "
            f"{' '.join([f'{o:.3E}' for o in omega[:3]])}"
        )
        return np.dot(V[:, non_zero], np.dot(f, V[:, non_zero]) / omega[non_zero])


class BFGS(NewtonRaphson):
    hessian_approx = BFGSHessian


class RFO(InternalCoordinatesOptimizer):
    def __init__(self, *args, mu=0, **kwargs):
        self.mu = mu
        # If the abstract base class has not been replaced, e.g. by subclassing:
        if self.hessian_approx == HessianApproximation:
            if self.mu == 0:
                self.hessian_approx = BFGSHessian
            else:
                self.hessian_approx = BofillHessian
        InternalCoordinatesOptimizer.__init__(self, *args, **kwargs)

    def calc_shift_parameter(self, f_trans, omega, mu: int = 0) -> float:
        # Use scipys implementation of Brent's root finding method to solve eq. (13)
        # in Banerjee et al. This is preferred over a derivative based method such
        # as Newtons method because of the discontinuities in the target function
        # and because it allows to ensure the bracketing property.

        f_squared = f_trans**2

        # Analytical solution for eq. (13) if the sum reduces to a single value
        if len(omega) == 1:
            sign = -1.0 if mu == 0 else 1.0
            lamb = 0.5 * omega[0] + sign * np.sqrt(0.25 * omega[0] ** 2 + f_squared[0])
            return lamb

        # Define the target function i.e. eq. (13)
        def target(x):
            return np.sum(f_squared / (x - omega)) - x

        # Find the correct interval based on the value of mu
        if 0 < mu < len(omega):
            # Solutions need to satisfy the bracketing property:
            a = omega[mu - 1]
            b = omega[mu]
        else:
            # Since the second part of the bracket in this case is +- inf a different
            # approach to find initial values for a and b is needed. The case where all
            # omegas are degenerate will yield the largest magnitude shift parameter.
            # Note, as additional safety factor to avoid numerical problems, this limit
            # is calculated using a different slope for 2 the r.h.s of the equation.
            # First figure out if we are maximizing or minimizing (to select the correct
            # reference eigenvalue and solution of a quadratic equation):
            if mu == 0:  # Minimization
                a = 0.5 * omega[0] - np.sqrt(0.25 * omega[0] ** 2 + 2 * f_squared.sum())
                assert target(a) > 0
                b = omega[0]
            else:  # Maximization
                a = omega[-1]
                b = 0.5 * omega[-1] + np.sqrt(0.25 * omega[-1] ** 2 + 2 * f_squared[-1])
                assert target(b) < 0

        # bisection search until both a and b have been updated (to ensure that the
        # value of the target function is finite). Note this only work for this specific
        # target function where the function is monotonically decresing in every
        # possible interval (a, b).
        def bisection_search(fun, a, b):
            last_updated = "None"
            for _ in range(100):
                x = 0.5 * (a + b)
                fun_x = fun(x)
                if not np.isfinite(fun_x):
                    raise ValueError("Bisection search stuck on infinite value")
                if fun_x > 0:  # Update a
                    a = x
                    if last_updated == "b":
                        return a, b
                    last_updated = "a"
                else:  # Update b
                    b = x
                    if last_updated == "a":
                        return a, b
                    last_updated = "b"
            raise ConvergenceError(
                "Bisection search not converged within 100 iterations"
            )

        a, b = bisection_search(target, a, b)
        try:
            lamb, result = brentq(target, a, b, full_output=True)
        except ValueError as m:
            logger.error(f"{a} {b} {target(a)} {target(b)} {f_squared.sum()} {mu}")
            logger.error(f_trans)
            logger.error(omega)
            raise m

        if not result.converged:
            logger.error(result)
            raise ConvergenceError("RFO shift parameter calculation failed")
        return lamb

    def construct_step(self, f_trans, V, omega, lamb):
        shifted_omega = omega - lamb
        # Only take steps in directions where the eigenvalues are larger than a fixed
        # threshold
        mask = np.abs(shifted_omega) > 1e-6
        return np.einsum(
            "j,ij,j->i", f_trans[mask], V[:, mask], 1 / shifted_omega[mask]
        )

    def internal_step(self, f):
        # Here we use equation (13) in Banerjee et al. (1984) to determine
        # the shift parameter using the iterative procedure defined in
        # calc_shift_parameter().
        omega, V = np.linalg.eigh(self.H)
        # Transform the force vector to the eigenbasis of the Hessian
        f_trans = np.dot(f, V)

        lamb = self.calc_shift_parameter(f_trans, omega, self.mu)

        logger.debug(
            "RFO step: first 3 eigenvalues are "
            f"{' '.join([f'{o:.3E}' for o in omega[:3]])}, "
            f"lambda = {lamb:.5E}"
        )

        if self.mu == 0:
            assert lamb < omega[0]
        elif self.mu == len(omega):
            assert lamb > omega[-1]
        else:
            assert omega[self.mu - 1] < lamb < omega[self.mu]

        return self.construct_step(f_trans, V, omega, lamb)


class PRFO(RFO):
    def __init__(self, *args, mu=1, **kwargs):
        RFO.__init__(self, *args, mu=mu, **kwargs)

    def internal_step(self, f):
        omega, V = np.linalg.eigh(self.H)
        # Transform the force vector to the eigenbasis of the Hessian
        f_trans = np.dot(f, V)
        # Partition into two subproblems.
        # The coordinates mu are maximized.
        max_ind = slice(0, self.mu)
        # The remaining coordinates that are minimized.
        min_ind = slice(self.mu, len(omega))

        # Calculate two separate shift parameters for maximization
        lamb_p = self.calc_shift_parameter(
            f_trans[max_ind], omega[max_ind], mu=len(omega[max_ind])
        )
        # and minimization
        lamb_n = self.calc_shift_parameter(f_trans[min_ind], omega[min_ind], mu=0)

        logger.debug(
            "pRFO step: first 3 eigenvalues are "
            f"{' '.join([f'{o:.3E}' for o in omega[:3]])}, "
            f"lambda_p = {lamb_p:.5E}, lambda_n = {lamb_n:.5E}"
        )

        # Construct full step by combining maximization and minimization steps
        step = self.construct_step(
            f_trans[max_ind], V[:, max_ind], omega[max_ind], lamb_p
        )
        step += self.construct_step(
            f_trans[min_ind], V[:, min_ind], omega[min_ind], lamb_n
        )
        return step


class LBFGS(ase.optimize.LBFGS):
    """Adaptation of ASEs implementation of the LBFGS optimizer to allow for
    arbitrary (internal) coordinate systems.
    """

    def __init__(self, atoms, coordinate_set, maxstep_internal=1.0, **kwargs):
        if kwargs.get("use_line_search", False):
            raise NotImplementedError("Line search is not implemented yet.")
        ase.optimize.LBFGS.__init__(self, atoms, **kwargs)
        self.coord_set = coordinate_set
        self.maxstep_internal = maxstep_internal

    def step(self, f=None):
        """Take a single step

        This is where actual modifications to account for internal coordinates
        have to be made. Modifications are highlighted by a # MOD comment.

        Use the given forces, update the history and calculate the next step --
        then take it"""

        if f is None:
            f = self.atoms.get_forces()

        r = self.atoms.get_positions()

        # MOD: Transform forces to internal coordinates
        f = self.coord_set.force_to_internals(r, f.flatten())
        # print('q internal', self.coord_set.to_internals(r))
        # print('f internal', f)

        self.update(r, f, self.r0, self.f0)

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty((loopmax,), dtype=np.float64)

        # ## The algorithm itself:
        q = -f.reshape(-1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = self.rho[i] * np.dot(self.s[i], q)
            q -= a[i] * self.y[i]
        z = self.H0 * q

        for i in range(loopmax):
            b = self.rho[i] * np.dot(self.y[i], z)
            z += self.s[i] * (a[i] - b)

        # MOD: Calculate internal step
        dq = -z
        if np.max(np.abs(dq)) > self.maxstep_internal:
            dq *= self.maxstep_internal / np.max(np.abs(dq))
        # MOD: Transform internal step to Cartesian step
        self.p = self.coord_set.to_cartesians(dq, r) - r
        # ##

        g = -f
        # MOD: Removed option for linesearch
        dr = self.determine_step(self.p) * self.damping
        # MOD: xyzs instead of r
        self.atoms.set_positions(r + dr)

        self.iteration += 1
        self.force_calls += 1
        self.function_calls += 1
        self.r0 = r
        self.f0 = -g
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )

    def update(self, r, f, r0, f0):
        """Update everything that is kept in memory

        This function is mostly here to allow for replay_trajectory.
        """
        if self.iteration > 0:
            # MOD: step s should be calculated in internals
            s0 = self.coord_set.diff_internals(r, r0)
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0.reshape(-1) - f.reshape(-1)
            self.y.append(y0)

            rho0 = 1.0 / np.dot(y0, s0)
            self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def replay_trajectory(self, traj):
        raise NotImplementedError("Trajectory replay is not yet supported!")
