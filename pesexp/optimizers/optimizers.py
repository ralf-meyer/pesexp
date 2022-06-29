from abc import abstractmethod
import numpy as np
import ase.optimize
import ase.units
from pesexp.geometry.coordinate_systems import CartesianCoordinates
from pesexp.hessians.hessian_approximations import (BFGSHessian,
                                                    BofillHessian)


class InternalCoordinatesOptimizer(ase.optimize.optimize.Optimizer):
    defaults = {**ase.optimize.optimize.Optimizer.defaults,
                'maxstep_internal': 1.0, 'H0': 70.0}

    def __init__(self, atoms, coordinate_set=None, restart=None, logfile='-',
                 trajectory=None, master=None, H0=None,
                 maxstep=None, maxstep_internal=None):

        if coordinate_set is None:
            self.coord_set = CartesianCoordinates(atoms)
        else:
            self.coord_set = coordinate_set
        if H0 is None:
            self.H0 = self.defaults['H0']
        else:
            self.H0 = H0

        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep
        if maxstep_internal is None:
            self.maxstep_internal = self.defaults['maxstep_internal']
        else:
            self.maxstep_internal = maxstep_internal
        ase.optimize.optimize.Optimizer.__init__(self, atoms, restart, logfile,
                                                 trajectory, master)

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

    def step(self, f=None):
        if f is None:
            f = self.atoms.get_forces()

        r = self.atoms.get_positions()
        e = self.atoms.get_potential_energy()
        # MOD: Transform forces to internal coordinates
        f = self.coord_set.force_to_internals(r, f.flatten())
        # MOD: remove the flattening here
        self.update(r, f, self.r0, self.f0)

        # MOD: step in internals
        dq = self.internal_step(f)
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

        self.atoms.set_positions(r + dr)
        self.r0 = r.copy()
        self.f0 = f.copy()
        self.e0 = e
        self.dump((self.coord_set, self.H, self.r0, self.f0, self.e0,
                   self.maxstep, self.maxstep_internal))

    @abstractmethod
    def internal_step(self, f):
        """this needs to be implemented by subclasses"""

    @property
    @abstractmethod
    def hessian_approx(self):
        """subclasses need to define a hessian approximation"""

    def read(self):
        (self.coord_set, self.H, self.r0, self.f0, self.e0, self.maxstep,
         self.maxstep_internal) = self.load()

    def update(self, r, f, r0, f0):
        if r0 is None or f0 is None:  # No update on the first iteration
            return

        dr = self.coord_set.diff_internals(r, r0).flatten()

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dg = -(f - f0)
        self.H.update(dr, dg)


class NewtonRaphson(InternalCoordinatesOptimizer):
    """Can not be instantiated because of the missing Hessian approximation
    that must be added in subclasses."""

    def internal_step(self, f):
        omega, V = np.linalg.eigh(self.H)
        # Only take step in the direction of non-zero eigenvalues
        non_zero = np.abs(omega) > 1e-6
        return np.dot(V[:, non_zero],
                      np.dot(f, V[:, non_zero]) / omega[non_zero])


class BFGS(NewtonRaphson):
    hessian_approx = BFGSHessian


class RFO(InternalCoordinatesOptimizer):

    def __init__(self, *args, mu=0, **kwargs):
        self.mu = mu
        if self.mu == 0:
            self.hessian_approx = BFGSHessian
        else:
            self.hessian_approx = BofillHessian
        InternalCoordinatesOptimizer.__init__(self, *args, **kwargs)

    def internal_step(self, f):
        # extended Hessian matrix
        H_ext = np.block([[self.H, -f[:, np.newaxis]], [-f, 0.]])

        _, V = np.linalg.eigh(H_ext)

        # Step is calculated by proper rescaling of the eigenvector
        # corresponding to the mu-th eigenvalue. For minimizations
        # the lowest i.e. zeroth eigenvalue is chosen.
        return V[:-1, self.mu] / V[-1, self.mu]


class PRFO(InternalCoordinatesOptimizer):
    hessian_approx = BofillHessian

    def __init__(self, *args, mu=0, **kwargs):
        self.mu = mu
        InternalCoordinatesOptimizer.__init__(self, *args, **kwargs)

    def internal_step(self, f):
        step = np.zeros_like(f)
        omega, V = np.linalg.eigh(self.H)
        # Tranform the force vector to the eigenbasis of the Hessian
        f_trans = np.dot(f, V)
        # Partition into two subproblems.
        # The coordinates mu are maximized.
        max_ind = np.zeros_like(omega, dtype=bool)
        max_ind[self.mu] = True
        H_max = np.block([[np.diag(omega[max_ind]),
                           -f_trans[max_ind, np.newaxis]],
                          [-f_trans[max_ind], 0.]])
        _, V_max = np.linalg.eigh(H_max)
        # The remaining coordinates that are minimized.
        min_ind = np.logical_not(max_ind)
        H_min = np.block([[np.diag(omega[min_ind]),
                           -f_trans[min_ind, np.newaxis]],
                          [-f_trans[min_ind], 0.]])
        _, V_min = np.linalg.eigh(H_min)
        # Calculate the step by combining the highest eigenvector
        # from the maximization subset
        step[max_ind] = V_max[:-1, -1] / V_max[-1, -1]
        # and the lowest eigenvector from the minimization subset
        step[min_ind] = V_min[:-1, 0] / V_min[-1, 0]
        # Tranform step back to original system
        return np.dot(V, step)


class LBFGS(ase.optimize.LBFGS):
    """Adaptation of ASEs implementation of the LBFGS optimizer to allow for
    arbitrary (internal) coordinate systems.
    """

    def __init__(self, atoms, coordinate_set, maxstep_internal=1.0, **kwargs):
        if kwargs.get('use_line_search', False):
            raise NotImplementedError('Line search is not implemented yet.')
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

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty((loopmax,), dtype=np.float64)

        # ## The algorithm itself:
        q = -f.reshape(-1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * np.dot(s[i], q)
            q -= a[i] * y[i]
        z = H0 * q

        for i in range(loopmax):
            b = rho[i] * np.dot(y[i], z)
            z += s[i] * (a[i] - b)

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
        self.dump((self.iteration, self.s, self.y,
                   self.rho, self.r0, self.f0, self.e0, self.task))

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
        raise NotImplementedError('Trajectory replay is not yet supported!')
