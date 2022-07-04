import time
import pickle
import numpy as np
from pesexp.utils.exceptions import ConvergenceError
from pesexp.geometry.utils import check_colinear
from pesexp.geometry.connectivity import get_primitives
from pesexp.hessians.hessian_guesses import LindhHessian
from warnings import warn


def get_coordinate_system(atoms, name="cart", coord_kwargs={}):
    if name.lower() in ["cartesian", "cart"]:
        return CartesianCoordinates(atoms, **coord_kwargs)
    elif name.lower() == "dlc":
        primitives = get_primitives(atoms)
        return DelocalizedCoordinates(
            primitives, xyzs=atoms.get_positions(), **coord_kwargs
        )
    elif name.lower() == "anc":
        return ApproximateNormalCoordinates(atoms, **coord_kwargs)
    else:
        raise NotImplementedError("Unknown coordinate system {name}")


class CoordinateSystem:
    def size(self):
        """Return the number of internal coordinates. Needed for example to
        setup a correctly sized hessian matrix."""

    def to_internals(self, xyzs):
        """Transform a given Cartesian geometry to the internal representation."""

    def to_cartesians(self, dq, xyzs_ref):
        """For a given step in internal coordinates find the new Cartesian
        geometry."""

    def diff_internals(self, xyzs1, xyzs2):
        """Calculate the distance (1-2) between the internal representations of
        two Cartesian geometries. This is a separate method as some internal
        representation might need cleaning up, e.g. resticting angle to a
        specific range."""
        return self.to_internals(xyzs1) - self.to_internals(xyzs2)

    def force_to_internals(self, xyzs, force_cart):
        """Transfrom a Cartesian force vector to the internal coordinate
        system."""

    def hessian_to_internals(self, xyzs, hess_cart, grad_cart=None):
        """Transform a Cartesian Hessian matrix to the internal coordinate
        system."""


class CartesianCoordinates(CoordinateSystem):
    def __init__(self, atoms):
        self.n_atoms = len(atoms)

    def size(self):
        return 3 * self.n_atoms

    def to_internals(self, xyzs):
        return xyzs.flatten()

    def to_cartesians(self, dq, xyzs_ref):
        return xyzs_ref + dq.reshape(xyzs_ref.shape)

    def force_to_internals(self, xyzs, force_cart):
        return force_cart.flatten()

    def hessian_to_internals(self, xyzs, hess_cart, grad_cart=None):
        return hess_cart


class InternalCoordinates(CoordinateSystem):
    def __init__(self, primitives, save_failures=True):
        self.primitives = primitives
        self.save_failures = save_failures

    def size(self):
        return len(self.primitives)

    def B(self, xyzs):
        B = np.zeros((len(self.primitives), xyzs.size))
        for i, prim in enumerate(self.primitives):
            B[i, :] = prim.derivative(xyzs)
        return B

    def Ginv(self, xyzs):
        B = self.B(xyzs)
        return np.linalg.pinv(B @ B.T)

    def Binv(self, xyzs):
        B = self.B(xyzs)
        return np.linalg.pinv(B @ B.T) @ B

    def to_internals(self, xyzs):
        q = np.zeros(len(self.primitives))
        for i, prim in enumerate(self.primitives):
            q[i] = prim.value(xyzs)
        return q

    def to_cartesians(
        self,
        dq,
        xyzs_ref,
        tol_q=1e-10,
        tol_x=1e-10,
        maxstep=0.05,
        maxiter=50,
        recursion_depth=0,
    ):
        xyzs = xyzs_ref.copy()
        dq_start = dq.copy()
        step = np.infty * np.ones_like(xyzs)
        for _ in range(maxiter):
            if np.linalg.norm(dq) < tol_q or np.linalg.norm(step) < tol_x:
                return xyzs
            step = (self.Binv(xyzs).T @ dq).reshape(xyzs.shape)
            steplengths = np.sqrt(np.sum(step**2, axis=-1))
            maxlength = np.max(steplengths)
            if maxlength > maxstep:
                step *= maxstep / maxlength
            # Calculate what this step corresponds to in internals
            step_q = self.diff_internals(xyzs + step, xyzs)
            xyzs = xyzs + step
            # Calculate the step for the next iteration
            dq -= step_q
        if recursion_depth >= 3:
            save_file = "not saved as requested."
            if self.save_failures:
                save_file = f"transformation_failure_{time.time():.0f}.pickle"
                with open(save_file, "wb") as fout:
                    pickle.dump([self, dq_start, xyzs_ref], fout)
            raise ConvergenceError(
                "Transformation to Cartesians not converged"
                f" within {maxiter} iterations, checkpoint"
                f" file {save_file}"
            )
        # Else warn, reduce dq_step by half, and try again
        warn("Reducing step in transformation to Cartesians")
        return self.to_cartesians(
            dq_start / 2,
            xyzs_ref,
            tol_q=tol_q,
            tol_x=tol_x,
            maxstep=maxstep,
            maxiter=maxiter,
            recursion_depth=recursion_depth + 1,
        )

    def diff_internals(self, xyzs1, xyzs2):
        dq = np.zeros(len(self.primitives))
        for i, prim in enumerate(self.primitives):
            dq[i] = prim.diff(xyzs1, xyzs2)
        return dq

    def force_to_internals(self, xyzs, force_cart):
        return self.Binv(xyzs) @ force_cart.flatten()

    def hessian_to_internals(self, xyzs, hess_cart, grad_cart=None):
        Binv = self.Binv(xyzs)

        if grad_cart is not None:
            raise NotImplementedError(
                "Transformation including gradient term " "is not implemented yet"
            )
            # hess_cart -= Binv @ grad_cart @ self.second_derivatives(xyzs)

        hess_int = Binv @ hess_cart @ Binv.T
        return hess_int


class DelocalizedCoordinates(InternalCoordinates):
    def __init__(self, primitives, xyzs, threshold=1e-6):
        InternalCoordinates.__init__(self, primitives)
        self.threshold = threshold
        self.delocalize(xyzs)

    def delocalize(self, xyzs):
        B = InternalCoordinates.B(self, xyzs)
        # G matrix without mass weighting
        G = B @ B.T
        w, v = np.linalg.eigh(G)
        # Set of nonredundant eigenvectors (eigenvalue =/= 0)
        self.U = v[:, np.abs(w) > self.threshold].copy()
        if check_colinear(xyzs):
            if self.size() != xyzs.size - 5:
                warn(
                    f"DelocalizedCoordinates found {self.size()} "
                    f"coordinates, expected 3N - 5 = {xyzs.size - 5} "
                    "(Linear molecule)."
                )
        elif self.size() != xyzs.size - 6:
            warn(
                f"DelocalizedCoordinates found {self.size()} coordinates,"
                f" expected 3N - 6 = {xyzs.size - 6}"
            )

    def size(self):
        return self.U.shape[1]

    def B(self, xyzs):
        return self.U.T @ InternalCoordinates.B(self, xyzs)

    def to_internals(self, xyzs):
        return self.U.T @ InternalCoordinates.to_internals(self, xyzs)

    def diff_internals(self, xyzs1, xyzs2):
        return self.U.T @ InternalCoordinates.diff_internals(self, xyzs1, xyzs2)


class ApproximateNormalCoordinates(CoordinateSystem):
    def __init__(self, atoms, H=None, threshold=0.0):
        self.threshold = threshold
        self.build(atoms, H=H)

    def build(self, atoms, H=None):
        if H is None:
            H = LindhHessian(h_trans=0.0, h_rot=0.0).build(atoms)
        vals, V = np.linalg.eigh(H)
        inds = np.abs(vals) >= self.threshold
        self.V = V[:, inds].copy()
        self.x0 = atoms.get_positions()
        if check_colinear(self.x0):
            if self.size() != self.x0.size - 5:
                warn(
                    f"ApproximateNormalCoordinates found {self.size()} "
                    f"coordinates, expected 3N - 5 = {self.x0.size - 5} "
                    "(Linear molecule)."
                )
        elif self.size() != self.x0.size - 6:
            warn(
                f"ApproximateNormalCoordinates found {self.size()} "
                f"coordinates, expected 3N - 6 = {self.x0.size - 6}"
            )

    def size(self):
        return self.V.shape[1]

    def to_internals(self, xyzs):
        return (xyzs - self.x0).flatten() @ self.V

    def to_cartesians(self, dq, xyzs_ref):
        return xyzs_ref + (self.V @ dq).reshape(xyzs_ref.shape)

    def diff_internals(self, xyzs1, xyzs2):
        return (xyzs1 - xyzs2).flatten() @ self.V

    def force_to_internals(self, xyzs, force_cart):
        return force_cart @ self.V

    def hessian_to_internals(self, xyzs, hess_cart, grad_cart=None):
        return self.V.T @ hess_cart @ self.V
