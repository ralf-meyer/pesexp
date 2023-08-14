import logging
from abc import abstractmethod
import numpy as np
from pesexp.linalg import dscaleh

logger = logging.getLogger(__name__)


def rescale_hessian_step(hessian_approximation):
    class RescaledHessianApproximation(hessian_approximation):
        def update(self, dx, dg):
            # transform
            B = np.diag(1 / dx)
            # Inverting a diagonal matrix is just inverting the elements:
            BTinv = np.diag(dx)

            H_trans = BTinv @ self @ BTinv.T
            dx_trans = B @ dx
            dg_trans = BTinv @ dg

            # calculate update in transformed space
            deltaH = hessian_approximation.deltaH(H_trans, dx_trans, dg_trans)

            # backtransform
            self += B.T @ deltaH @ B

    return RescaledHessianApproximation


def rescale_hessian_eigs(hessian_approximation, weighting=(1.0, 10.0)):
    class RescaledHessianApproximation(hessian_approximation):
        def update(self, dx, dg):
            # transform
            vals, vecs = np.linalg.eigh(self)
            # No rescaling along singular directions
            vals[np.abs(vals) < 1e-6] = 1.0
            weights = np.sqrt(
                abs(vals)
                / (
                    (abs(vals) - abs(vals).min())
                    / (abs(vals).max() - abs(vals).min())
                    * (weighting[1] - weighting[0])
                    + weighting[0]
                )
            )
            B = np.diag(weights) @ vecs.T
            BTinv = np.linalg.pinv(B @ B.T) @ B

            H_trans = BTinv @ self @ BTinv.T
            dx_trans = B @ dx
            dg_trans = BTinv @ dg

            # calculate update in transformed space
            deltaH = hessian_approximation.deltaH(H_trans, dx_trans, dg_trans)

            # backtransform
            self += B.T @ deltaH @ B

    return RescaledHessianApproximation


def dscale_hessian(hessian_approximation):
    class DscaledHessianApproximation(hessian_approximation):
        def update(self, dx, dg):
            # transform
            H_trans, d = dscaleh(self)
            Binv = np.diag(d)
            B = np.diag(1 / d)
            dx_trans = B @ dx
            dg_trans = Binv @ dg

            # calculate update in transformed space
            deltaH = hessian_approximation.deltaH(H_trans, dx_trans, dg_trans)

            # backtransform
            self += B.T @ deltaH @ B

    return DscaledHessianApproximation


class HessianApproximation(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def update(self, dx, dg):
        self += self.deltaH(dx, dg)

    @abstractmethod
    def deltaH(self, dx, dg):
        """Separated to allow for composite methods"""


class BFGSHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        v = np.dot(self, dx)
        return np.outer(dg, dg) / np.dot(dx, dg) - np.outer(v, v) / np.dot(dx, v)


class DFPHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        v = np.dot(self, dx)
        norm = np.dot(dg, dx)
        return (
            (1 + np.dot(dx, v) / norm) * np.outer(dg, dg) / norm
            - np.outer(dg, v) / norm
            - np.outer(v, dg) / norm
        )


class ForcedDeterminantBFGSHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        deltaBFGS = BFGSHessian.deltaH(self, dx, dg)
        det = np.linalg.det(self + deltaBFGS)
        logger.debug(f"ForcedDeterminantBFGS: new determinant will be {det}")
        if det > 0.0:
            return deltaBFGS
        logger.info("Skipping BFGS update to conserve positive definiteness.")
        return np.zeros_like(self)


class TSBFGSHessian(HessianApproximation):
    """This follows the definition in
    Bofill, International Journal of Quantum Chemistry, Vol 94, 324-332 (2003)
    """

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        vals, vecs = np.linalg.eigh(self)
        abs_B = np.einsum("ik,k,jk->ij", vecs, np.abs(vals), vecs)
        abs_B_dx = np.dot(abs_B, dx)
        u = np.dot(dx, dg) * dg + np.dot(dx, abs_B_dx) * abs_B_dx
        u /= np.dot(dx, dg) ** 2 + np.dot(dx, abs_B_dx) ** 2
        su = np.outer(s, u)
        return (su + su.T) - np.dot(s, dx) * np.outer(u, u)


class TSBFGSHessian2(HessianApproximation):
    """Alternative definition"""

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        vals, vecs = np.linalg.eigh(self)
        abs_B = np.einsum("ik,k,jk->ij", vecs, np.abs(vals), vecs)
        abs_B_dx = np.dot(abs_B, dx)
        u = np.dot(dx, dg) * dg + np.dot(dx, abs_B_dx) * abs_B_dx
        u /= np.dot(dx, dg) ** 2 + np.dot(dx, abs_B_dx) ** 2
        su = np.outer(s, u)
        return (su + su.T) - np.dot(s, dx) * np.outer(u, u)


class PartitionedBFGS(HessianApproximation):
    def deltaH(self, dx, dg):
        vals, vecs = np.linalg.eigh(self)
        dx_max = dx @ vecs[:, :1]
        dg_max = dg @ vecs[:, :1]
        B_max = vecs[:, :1].T @ self @ vecs[:, :1]
        logger.debug(
            f"d_max = {dx_max @ dg_max:.2E}, b_max = {dx_max @ B_max @ dx_max:.2E}"
        )
        v_max = np.dot(B_max, dx_max)
        delta_max = np.outer(dg_max, dg_max) / np.dot(dx_max, dg_max) - np.outer(
            v_max, v_max
        ) / np.dot(dx_max, v_max)
        delta_max = vecs[:, :1] @ delta_max @ vecs[:, :1].T

        dx_min = dx @ vecs[:, 1:]
        dg_min = dg @ vecs[:, 1:]
        v_min = np.dot(vecs[:, 1:].T @ self @ vecs[:, 1:], dx_min)
        delta_min = np.outer(dg_min, dg_min) / np.dot(dx_min, dg_min) - np.outer(
            v_min, v_min
        ) / np.dot(dx_min, v_min)
        delta_min = vecs[:, 1:] @ delta_min @ vecs[:, 1:].T
        return delta_max + delta_min


class VariationalBroydenFamily(HessianApproximation):
    @abstractmethod
    def theta_squared(self, dx, dg):
        """Should be implemented by subclasses"""

    def deltaH(self, dx, dg):
        theta2 = self.theta_squared(dx, dg)
        s = dg - np.dot(self, dx)
        dxs = np.dot(dx, s)
        v = np.dot(dx, self).dot(dx) * dg - np.dot(dg, dx) * np.dot(self, dx)
        return np.outer(s, s) / dxs - theta2 * np.outer(v, v) / dxs


class BroydenTSBFGSHessian(VariationalBroydenFamily):
    def theta_squared(self, dx, dg):
        d = dx @ dg
        b = dx @ self @ dx
        h = dg @ np.linalg.inv(self) @ dg

        target = d / b
        # theta_sq = np.log1p(np.exp(-1 / abs(d * b))) + np.maximum(1 / (d * b), 0)
        # if theta_sq < 0.0:
        # theta_sq = (target * (d - b) + d - h) / (d * (d**2 - h * b))
        # theta_sq = max(0.0, theta_sq)
        if np.linalg.det(self) < 0:
            if d * b > 0:
                logger.debug("BFGS")
                theta_sq = 1 / (d * b)
            else:
                logger.debug("Fixed-target")
                target = 1.1
                theta_sq = (target * (d - b) + d - h) / (d * (d**2 - h * b))
        else:
            logger.debug("Fixed-target")
            target = -0.9
            theta_sq = (target * (d - b) + d - h) / (d * (d**2 - h * b))

        f = (h - d + theta_sq * d * (d**2 - h * b)) / (d - b)
        logger.debug(
            f"Broyden: theta^2 = {theta_sq:.2E}, BFGS theta^2 = {1 / (d * b):.2E} "
            f"f = {f:.2E}, target = {target:.2E}, d = {d:.2E}, b = {b:.2E}, h = {h:.2E}"
        )
        return theta_sq


class RelativeErrorHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        vals, vecs = np.linalg.eigh(self)
        M = np.einsum("im,m,jm->ij", vecs, np.abs(vals), vecs)
        M_dx = np.dot(M, dx)
        u = M_dx / np.dot(dx, M_dx)
        su = np.outer(s, u)
        return (su + su.T) - np.dot(s, dx) * np.outer(u, u)


class MurtaghSargentHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        return np.outer(s, s) / np.dot(s, dx)


class PowellHessian(HessianApproximation):
    "Powell-symmetric-Broyden"

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        dxdx = np.dot(dx, dx)
        return (
            np.outer(s, dx) + np.outer(dx, s) - np.dot(dx, s) / dxdx * np.outer(dx, dx)
        ) / dxdx


class BofillHessian(HessianApproximation):
    """Bofill, J. Comput. Chem., 15:1-11 (1994)
    Note that the notation here is different to the reference.
    The variable s corresponds to xi, dg to gamma, dx to delta,
    in the reference,"""

    def phi(self, dx, dg):
        s = dg - np.dot(self, dx)
        return 1.0 - np.dot(s, dx) ** 2 / (np.dot(s, s) * np.dot(dx, dx))

    def deltaH(self, dx, dg):
        phi = self.phi(dx, dg)
        logger.debug(
            f"BofillHessian: phi = {phi:.2E}, d = {dx @ dg:.2E}, "
            f"b = {dx @ self @ dx:.2E}, det(B) = {np.linalg.det(self):.2E}"
        )
        if phi < 1.0:  # Check to avoid division by zero
            return (1 - phi) * MurtaghSargentHessian.deltaH(
                self, dx, dg
            ) + phi * PowellHessian.deltaH(self, dx, dg)
        return PowellHessian.deltaH(self, dx, dg)


class ModifiedBofillHessian(BofillHessian):
    """Bofill JM. Int. J. Quantum. Chem.,94:324-332 (2003)

    Note that the vectors y, d, and j are renamed to dg, dx, and s to stay
    consistent with the names used throughout this module."""

    def phi(self, dx, dg):
        # TODO: The calls to det and inv are highly wasteful!
        detH = np.linalg.det(self)
        Hinv = np.linalg.inv(self)
        s = dg - np.dot(self, dx)
        z = dx / np.dot(dx, dx) - s / np.dot(s, dx)
        # The function f(phi) is expressed as f(phi) = a - phi b
        a = 1 + np.dot(s, np.dot(Hinv, s)) / np.dot(s, dx)
        b = (
            np.dot(s, dx) * np.dot(z, np.dot(Hinv, z))
            + np.dot(s, np.dot(Hinv, s)) * np.dot(z, np.dot(Hinv, z))
            - np.dot(z, np.dot(Hinv, s)) ** 2
        )
        logger.debug(
            f"ModifiedBofillHessian detH {detH}, a-b = {a-b:.2E}, "
            f"a = {a:.2E}, b = {b:.2E}"
        )
        # TODO: This logic should be easier to express
        if detH > 0.0:
            # "If the determinant of the Bk matrix is positive definite"
            if np.sign(a) == np.sign(a - b):
                # "and the function f(phi) is either positive or negative
                # in all domains phi, we take phi = 0, which
                # corresponds to the Murtagh–Sargent formula."
                return 0.0
            else:
                # "if the function f(phi) is negative for phi = 0 and positive phi = 1
                # or vice versa, we select the phi value such that f(phi) is negative."
                if a < 0.0:
                    return 0.0
                else:
                    return 1.0
        else:
            # "On the other hand, if the determinant of the Bk matrix is negative
            # definite"
            if np.sign(a) == np.sign(a - b):
                # "and f(phi) is either positive or negative then we select phi = 1,
                # which corresponds to the Powell formula."
                return 1.0
            else:
                # "Finally, in the same situation, if the function f(phi) is positive
                # for phi = 0 and negative for phi = 1 or vice versa we select the
                # phi value such that f(phi) is positive"
                if a > 0.0:
                    return 0.0
                else:
                    return 1.0


class ConstantDeterminantBofillHessian(BofillHessian):
    """Alternate implementation of the idea in
    Bofill JM. Int. J. Quantum. Chem.,94:324-332 (2003)

    Instead of simply choosing between phi = 0.0 and phi = 1.0 we select
    a value 0 <= phi <= 1 that keeps the determinant as close to constant
    as possible, while enforcing a negative sign if possible.
    """

    def phi(self, dx, dg):
        detH = np.linalg.det(self)
        Hinv = np.linalg.inv(self)
        s = dg - np.dot(self, dx)
        z = dx / np.dot(dx, dx) - s / np.dot(s, dx)
        # The function f(phi) is expressed as f(phi) = a - phi b
        a = 1 + np.dot(s, np.dot(Hinv, s)) / np.dot(s, dx)
        b = (
            np.dot(s, dx) * np.dot(z, np.dot(Hinv, z))
            + np.dot(s, np.dot(Hinv, s)) * np.dot(z, np.dot(Hinv, z))
            - np.dot(z, np.dot(Hinv, s)) ** 2
        )
        if detH > 0.0:
            # Try to invert the sign while conserving the magnitude, i.e. f = -1
            target = -1.0
        else:
            target = 1.0
        phi = (a - target) / b
        # Restrict to valid range
        phi = max(0.0, min(phi, 1.0))
        logger.debug(
            f"ConstantDeterminantBofillHessian detH {detH}, a-b = {a-b:.2E}, "
            f"a = {a:.2E}, b = {b:.2E}, phi = {phi:.2f}, "
            f"new det {detH * (a - phi * b):.2E}"
        )
        return phi


class ForcedDeterminantBofillHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        delta_MS = MurtaghSargentHessian.deltaH(self, dx, dg)
        delta_Powell = PowellHessian.deltaH(self, dx, dg)
        # Determinants
        detH = np.linalg.det(self)
        detHMS = np.linalg.det(self + delta_MS)
        detHPowell = np.linalg.det(self + delta_Powell)
        logger.debug(
            f"ForcedDetBofill: det(H) = {detH:.2E} "
            f"det(H + MS) = {detHMS:.2E} det(H + Powell) = {detHPowell:.2E}"
        )
        # If both choices lead to the same sign
        if np.sign(detHMS) == np.sign(detHPowell):
            # and the structure is incorrect
            if detH > 0.0:
                # Use MS to change the eigenvalues as much as possible
                return delta_MS
            else:
                # If the structure is correct use Powell
                return delta_Powell
        # Otherwise use the one that gives the correct sign
        if detHMS < 0.0:
            return delta_MS
        else:
            return delta_Powell


class FarkasSchlegelHessian(BofillHessian):
    """Farkas and Schlegel, J. Chem. Phys., 111:10806-10814 (1999)"""

    def deltaH(self, dx, dg):
        # Note: the definition of phi used in this paper is different
        # from the original, therefore 1 - phi is used here
        sqrt_phi = np.sqrt(1.0 - self.phi(dx, dg))
        return sqrt_phi * MurtaghSargentHessian.deltaH(self, dx, dg) + (
            1 - sqrt_phi
        ) * BFGSHessian.deltaH(self, dx, dg)


class ForcedDeterminantFarkasSchlegelHessian(HessianApproximation):
    def deltaH(self, dx, dg):
        delta_MS = MurtaghSargentHessian.deltaH(self, dx, dg)
        delta_BFGS = BFGSHessian.deltaH(self, dx, dg)
        # Determinants
        detH = np.linalg.det(self)
        detHMS = np.linalg.det(self + delta_MS)
        detHBFGS = np.linalg.det(self + delta_BFGS)
        logger.debug(
            f"ForcedDetFarkasSchlegel: det(H) = {detH:.2E} "
            f"det(H + MS) = {detHMS:.2E} det(H + BFGS) = {detHBFGS:.2E}"
        )
        # If both choices lead to the same sign
        if np.sign(detHMS) == np.sign(detHBFGS):
            # and the structure is incorrect
            if detH < 0.0:
                # Use MS to change the eigenvalues as much as possible
                return delta_MS
            else:
                # If the structure is correct use BFGS
                return delta_BFGS
        # Otherwise use the one that gives the correct sign
        if detHMS > 0.0:
            return delta_MS
        else:
            return delta_BFGS


def thresholding(hessian_approx, dx_thresh=0.0, dg_thresh=0.0, dx_dg_thresh=0.0):
    """Decorator used to skip Hessian updates if the root mean squared change in
    gradient or the displacement are smaller than the given thesholds.
    """

    class ThresholdedHessianApproximation(hessian_approx):
        def update(self, dx, dg):
            if (
                np.sqrt(np.mean(dx**2)) >= self.dx_thresh
                and np.sqrt(np.mean(dg**2)) >= self.dg_thresh
                and np.dot(dx, dg) >= self.dx_dg_thresh
            ):
                self += self.deltaH(dx, dg)
            else:
                logger.debug(
                    f"Skipping Hessian update for dx: {np.sqrt(np.mean(dx**2)):.2E}, "
                    f"dg: {np.sqrt(np.mean(dg**2)):.2E} (RMS)"
                )

    ThresholdedHessianApproximation.dx_thresh = dx_thresh
    ThresholdedHessianApproximation.dg_thresh = dg_thresh
    ThresholdedHessianApproximation.dx_dg_thresh = dx_dg_thresh
    return ThresholdedHessianApproximation
