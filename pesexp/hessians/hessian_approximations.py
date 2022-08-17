import logging
from abc import abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


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
    def update(self, dx, dg):
        if np.dot(dx, dg) < 0:
            logger.info("Skipping BFGS update to conserve positive definiteness.")
            return
        self += self.deltaH(dx, dg)

    def deltaH(self, dx, dg):
        v = np.dot(self, dx)
        return np.outer(dg, dg) / np.dot(dx, dg) - np.outer(v, v) / np.dot(dx, v)


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
    The variable s corresponds to xi, dg to gamma, dx to delta, and
    phi to (1 - phi) in the reference,"""

    def phi(self, dx, dg):
        s = dg - np.dot(self, dx)
        return np.dot(s, dx) ** 2 / (np.dot(s, s) * np.dot(dx, dx))

    def deltaH(self, dx, dg):
        phi = self.phi(dx, dg)
        logger.debug(f"BofillHessian: phi = {phi:.2E}")
        if phi > 0.0:  # Check to avoid division by zero
            return phi * MurtaghSargentHessian.deltaH(self, dx, dg) + (
                1 - phi
            ) * PowellHessian.deltaH(self, dx, dg)
        return PowellHessian.deltaH(self, dx, dg)


class ModifiedBofillHessian(BofillHessian):
    """Bofill JM. Int. J. Quantum. Chem.,94:324-332 (2003)

    Note that the vectors y, d, and j are renamed to dg, dx, and s to stay
    consistent with the names used throughout this module."""

    def phi(self, dx, dg):
        detH = np.linalg.det(self)
        s = dg - np.dot(self, dx)
        z = dx / np.dot(dx, dx) - s / np.dot(s, dx)
        # f(phi) = a - phi b
        a = 1 + np.dot(s, np.dot(self, s)) / np.dot(s, dx)
        b = (
            np.dot(s, dx) * np.dot(z, np.dot(self, z))
            + np.dot(s, np.dot(self, s)) * np.dot(z, np.dot(self, z))
            - np.dot(z, np.dot(self, s)) ** 2
        )
        logger.debug(
            f"ModifiedBofillHessian detH {detH}, a-b = {a-b:.2E}, "
            f"a = {a:.2E}, b = {b:.2E}"
        )
        # TODO: This logic should be easier to express
        if detH > 0.0:
            # "If the determinant of the Bk matrix is positive definite"
            if (a > 0.0 and a - b > 0.0) or (a < 0.0 and a - b < 0.0):
                # "and the function f(phi) is either positive or negative
                # in all domains 0 <= phi <= 1, we take phi = 0"
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
            if (a > 0.0 and a - b > 0.0) or (a < 0.0 and a - b < 0.0):
                # "and f(phi) is either positive or negative then we select phi = 1"
                return 1.0
            else:
                # "Finally, in the same situation, if the function f(phi) is positive
                # for phi = 0 and negative for phi = 1 or vice versa we select the
                # phi value such that f(phi) is positive"
                if a > 0.0:
                    return 0.0
                else:
                    return 1.0


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
            f"det(H + MS) = {detHMS:.2E} det(H + Pow) = {detHPowell:.2E}"
        )
        if detH > 0.0:
            # If we can't change the determinant use MS
            if detHMS > 0.0 and detHPowell > 0.0:
                return delta_MS
            else:
                if detHMS < 0.0:
                    return delta_MS
                else:
                    return delta_Powell
        else:
            # We have the correct structure so we want to keep it
            if detHMS < 0.0 and detHPowell < 0.0:
                return delta_Powell
            else:
                if detHMS < 0.0:
                    return delta_MS
                else:
                    return delta_Powell


class FarkasSchlegelHessian(BofillHessian):
    """Farkas and Schlegel, J. Chem. Phys., 111:10806-10814 (1999)"""

    def deltaH(self, dx, dg):
        sqrt_phi = np.sqrt(self.phi(dx, dg))
        return sqrt_phi * MurtaghSargentHessian.deltaH(self, dx, dg) + (
            1 - sqrt_phi
        ) * BFGSHessian.deltaH(self, dx, dg)


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
