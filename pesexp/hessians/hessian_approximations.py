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
        logger.debug(np.linalg.eigh(self)[0][:4])

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
        logger.debug(f"dx: {np.linalg.norm(dx):.2E}, dg: {np.linalg.norm(dg):.2E}")
        s = dg - np.dot(self, dx)
        dxdx = np.dot(dx, dx)
        u = dx / np.sqrt(dxdx)
        logger.debug(
            f"np.dot(dx, dx) = {dxdx:.2E}, np.dot(dx, s) = {np.dot(dx, s):.2E}, "
            f"np.dot(dx, s)/np.dot(dx, dx) = {np.dot(dx, s) / dxdx:2E}, "
            f"np.dot(dx, dg) = {np.dot(dx, dg):.2E}, "
            f"dx.H.dx = {dx.dot(np.dot(self, dx)):.2E}"
        )
        logger.debug(np.linalg.eigh((np.outer(s, dx) + np.outer(dx, s)) / dxdx)[0][:4])
        logger.debug(np.linalg.eigh(-np.dot(dx, s) / dxdx * np.outer(u, u))[0][-4:])
        logger.debug(
            np.linalg.eigh(
                (
                    np.outer(s, dx)
                    + np.outer(dx, s)
                    - np.dot(dx, s) / dxdx * np.outer(dx, dx)
                )
                / dxdx
            )[0][:4]
        )
        return (
            np.outer(s, dx) + np.outer(dx, s) - np.dot(dx, s) / dxdx * np.outer(dx, dx)
        ) / dxdx


class BofillHessian(HessianApproximation):
    """Bofill, J. Comput. Chem., 15:1-11 (1994)
    Note that the notation here is different to the reference.
    The variable s corresponds to xi, dg to gamma, dx to delta, and
    phi to (1 - phi) in the reference,"""

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        phi = np.dot(s, dx) ** 2
        if phi > 0.0:  # Check to avoid division by zero
            phi = phi / (np.dot(s, s) * np.dot(dx, dx))
            return phi * MurtaghSargentHessian.deltaH(self, dx, dg) + (
                1 - phi
            ) * PowellHessian.deltaH(self, dx, dg)
        return PowellHessian.deltaH(self, dx, dg)


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
