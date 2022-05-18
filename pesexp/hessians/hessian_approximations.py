from abc import abstractmethod
import numpy as np


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
            print('Skipping BFGS update to conserve positive '
                  'definiteness.')
            return
        self += self.deltaH(dx, dg)

    def deltaH(self, dx, dg):
        v = np.dot(self, dx)
        return (np.outer(dg, dg) / np.dot(dx, dg)
                - np.outer(v, v) / np.dot(dx, v))


class MurtaghSargentHessian(HessianApproximation):

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        return np.outer(s, s)/np.dot(s, dx)


class PsBHessian(HessianApproximation):
    "Powell-symmetric-Broyden"

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        dxdx = np.dot(dx, dx)
        return ((np.outer(s, dx) + np.outer(dx, s))/dxdx
                - np.dot(dx, s)/dxdx**2 * np.outer(dx, dx))


class BofillHessian(HessianApproximation):
    """Bofill, J. Comput. Chem., 15:1-11 (1994)"""

    def deltaH(self, dx, dg):
        s = dg - np.dot(self, dx)
        phi = np.dot(s, dx)**2 / (np.dot(s, s) * np.dot(dx, dx))
        return (phi * MurtaghSargentHessian.deltaH(self, dx, dg)
                + (1 - phi) * PsBHessian.deltaH(self, dx, dg))
