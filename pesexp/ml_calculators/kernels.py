import numpy as np
from abc import abstractmethod


class Kernel:
    """Base class for all kernels"""

    @abstractmethod
    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False):
        """Exaluates the kernel"""

    @abstractmethod
    def diag(self, X):
        """Evaluates only the diagonal of the kernel which often can be
        achieved faster and is needed for the uncertainty prediction"""


class RBFKernel(Kernel):
    def __init__(
        self,
        constant=0.0,
        factor=1.0,
        length_scale=1.0,
        factor_bounds=(1e-3, 1e3),
        length_scale_bounds=(1e-3, 1e3),
    ):
        self.factor = factor
        self.constant = constant
        self.length_scale = length_scale
        self.factor_bounds = factor_bounds
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def theta(self):
        return np.log(np.hstack([self.factor, self.length_scale]))

    @theta.setter
    def theta(self, theta):
        self.factor = np.exp(theta[0])
        self.length_scale = np.exp(theta[1:])

    @property
    def bounds(self):
        if np.ndim(self.length_scale_bounds) == 1:
            return np.log(np.asarray([self.factor_bounds, self.length_scale_bounds]))
        elif np.ndim(self.length_scale_bounds) == 2:
            return np.log(np.asarray([self.factor_bounds] + self.length_scale_bounds))

    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n * (1 + n_dim), m * (1 + n_dim)))
        if eval_gradient:
            # Array to hold the derivatives with respect to the factor and
            # the length_scale
            if self.anisotropic:
                K_gradient = np.zeros(
                    (n * (1 + n_dim), m * (1 + n_dim), 1 + self.length_scale.shape[0])
                )
            else:  # isotropic
                K_gradient = np.zeros((n * (1 + n_dim), m * (1 + n_dim), 2))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n + a * n_dim, n + (a + 1) * n_dim, 1)
                db = slice(m + b * n_dim, m + (b + 1) * n_dim, 1)
                # A few helpful quantities:
                scaled_diff = (X[a, :] - Y[b, :]) / self.length_scale
                inner_prod = scaled_diff.dot(scaled_diff)
                outer_prod_over_l = np.outer(
                    scaled_diff / self.length_scale, scaled_diff / self.length_scale
                )
                exp_term = np.exp(-0.5 * inner_prod)
                # populate kernel matrix:
                K[a, b] = exp_term
                K[da, b] = -exp_term * scaled_diff / self.length_scale
                K[a, db] = exp_term * scaled_diff / self.length_scale
                K[da, db] = exp_term * (
                    np.eye(n_dim) / self.length_scale**2 - outer_prod_over_l
                )

                # Gradient with respect to the length_scale
                if eval_gradient:
                    if self.anisotropic:
                        # Following the accompaning latex documents the
                        # three matrix dimensions are refered to as q, p and s.
                        K_gradient[a, b, 1:] = exp_term * (
                            scaled_diff**2 / self.length_scale
                        )
                        K_gradient[da, b, 1:] = exp_term * (
                            2 * np.diag(scaled_diff / self.length_scale**2)
                            - np.outer(
                                scaled_diff / self.length_scale,
                                scaled_diff**2 / self.length_scale,
                            )
                        )
                        K_gradient[a, db, 1:] = -exp_term * (
                            2 * np.diag(scaled_diff / self.length_scale**2)
                            - np.outer(
                                scaled_diff / self.length_scale,
                                scaled_diff**2 / self.length_scale,
                            )
                        )
                        delta_qp_over_lq2 = np.repeat(
                            (np.eye(n_dim) / self.length_scale**2)[:, :, np.newaxis],
                            n_dim,
                            axis=2,
                        )
                        delta_qs = np.repeat(
                            np.eye(n_dim)[:, np.newaxis, :], n_dim, axis=1
                        )
                        delta_ps = np.repeat(
                            np.eye(n_dim)[np.newaxis, :, :], n_dim, axis=0
                        )
                        scaled_diff_s_squared = np.tile(
                            scaled_diff**2, (n_dim, n_dim, 1)
                        )
                        K_gradient[da, db, 1:] = (
                            exp_term
                            * (
                                delta_qp_over_lq2
                                * (scaled_diff_s_squared - 2 * delta_qs)
                                + np.repeat(
                                    np.outer(
                                        scaled_diff / self.length_scale,
                                        scaled_diff / self.length_scale,
                                    )[:, :, np.newaxis],
                                    n_dim,
                                    axis=2,
                                )
                                * (2 * delta_qs + 2 * delta_ps - scaled_diff_s_squared)
                            )
                            / self.length_scale
                        )

                    else:  # isotropic
                        outer_prod = np.outer(scaled_diff, scaled_diff)
                        K_gradient[a, b, 1] = exp_term * (
                            inner_prod / self.length_scale
                        )
                        K_gradient[da, b, 1] = exp_term * (
                            scaled_diff * (2 - inner_prod) / self.length_scale**2
                        )
                        K_gradient[a, db, 1] = -exp_term * (
                            scaled_diff * (2 - inner_prod) / self.length_scale**2
                        )
                        K_gradient[da, db, 1] = (
                            exp_term
                            * (
                                np.eye(n_dim) * (inner_prod - 2)
                                + outer_prod * (4 - inner_prod)
                            )
                            / self.length_scale**3
                        )

        if eval_gradient:
            # Gradient with respect to the factor
            K_gradient[:, :, 0] = K
            # Multiply gradient with respect to the length_scale by factor
            K_gradient[:, :, 1:] *= self.factor

        # Multiply by factor
        K *= self.factor
        # Add constant term only on non-derivative block
        K[:n, :m] += self.constant

        if eval_gradient:
            return K, K_gradient
        return K

    def diag(self, X):
        diag = np.zeros(X.shape[0] * (1 + X.shape[1]))
        diag[: X.shape[0]] = self.factor + self.constant
        diag[X.shape[0] :] = (
            self.factor * (np.ones_like(X) / self.length_scale**2).flatten()
        )
        return diag
