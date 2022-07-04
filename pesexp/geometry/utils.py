import itertools
import numpy as np


class CubicHermiteSpline:
    """Follows https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    First the 3N + 1 dimensional vectors p and m (3N spatial + energy) are
    calculated."""

    def __init__(self, images):
        self.n_images = len(images)
        self.n_atoms = len(images[0])
        # Corresponds to p in the reference
        self.points = np.zeros((self.n_images, 3 * self.n_atoms + 1))
        # Corresponds to m in the reference
        self.tangents = np.zeros((self.n_images, 3 * self.n_atoms + 1))

        for i, img in enumerate(images):
            self.points[i, :-1] = img.get_positions().flatten()
            self.points[i, -1] = img.get_potential_energies()
            if i == 0:
                self.tangents[i, :-1] = (
                    images[i + 1].get_positions().flatten()
                    - images[i].get_positions().flatten()
                )
            elif i == self.n_images - 1:
                self.tangents[i, :-1] = (
                    images[i].get_positions().flatten()
                    - images[i - 1].get_positions().flatten()
                )
            else:
                self.tangents[i, :-1] = (
                    images[i + 1].get_positions().flatten()
                    - images[i - 1].get_positions().flatten()
                )
            self.tangents[i, -1] = np.dot(
                img.get_forces().flatten(), self.tangents[i, :-1]
            ) / np.linalg.norm(self.tangents[i, :-1])


def gram_schmidt(V, norm=np.linalg.norm):
    """Implements Gram-Schmidt following
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Algorithm

    Parameters
    ----------
    V : list
        list of vectors to orthonormalize
    norm : function, optional
        norm used to normalize the vectors, by default np.linalg.norm
    """
    U = np.zeros_like(V)
    U[0] = V[0] / norm(V[0])
    for i in range(1, len(V)):
        U[i] = V[i]
        for j in range(0, i):
            U[i] -= np.dot(U[j], U[i]) * U[j]
        U[i] /= norm(U[i])
    return U


def check_colinear(xyzs, threshold=1e-3):
    """Takes an N x 3 array and checks if all points are colinear.
    Calculates the area of all possible triangles spanned by the
    vertices and returns True if all are smaller than the threshold."""
    for (a, b, c) in itertools.combinations(xyzs, 3):
        ab = b - a
        ab /= np.linalg.norm(ab)
        ac = c - a
        ac /= np.linalg.norm(ac)
        area = np.linalg.norm(np.cross(ab, ac)) / 2
        if area > threshold:
            return False
    return True
