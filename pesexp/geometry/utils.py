import itertools
import numpy as np


class CubicHermiteSpline:
    """Follows https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    First the 3N + 1 dimensional vectors p and m (3N spatial + energy) are
    calculated."""

    def __init__(self, points, tangents, coordinates=None):
        self.points = points
        self.tangents = tangents
        if coordinates is None:
            # Calculate from points
            raise NotImplementedError()
        self.coordinates = coordinates

    @classmethod
    def from_images(cls, images):
        n_images = len(images)
        n_atoms = len(images[0])
        # Corresponds to p in the reference
        points = np.zeros((n_images, 3 * n_atoms + 1))
        # Corresponds to m in the reference
        tangents = np.zeros((n_images, 3 * n_atoms + 1))

        path = np.zeros(n_images)
        for i, img in enumerate(images):
            points[i, :-1] = img.get_positions().flatten()
            points[i, -1] = img.get_potential_energy()
            if i != 0:
                # Use only Cartesian coordinates for path length
                path[i] = path[i - 1] + np.linalg.norm(
                    points[i, :-1] - points[i - 1, :-1]
                )
        # Normalized path coordinate s
        coordinates = path / path[-1]

        # Calculate and (properly) normalize tangents
        for i, img in enumerate(images):
            if i == 0:
                tangents[i, :-1] = points[i + 1, :-1] - points[i, :-1]
                tangents[i, :-1] /= coordinates[i + 1] - coordinates[i]
            elif i == n_images - 1:
                tangents[i, :-1] = points[i, :-1] - points[i - 1, :-1]
                tangents[i, :-1] /= coordinates[i] - coordinates[i - 1]
            else:
                tangents[i, :-1] = 0.5 * (
                    (points[i + 1, :-1] - points[i, :-1])
                    / (coordinates[i + 1] - coordinates[i])
                    + (points[i, :-1] - points[i - 1, :-1])
                    / (coordinates[i] - coordinates[i - 1])
                )
            # Gradient is -force
            tangents[i, -1] = np.dot(-img.get_forces().flatten(), tangents[i, :-1])

        return cls(points, tangents, coordinates)

    def __call__(self, s):
        result = np.zeros((len(s), self.points.shape[1]))
        inds = np.searchsorted(self.coordinates, s)
        # extrapolate (linearly) where inds == 0 or len(self.points) (s < 0) or (s >= 1)
        mask = np.logical_or(inds == 0, inds == len(self.points))
        # Index of the reference point from which to extrapolate:
        ref_inds = np.clip(inds[mask] - 1, a_min=0, a_max=None)
        s_extrap = s[mask]
        s_extrap[s_extrap > 1.0] -= 1
        result[mask] = (
            self.points[ref_inds] + s_extrap[:, np.newaxis] * self.tangents[ref_inds]
        )
        # interpolate for other inds
        mask = np.logical_not(mask)
        delta_s = np.reshape(
            self.coordinates[inds[mask]] - self.coordinates[inds[mask] - 1], (-1, 1)
        )
        t = (s[mask] - self.coordinates[inds[mask] - 1])[:, np.newaxis] / delta_s
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        result[mask] = (
            h00 * self.points[inds[mask] - 1]
            + h10 * delta_s * self.tangents[inds[mask] - 1]
            + h01 * self.points[inds[mask]]
            + h11 * delta_s * self.tangents[inds[mask]]
        )
        return result


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
