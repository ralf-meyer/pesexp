import itertools
import numpy as np


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
        area = np.linalg.norm(np.cross(ab, ac))/2
        if area > threshold:
            return False
    return True
