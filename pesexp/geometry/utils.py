import itertools
import numpy as np


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
