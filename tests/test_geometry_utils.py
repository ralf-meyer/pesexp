import numpy as np
from pesexp.geometry.utils import check_colinear


def test_check_colinear():
    X = np.array([[1., 0., 0.],
                  [2., 0., 0.],
                  [-4., 0., 0]])
    assert check_colinear(X)

    X = np.array([[0., 0., 0.],
                  [2., 0., 0.],
                  [0., 1., 0]])
    assert check_colinear(X) is False

    # Two points are always colinear:
    X = np.array([[1., 0., 0.],
                  [0., 2., 3.]])
    assert check_colinear(X)

    X = np.array([[1., 0., 0.],
                  [2., 0., 0.],
                  [-4., 0., 0],
                  [10., 0., 0],
                  [1.0000001, 0., 0.]])
    assert check_colinear(X)
