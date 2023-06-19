import numpy as np
from pesexp.hessians.hessian_tools import filter_hessian


def test_filter_hessian():
    H = np.diag([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    H = filter_hessian(H, thresh=1.1e-5)
    np.testing.assert_allclose(H, np.diag([1.1e-5, 1.1e-5, 1.0, 2.0, 3.0, 4.0]))

    # Build random matrix with eigenvalue above 0.1
    A = np.array(
        [
            [0.7432, 0.4965, 0.2700, 0.0742, -0.0800, -0.1814],
            [0.4965, 1.0133, 0.5708, 0.1900, -0.1071, -0.2977],
            [0.2700, 0.5708, 0.9332, 0.3893, -0.0277, -0.2830],
            [0.0742, 0.1900, 0.3893, 0.7155, 0.2134, -0.0696],
            [-0.0800, -0.1071, -0.0277, 0.2134, 0.6736, 0.4129],
            [-0.1814, -0.2977, -0.2830, -0.0696, 0.4129, 1.2388],
        ]
    )
    A_ref = A.copy()
    A = filter_hessian(A)
    # Test that it remained unaltered
    np.testing.assert_allclose(A, A_ref)
    # Increase the threshold to 1.0
    A = filter_hessian(A, thresh=1.0)
    vals, _ = np.linalg.eigh(A)
    # Test that the smaller eigenvalues have been filtered correctly
    np.testing.assert_allclose(vals, [1.0, 1.0, 1.0, 1.0, 1.27649882, 2.20586986])