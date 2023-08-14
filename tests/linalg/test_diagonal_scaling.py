import pytest
import numpy as np
from pesexp.linalg import dscale, dscaleh


@pytest.mark.parametrize(
    ("A", "S_ref", "dl_ref", "dr_ref"),
    [
        (
            np.array([[1.0, 3.0, -2.0], [2.0, 1.0, -1.0], [-2.0, -3.0, 1.0]]),
            np.array(
                [
                    [0.5576, 1.2765, -1.4050],
                    [1.6083, 0.6137, -1.0132],
                    [-1.1151, -1.2765, 0.7025],
                ]
            ),
            np.array([0.8851, 1.2765, 0.8851]),
            np.array([0.6300, 0.4807, 0.7937]),
        ),
        (
            np.array(
                [[1.0, 3.0, -2.0, 1.5], [2.0, 1.0, -1.0, -4.5], [-2.0, -3.0, 1.0, 0.5]]
            ),
            np.array(
                [
                    [0.5748, 1.3161, -1.4485, 0.9125],
                    [1.1497, 0.4387, -0.7243, -2.7375],
                    [-1.5131, -1.7321, 0.9532, 0.4003],
                ]
            ),
            np.array([0.9125, 0.9125, 1.2009]),
            np.array([0.6300, 0.4807, 0.7937, 0.6667]),
        ),
    ],
)
def test_dscale(A, S_ref, dl_ref, dr_ref, atol=1e-4):
    S, dl, dr = dscale(A)

    np.testing.assert_allclose(S, S_ref, atol=atol)
    np.testing.assert_allclose(dl, dl_ref, atol=atol)
    np.testing.assert_allclose(dr, dr_ref, atol=atol)


@pytest.mark.parametrize(
    "A", [np.array([[1.2, -0.2, 0.8], [-0.2, 0.6, 0.3], [0.8, 0.3, -0.8]])]
)
def test_dscaleh(A, atol=1e-8):
    S, d = dscaleh(A)
    S_ref, dl, dr = dscale(A)
    d_ref = np.sqrt(dl * dr)

    np.testing.assert_allclose(S, S_ref, atol=atol)
    np.testing.assert_allclose(d, d_ref, atol=atol)
