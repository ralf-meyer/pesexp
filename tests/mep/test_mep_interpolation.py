import numpy as np
from ase.io import read
from pesexp.geometry.utils import CubicHermiteSpline


def test_cubic_hermite_spline(resource_path_root):
    images = read(resource_path_root / "mep/MB_7_images.traj", index=":")
    spline = CubicHermiteSpline.from_images(images)
    trajectory = spline(np.linspace(-0.1, 1.1, 101))
    ts_ind = np.argmax(trajectory[:, -1])

    np.testing.assert_allclose(
        trajectory[ts_ind], [-0.8220, 0.6243, 0.0, -41.2], atol=0.05
    )
