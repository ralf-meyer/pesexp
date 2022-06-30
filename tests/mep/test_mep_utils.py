import ase.io
import numpy as np
from pesexp.mep.utils import pRFO_guess_from_neb


def test_pRFO_workflow(resource_path_root):
    images = ase.io.read(resource_path_root / 'mep' / 'Fe_(NH3)_5+N2O.traj',
                         index=':')
    atoms, H = pRFO_guess_from_neb(images)
    vals, vecs = np.linalg.eigh(H)
    # Assert that there is only one negative eigenvalue (neglecting values
    # close to zero)
    assert np.count_nonzero(vals < -1e-8) == 1
    # Assert that the smallest eigenvalue corresponds to the bond breaking
    # direction, i.e. it is approximately parallel to the tangent at image 4.
    assert vals[0] < 15
    tangent = (images[3].get_positions() - images[5].get_positions()).flatten()
    tangent /= np.linalg.norm(tangent)
    assert np.abs(np.dot(vecs[:, 0], tangent)) > 0.95
