import numpy as np
from ase.atoms import Atoms
from pesexp.optimizers.optimizers import scaledPRFO


def test_scaled_pRFO_step():
    # Based on the third step in a Baker system 01 optimization
    H = np.array(
        [
            [-10.75201976, -6.42166332, 5.57618174],
            [-6.42166332, 8.23167565, 14.23864386],
            [5.57618174, 14.23864386, 283.73471308],
        ]
    )
    f = np.array([-1.04432943, 1.98562983, 1.97730308])

    opt = scaledPRFO(Atoms(["H"], [(0.0, 0.0, 0)]), H0=H, s=1.0)

    print(opt.internal_step(f))

    omega, V = np.linalg.eigh(H)
    # assert False
