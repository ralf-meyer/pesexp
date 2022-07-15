import logging
import numpy as np
import ase.atoms
from pesexp.hessians.hessian_guesses import get_hessian_guess
from pesexp.geometry.utils import CubicHermiteSpline, gram_schmidt
from typing import List

logger = logging.getLogger(__name__)


def pRFO_guess_from_neb(
    images: List[ase.atoms.Atoms], guess_hessian: str = "fischer_almloef"
):
    spline = CubicHermiteSpline.from_images(images)
    s = np.linspace(0, 1, 101)
    interp = spline(s)
    energies = interp[:, -1]
    imax = np.argmax(energies)

    atoms = images[0].copy()
    atoms.set_positions(interp[imax, :-1].reshape(-1, 3))

    H = get_hessian_guess(atoms, guess_hessian)
    vals, vecs = np.linalg.eigh(H)

    # Calculate curvature
    h1 = np.linalg.norm(interp[imax - 1, :-1] - interp[imax, :-1])
    h2 = np.linalg.norm(interp[imax + 1, :-1] - interp[imax, :-1])
    e0 = interp[imax, -1]
    e1 = interp[imax - 1, -1]
    e2 = interp[imax + 1, -1]
    curv = 2 * (h1 * e2 + h2 * e1 - (h1 + h2) * e0) / (h1 * h2 * (h1 + h2))
    logger.info(f"Curvature: {curv:.2f} eV/Ang^2")
    assert curv < 0.0

    tau = interp[imax + 1, :-1] - interp[imax - 1, :-1]
    # Normalize
    tau = tau / np.linalg.norm(tau)
    # Find highest overlap
    ind = int(np.argmax(np.dot(tau, vecs)))

    V = np.concatenate(
        [tau[np.newaxis, :], vecs[:, :ind].T, vecs[:, ind + 1 :].T], axis=0
    )
    U = gram_schmidt(V)

    H = U[1:].T @ U[1:] @ H @ U[1:].T @ U[1:] + curv * np.outer(tau, tau)

    return atoms, H
