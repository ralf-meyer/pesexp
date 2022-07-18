import logging
import numpy as np
import ase.atoms
from ase.build.rotate import rotation_matrix_from_points
from pesexp.hessians.hessian_guesses import get_hessian_guess
from pesexp.geometry.utils import CubicHermiteSpline, gram_schmidt
from typing import List

logger = logging.getLogger(__name__)


def pRFO_guess_from_neb(
    images: List[ase.atoms.Atoms],
    guess_hessian: str = "fischer_almloef",
    remove_translation_and_rotation=True,
):
    spline = CubicHermiteSpline.from_images(images)
    s = np.linspace(0, 1, 101)
    interp = spline(s)

    # Find the highest image along the interpolated path
    energies = interp[:, -1]
    imax = np.argmax(energies)

    atoms = images[0].copy()
    pos_0 = interp[imax, :-1].reshape(-1, 3)
    atoms.set_positions(pos_0)

    H = get_hessian_guess(atoms, guess_hessian)
    _, vecs = np.linalg.eigh(H)

    # Find the positions of the neighbors to the highest image:
    pos_1 = interp[imax - 1, :-1].reshape(-1, 3)
    pos_2 = interp[imax + 1, :-1].reshape(-1, 3)

    if remove_translation_and_rotation:
        center_0 = np.mean(pos_0, axis=0)
        center_1 = np.mean(pos_1, axis=0)
        center_2 = np.mean(pos_2, axis=0)

        # Align the first neighbor with the highest image:
        R = rotation_matrix_from_points((pos_1 - center_1).T, (pos_0 - center_0).T)
        pos_1 = np.dot(pos_1 - center_1, R.T) + center_0
        # Align the second neighbor with the highest image:
        R = rotation_matrix_from_points((pos_2 - center_2).T, (pos_0 - center_0).T)
        pos_2 = np.dot(pos_2 - center_2, R.T) + center_0

        # TODO: I would prefer to directly remove rotation and translation from the
        # displacement vectors (and tau).

    # Calculate curvature from finite differences (with uneven grid spacing)
    h1 = np.linalg.norm((pos_1 - pos_0).flatten())
    h2 = np.linalg.norm((pos_2 - pos_0).flatten())
    e0 = interp[imax, -1]
    e1 = interp[imax - 1, -1]
    e2 = interp[imax + 1, -1]
    curv = 2 * (h1 * e2 + h2 * e1 - (h1 + h2) * e0) / (h1 * h2 * (h1 + h2))
    logger.info(f"Curvature: {curv:.2f} eV/Ang^2")
    assert curv < 0.0

    tau = pos_2 - pos_1
    # Normalize
    tau = tau.flatten() / np.linalg.norm(tau)
    # Find eigenvector of H with the highest overlap with tau
    ind = int(np.argmax(np.dot(tau, vecs)))
    # Remove that eigenvector and add tau instead
    V = np.concatenate(
        [tau[np.newaxis, :], vecs[:, :ind].T, vecs[:, ind + 1 :].T], axis=0
    )
    # Orthogonalize the eigenvectors with respect to the first vector, i.e. tau
    U = gram_schmidt(V)
    # Rebuild the Hessian matrix by transforming it to the new basis U and
    # adding the approximated imaginary mode
    H = U[1:].T @ U[1:] @ H @ U[1:].T @ U[1:] + curv * np.outer(tau, tau)

    return atoms, H
