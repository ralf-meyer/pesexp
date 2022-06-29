import numpy as np
import ase.atoms
from ase.build import minimize_rotation_and_translation
from pesexp.hessians.hessian_guesses import compute_hessian_guess
from pesexp.geometry.utils import gram_schmidt
from typing import List


def pRFO_guess_from_neb(images: List[ase.atoms.Atoms], guess_hessian: str = 'fischer_almloef'):
    imax = np.argmax([im.get_potential_energy() for im in images])
    atoms = images[imax]
    neighbor1 = images[imax - 1]
    neighbor2 = images[imax + 1]

    H = compute_hessian_guess(atoms, guess_hessian)
    vals, vecs = np.linalg.eigh(H)
    # Approximiate curvature along NEB using finite differences
    h1 = np.linalg.norm(neighbor1.get_positions() - atoms.get_positions())
    h2 = np.linalg.norm(neighbor2.get_positions() - atoms.get_positions())
    e1 = neighbor1.get_potential_energy()
    e2 = neighbor2.get_potential_energy()
    curv = (2 * (h1 * e2 + h2 * e1 - (h1 + h2) * atoms.get_potential_energy())
            / (h1 * h2 * (h1 + h2)))
    assert(curv < 0.)
    # Align the neighbors before calculating the tangent
    minimize_rotation_and_translation(atoms, neighbor1)
    minimize_rotation_and_translation(atoms, neighbor2)
    tau = neighbor1.get_positions() - neighbor2.get_positions()
    # Normalize
    tau = tau.flatten()/np.linalg.norm(tau)
    # Find highest overlap
    ind = np.argmax(np.dot(tau, vecs))

    V = np.concatenate([tau[np.newaxis, :],
                        vecs[:, :ind].T,
                        vecs[:, ind+1:].T], axis=0)
    U = gram_schmidt(V)

    H = U[1:].T @ U[1:] @ H @ U[1:].T @ U[1:] + curv*np.outer(tau, tau)

    return atoms, H
