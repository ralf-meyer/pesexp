import numpy as np
import logging

logger = logging.getLogger(__name__)


def filter_hessian(H, thresh=1.0e-4):
    """GeomeTRIC resets calculations if Hessian eigenvalues below
    a threshold of 1e-5 are encountered. This method is used to
    construct a new Hessian matrix where all eigenvalues smaller
    than the threshold are set exactly to the threshold value
    which by default is an order of magnitude above geomeTRICs cutoff.

    Parameters
    ----------
    H : np.array
        input Hessian
    thresh : float
        filter threshold. Default 1.e-4

    Returns
    -------
    H : np.array
        filtered Hessian
    """
    vals, vecs = np.linalg.eigh(H)
    logger.debug(f"Hessian eigenvalues:\n{vals}")
    logger.info(f"Filtering {np.sum(vals < thresh)} Hessian eigenvalues")
    vals[vals < thresh] = thresh
    H = np.einsum("ji,i,ki->jk", vecs, vals, vecs)
    return H


def project_hessian(H, atoms, remove_translation=True, remove_rotation=True):
    """This roughly follows the ideas/implementation in
    https://github.com/grimme-lab/xtb/blob/main/src/freq/project.f90"""

    if not remove_translation and not remove_rotation:
        return H

    n_atoms = len(atoms)
    xyzs = atoms.get_positions()
    masses = atoms.get_masses()
    mass_vector = np.repeat(np.sqrt(masses), 3)
    mass_matrix = np.outer(mass_vector, mass_vector)

    # Initialized as zeros since that corresponds to the B matrix of a constant
    # internal coordinate (in case a coordinate is not evaluated)
    B_ext = np.zeros((6, 3 * n_atoms))

    # First 3 coordinates are the center of mass coordinates. Their derivative
    # with respect to (mass-weighted) Cartesian coordinates is constant.
    if remove_translation:
        for i in range(3):
            B_ext[i, i::3] = 1.0

    if remove_rotation:
        center = atoms.get_center_of_mass()
        moments, axes = atoms.get_moments_of_inertia(vectors=True)

        for i in range(3):
            if moments[i] < 1e-8:
                continue
            # TODO: this loop could probably be contracted since np.cross
            # can work on arrays
            for i_at in range(n_atoms):
                vec = xyzs[i_at, :] - center
                vec = np.cross(axes[i, :], vec)
                B_ext[3 + i, 3 * i_at : 3 * i_at + 3] = vec

    # Mass scaling
    B_ext *= mass_vector[np.newaxis, :]

    # The xtb implementation uses an interesting trick to construct
    # the Wilson B matrix for the internal coordinates B_int.
    # A standard approach would be to build B_int as orthogonal
    # vectors to B_ext e.g. using the Gram-Schmidt algorithm
    # and constructing a projection matrix P = (B_int)^+ B_int,
    # where (B_int)^+ is the pseudo-inverse of B_int. For orthonormalized
    # rows in the B matrix calculating the pseudo-inverse corresponds
    # to simply transposing the matrix.

    # Since the rows in B_ext are orthogonal by construction they only
    # need to be normalized.
    norm = np.linalg.norm(B_ext, axis=1)
    B_ext[norm > 1e-8, :] /= norm[norm > 1e-8, np.newaxis]

    # For orthonomal B matrices the projection on both internal and
    # external degrees of freedom should equal the unit operation
    # B_int.T @ B_int + B_ext.T @ B_ext = 1. The projection operator
    # on the internal degrees of freedom can therefore be calculated
    # without explicitly building a set of internal coordinates:
    P = np.eye(3 * n_atoms) - B_ext.T @ B_ext

    # TODO: simplify this because I think that transforming from to a
    # mass weighted Hessian and back immediately after is confusing
    return (P @ (H / mass_matrix) @ P.T) * mass_matrix
