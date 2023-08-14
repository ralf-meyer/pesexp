import numpy as np


def dscale(A, tol=1e-15):
    """Literal translation of the MATLAB code in Appendix C. of
    J. Uhlmann, SIAM Journal on Matrix Analysis and Applications,
    Vol. 39, Issue 2, 781-800 (2018):
    https://doi.org/10.1137/17M113890X
    """
    m, n = A.shape
    L = np.zeros_like(A)
    M = np.ones_like(A)
    S = np.sign(A)
    A = np.abs(A)
    mask = A > 0.0
    L[mask] = np.log(A[mask])
    # Why would this not be the case???
    L[~mask] = 0.0
    M[~mask] = 0.0
    r = np.sum(M, axis=1)
    c = np.sum(M, axis=0)
    u = np.zeros(m)
    v = np.zeros(n)
    dx = 2 * tol

    while dx > tol:
        idx = c > 0.0
        p = np.sum(L[:, idx], axis=0) / c[idx]
        L[:, idx] -= np.tile(p, (m, 1)) * M[:, idx]
        v[idx] -= p
        dx = np.mean(np.abs(p))
        idx = r > 0.0
        p = np.sum(L[idx, :], 1) / r[idx]
        L[idx, :] -= np.tile(p, (n, 1)).T * M[idx, :]
        u[idx] -= p
        dx += np.mean(np.abs(p))

    dl = np.exp(u)
    dr = np.exp(v)
    S = S * np.exp(L)
    return S, dl, dr


def dscaleh(A, tol=1e-15):
    """Adapts the dscale algorithm by Uhlmann for hermitian/symmetric matrices."""
    # TODO: Obviously this needs to be optimized
    m, n = A.shape
    if m != n:
        raise ValueError("Array must be square.")
    L = np.zeros_like(A)
    M = np.ones_like(A)
    S = np.sign(A)
    A = np.abs(A)
    mask = A > 0.0
    L[mask] = np.log(A[mask])
    L[~mask] = 0.0
    M[~mask] = 0.0
    r = np.sum(M, axis=1)
    c = np.sum(M, axis=0)
    u = np.zeros(m)
    dx = 2 * tol

    while dx > tol:
        idx = c > 0.0
        p = np.sum(L[:, idx], axis=0) / c[idx]
        L[:, idx] -= np.tile(p, (m, 1)) * M[:, idx]
        u[idx] -= p
        dx = np.mean(np.abs(p))
        idx = r > 0.0
        p = np.sum(L[idx, :], 1) / r[idx]
        L[idx, :] -= np.tile(p, (n, 1)).T * M[idx, :]
        u[idx] -= p
        dx += np.mean(np.abs(p))

    # Taking the square root is contracted into a division by 2
    d = np.exp(u / 2)
    S = S * np.exp(L)
    return S, d
