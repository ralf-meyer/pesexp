import numpy as np
from abc import abstractmethod


class Primitive():

    @abstractmethod
    def value(self, xyzs):
        """ Evaluate the Primitive on an (N, 3) array"""

    @abstractmethod
    def derivative(self, xyzs):
        """ Calculate the derivative for the Primitive on an (N, 3) array"""

    def diff(self, xyzs1, xyzs2):
        return self.value(xyzs1) - self.value(xyzs2)


class Cartesian(Primitive):

    def __init__(self, i, axis=0):
        self.i = i
        self.axis = axis

    def __repr__(self):
        return f'Cartesian({self.i}, axis={self.axis})'

    def value(self, xyzs):
        return xyzs[self.i, self.axis]

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        dq[3*self.i+self.axis] = 1.0
        return dq


class Distance(Primitive):

    def __init__(self, i, j):
        self.i, self.j = i, j

    def __repr__(self):
        return f'Distance({self.i}, {self.j})'

    def value(self, xyzs):
        rij = xyzs[self.i, :] - xyzs[self.j, :]
        return np.linalg.norm(rij)

    def derivative(self, xyzs):
        rij = xyzs[self.i, :] - xyzs[self.j, :]
        r = np.linalg.norm(rij)
        dr = np.zeros(xyzs.size)
        dr[3*self.i:3*(self.i+1)] = rij/r
        dr[3*self.j:3*(self.j+1)] = -rij/r
        return dr


class InverseDistance(Distance):

    def __repr__(self):
        return f'InverseDistance({self.i}, {self.j})'

    def value(self, xyzs):
        return 1./Distance.value(self, xyzs)

    def derivative(self, xyzs):
        rij = xyzs[self.i, :] - xyzs[self.j, :]
        q = 1.0/np.linalg.norm(rij)
        dq = np.zeros(xyzs.size)
        dq[3*self.i:3*(self.i+1)] = -rij * q**3
        dq[3*self.j:3*(self.j+1)] = rij * q**3
        return dq


class Angle(Primitive):
    """Bakken and Helgaker, J. Chem. Phys. 117, 9160 (2002)
    https://doi.org/10.1063/1.1515483
    """

    def __init__(self, i, j, k):
        """Atom j is center of angle
        """
        self.i, self.j, self.k = i, j, k

    def __repr__(self):
        return f'Angle({self.i}, {self.j}, {self.k})'

    def value(self, xyzs):
        u = xyzs[self.i, :] - xyzs[self.j, :]
        u /= np.linalg.norm(u)
        v = xyzs[self.k, :] - xyzs[self.j, :]
        v /= np.linalg.norm(v)
        return np.arccos(np.dot(u, v))

    def derivative(self, xyzs):
        u = xyzs[self.i, :] - xyzs[self.j, :]
        norm_u = np.linalg.norm(u)
        u /= norm_u
        v = xyzs[self.k, :] - xyzs[self.j, :]
        norm_v = np.linalg.norm(v)
        v /= norm_v
        w = np.cross(u, v)
        norm_w = np.linalg.norm(w)
        if norm_w > 0:
            w /= norm_w
        else:
            w = np.cross(u, np.array([1, -1, 1]))
            norm_w = np.linalg.norm(w)
            if norm_w > 0:
                w /= norm_w
            else:
                w = np.cross(u, np.array([-1, 1, 1]))
                w /= np.linalg.norm(w)
        dt = np.zeros(xyzs.size)
        cross_u = np.cross(u, w)/norm_u
        cross_v = np.cross(w, v)/norm_v
        dt[3*self.i:3*(self.i+1)] = cross_u
        dt[3*self.j:3*(self.j+1)] = - cross_u - cross_v
        dt[3*self.k:3*(self.k+1)] = cross_v
        return dt


class LinearAngle(Primitive):

    def __init__(self, i, j, k, axis):
        """Closely follows the implementation in geomeTRIC

        Parameters
        ----------
        i : int
            Index of the first atom
        j : int
            Index of the center atom
        k : int
            Index of the last atom
        axis : int
            Projection axis. Can take values 0 or 1.
        """
        self.i, self.j, self.k = i, j, k
        self.axis = axis
        self.eref = None

    def __repr__(self):
        return f'LinearAngle({self.i}, {self.j}, {self.k}, axis={self.axis})'

    def _calc_reference(self, xyzs):
        rik = xyzs[self.k, :] - xyzs[self.i, :]
        # Cartesian axes.
        cart_vecs = np.eye(3)
        # Select Cartesian axis with the least overlap with rik as
        # reference direction.
        ind = np.argmin([np.dot(ei, rik)**2 for ei in cart_vecs])
        self.eref = cart_vecs[ind]

    def value(self, xyzs):
        # Unit vector pointing from i to k.
        rik = xyzs[self.k, :] - xyzs[self.i, :]
        eik = rik / np.linalg.norm(rik)
        rji = xyzs[self.i, :] - xyzs[self.j, :]
        eji = rji / np.linalg.norm(rji)
        rjk = xyzs[self.k, :] - xyzs[self.j, :]
        ejk = rjk / np.linalg.norm(rjk)

        if self.eref is None:
            self._calc_reference(xyzs)
        # Define the vector u perpendicular to rik using the reference vector
        u = np.cross(eik, self.eref)
        u /= np.linalg.norm(u)

        if self.axis == 0:
            return np.dot(eji, u) + np.dot(ejk, u)
        # Else use a vector w perpendicular to rik and u as projection axis.
        # Since eik and u are perpendicular and normalized w is normalized by
        # construction.
        w = np.cross(eik, u)
        return np.dot(eji, w) + np.dot(ejk, w)

    def derivative(self, xyzs):
        # Initialize return array
        dt = np.zeros(xyzs.size)

        rik = xyzs[self.k, :] - xyzs[self.i, :]
        norm_ik = np.linalg.norm(rik)
        eik = rik / norm_ik
        # deik/drik
        deik = np.eye(3) / norm_ik - np.outer(rik, rik) / norm_ik**3

        rji = xyzs[self.i, :] - xyzs[self.j, :]
        norm_ji = np.linalg.norm(rji)
        eji = rji / norm_ji
        # deji/drji
        deji = np.eye(3) / norm_ji - np.outer(rji, rji) / norm_ji**3

        rjk = xyzs[self.k, :] - xyzs[self.j, :]
        norm_jk = np.linalg.norm(rjk)
        ejk = rjk / norm_jk
        # dejk/drjk
        dejk = np.eye(3) / norm_jk - np.outer(rjk, rjk) / norm_jk**3

        # Setup first projection axis u
        if self.eref is None:
            self._calc_reference(xyzs)
        u_raw = np.cross(eik, self.eref)
        # Since eref is constant: deref/drik = 0. Caution: While other
        # derivative matrices defined here are symmetric du_raw is not!
        du_raw = np.cross(deik, self.eref, axis=0)
        # Normalization
        norm_u = np.linalg.norm(u_raw)
        u = u_raw / norm_u
        # Inner derivative of norm_u in the second term is again du_raw
        du = du_raw / norm_u - np.outer(u_raw, u_raw) @ du_raw / norm_u**3

        if self.axis == 0:
            # derivative w.r.t. atom i: drji/dri = 1, drik/dri = -1
            dt[3*self.i:3*(self.i+1)] = (np.dot(deji, u) + np.dot(eji, -du)
                                         + np.dot(ejk, -du))
            # derivative w.r.t. atom j: drji/drj = -1, drjk/drj = -1
            # u is independent of atom j : du/drj = 0
            dt[3*self.j:3*(self.j+1)] = np.dot(-deji, u) + np.dot(-dejk, u)
            # derivative w.r.t atom k: drik/drk = 1, drjk/drk = 1
            dt[3*self.k:3*(self.k+1)] = (np.dot(dejk, u) + np.dot(eji, du)
                                         + np.dot(ejk, du))
        else:
            # Setup second projection axis
            w = np.cross(eik, u)
            # Derivative w.r.t rik
            dw = np.cross(deik, u, axis=0) + np.cross(eik, du, axis=0)
            # derivative w.r.t. atom i: drji/dri = 1, drik/dri = -1
            dt[3*self.i:3*(self.i+1)] = (np.dot(deji, w) + np.dot(eji, -dw)
                                         + np.dot(ejk, -dw))
            # derivative w.r.t. atom j: drji/drj = -1, drjk/drj = -1
            # w is independent of the atom j: dw/drj = 0
            dt[3*self.j:3*(self.j+1)] = np.dot(-deji, w) + np.dot(-dejk, w)
            # derivative w.r.t atom k: drik/drk = 1, drjk/drk = 1
            dt[3*self.k:3*(self.k+1)] = (np.dot(dejk, w) + np.dot(eji, dw)
                                         + np.dot(ejk, dw))
        return dt


class Dihedral(Primitive):

    def __init__(self, i, j, k, l):  # noqa: E741
        """Implementation follows:
        Blondel, A. and Karplus, M., J. Comput. Chem., 17: 1132-1141. (1996)
        """
        self.i, self.j, self.k, self.l = i, j, k, l  # noqa: E741

    def __repr__(self):
        return f'Dihedral({self.i}, {self.j}, {self.k}, {self.l})'

    def value(self, xyzs):
        f = xyzs[self.i, :] - xyzs[self.j, :]
        g = xyzs[self.j, :] - xyzs[self.k, :]
        h = xyzs[self.l, :] - xyzs[self.k, :]
        a = np.cross(f, g)
        b = np.cross(h, g)
        norm_g = np.linalg.norm(g)
        w = np.arctan2(
                np.dot(np.cross(b, a), g/norm_g),
                np.dot(a, b))
        return w

    def derivative(self, xyzs):
        """Formula 27 (i,j,k,l)"""
        f = xyzs[self.i, :] - xyzs[self.j, :]
        g = xyzs[self.j, :] - xyzs[self.k, :]
        h = xyzs[self.l, :] - xyzs[self.k, :]
        a = np.cross(f, g)
        b = np.cross(h, g)
        norm_g = np.linalg.norm(g)
        a_sq = np.dot(a, a)
        b_sq = np.dot(b, b)
        dw = np.zeros(xyzs.size)
        dw[3*self.i:3*(self.i+1)] = - norm_g/a_sq*a
        dw[3*self.j:3*(self.j+1)] = (norm_g/a_sq*a
                                     + np.dot(f, g)/(a_sq*norm_g)*a
                                     - np.dot(h, g)/(b_sq*norm_g)*b)
        dw[3*self.k:3*(self.k+1)] = (np.dot(h, g)/(b_sq*norm_g)*b
                                     - np.dot(f, g)/(a_sq*norm_g)*a
                                     - norm_g/b_sq*b)
        dw[3*self.l:3*(self.l+1)] = norm_g/b_sq*b
        return dw

    def diff(self, xyzs1, xyzs2):
        dw = self.value(xyzs1) - self.value(xyzs2)
        if dw > np.pi:
            return dw - 2*np.pi
        if dw < -np.pi:
            return dw + 2*np.pi
        return dw


class Improper(Dihedral):
    """Alias for Dihedral since it is often necessary to distinguish between
    actual dihedrals and improper (out-of-plane) bends."""

    def __repr__(self):
        return f'Improper({self.i}, {self.j}, {self.k}, {self.l})'


class Octahedral(Primitive):
    """
    Base class for the 15 octahedral coordinates.
    Assumes the following arrangement of atoms:
            a5  a4
            | /
            |/
    a1 - - a0 - - a3
           /|
          / |
        a2  a6
    """

    def __init__(self, a0, a1, a2, a3, a4, a5, a6):
        self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6 = (
            a0, a1, a2, a3, a4, a5, a6)
        self.distances = [Distance(a0, ai) for ai in [a1, a2, a3, a4, a5, a6]]


class OctahedralA1g(Octahedral):
    """Breathing mode"""

    def value(self, xyzs):
        return sum([d.value(xyzs) for d in self.distances])

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        for d in self.distances:
            dq += d.derivative(xyzs)
        return dq


class OctahedralEg1(Octahedral):

    def value(self, xyzs):
        q = (self.distances[0].value(xyzs)
             + self.distances[1].value(xyzs)
             + self.distances[2].value(xyzs)
             + self.distances[3].value(xyzs)
             - 2 * self.distances[4].value(xyzs)
             - 2 * self.distances[5].value(xyzs))
        return q

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        dq += self.distances[0].derivative(xyzs)
        dq += self.distances[1].derivative(xyzs)
        dq += self.distances[2].derivative(xyzs)
        dq += self.distances[3].derivative(xyzs)
        dq += -2 * self.distances[4].derivative(xyzs)
        dq += -2 * self.distances[5].derivative(xyzs)
        return dq


class OctahedralEg2(Octahedral):

    def value(self, xyzs):
        q = (self.distances[0].value(xyzs)
             - self.distances[1].value(xyzs)
             + self.distances[2].value(xyzs)
             - self.distances[3].value(xyzs))
        return q

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        dq += self.distances[0].derivative(xyzs)
        dq += -self.distances[1].derivative(xyzs)
        dq += self.distances[2].derivative(xyzs)
        dq += -self.distances[3].derivative(xyzs)
        return dq


class OctahedralT1u1(Octahedral):

    def value(self, xyzs):
        return self.distances[0].value(xyzs) - self.distances[2].value(xyzs)

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        dq += self.distances[0].derivative(xyzs)
        dq += -self.distances[2].derivative(xyzs)
        return dq


class OctahedralT1u2(Octahedral):

    def value(self, xyzs):
        return self.distances[1].value(xyzs) - self.distances[3].value(xyzs)

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        dq += self.distances[1].derivative(xyzs)
        dq += -self.distances[3].derivative(xyzs)
        return dq


class OctahedralT1u3(Octahedral):

    def value(self, xyzs):
        return self.distances[4].value(xyzs) - self.distances[5].value(xyzs)

    def derivative(self, xyzs):
        dq = np.zeros(xyzs.size)
        dq += self.distances[4].derivative(xyzs)
        dq += -self.distances[5].derivative(xyzs)
        return dq
