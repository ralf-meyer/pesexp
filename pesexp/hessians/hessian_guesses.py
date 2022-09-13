import os
import subprocess
import tempfile
import logging
import numpy as np
import ase.io
import ase.units
from pesexp.calculators import _xtb_methods, _openbabel_methods, get_calculator
from pesexp.geometry.connectivity import (
    find_connectivity,
    find_primitives,
    get_primitives,
)
from pesexp.geometry.primitives import Distance, Angle, LinearAngle, Dihedral, Improper


logger = logging.getLogger(__name__)


def get_hessian_guess(atoms, method):
    if method.lower() in _xtb_methods:
        return xtb_hessian(atoms, method)
    elif method.lower() in _openbabel_methods:
        old_calc = atoms.calc
        atoms.calc = get_calculator(method.lower())
        H = numerical_hessian(atoms)
        atoms.calc = old_calc
        return H
    elif method.lower() == "trivial":
        return TrivialGuessHessian().build(atoms)
    elif method.lower() == "schlegel":
        return SchlegelHessian().build(atoms)
    elif method.lower() == "fischer_almloef":
        return FischerAlmloefHessian().build(atoms)
    elif method.lower() == "lindh":
        return LindhHessian().build(atoms)
    else:
        raise NotImplementedError(f"Unknown hessian_guess {method}")


class TrivialGuessHessian:
    """Base class for guess Hessians constructed in internal coordinates. The force
    constants in this trivial implementation follow the suggestion in
    Baker et al., J. Chem. Phys. 105, 192 (1996)
    https://doi.org/10.1063/1.471864"""

    def __init__(self, threshold=1.35, h_trans=0.0, h_rot=0.0):
        """
        Parameters
        ----------
        threshold : float
            Atoms closer than this threshold times the sum of their covalent
            radii are considered bound. Default value is 1.35.
        h_trans : float
            Force constant for translations.
        h_rot : float
            Force constant for rotations.
        """
        self.threshold = threshold
        self.h_trans = h_trans
        self.h_rot = h_rot

    def build(self, atoms):
        """
        Parameters
        ----------
        atoms : ase.atoms.Atoms
            Arrangement of atoms.

        Returns
        -------
        H : numpy.ndarray
            Guess Hessian in cartesian coordinates and ase units (eV/Ang**2)
        """

        N = len(atoms)
        zs = atoms.get_atomic_numbers()
        xyzs = atoms.get_positions()
        bonds = find_connectivity(
            atoms, threshold=self.threshold**2, connect_fragments=True
        )
        primitives = get_primitives(
            atoms, threshold=self.threshold**2, connect_fragments=True
        )

        # Calculate the number of bonds on each atom for the torsion
        # coefficient in the Fischer Almloef Hessian.
        N_bonds = np.zeros(N)
        for b in bonds:
            N_bonds[b[0]] += 1
            N_bonds[b[1]] += 1

        # Initialize Hessian in Cartesian coordinates
        H = np.zeros((3 * N, 3 * N))

        for prim in primitives:
            # Default for "unknown" primitives
            h_ii = 0.0
            if type(prim) is Distance:
                h_ii = self.distance(xyzs[prim.i], xyzs[prim.j], zs[prim.i], zs[prim.j])
            elif type(prim) is Angle:
                h_ii = self.angle(
                    xyzs[prim.i],
                    xyzs[prim.j],
                    xyzs[prim.k],
                    zs[prim.i],
                    zs[prim.j],
                    zs[prim.k],
                )
            elif type(prim) is LinearAngle:
                h_ii = self.linear_angle(
                    xyzs[prim.i],
                    xyzs[prim.j],
                    xyzs[prim.k],
                    zs[prim.i],
                    zs[prim.j],
                    zs[prim.k],
                )
            elif type(prim) is Dihedral:
                h_ii = self.dihedral(
                    xyzs[prim.i],
                    xyzs[prim.j],
                    xyzs[prim.k],
                    xyzs[prim.l],
                    zs[prim.i],
                    zs[prim.j],
                    zs[prim.k],
                    zs[prim.l],
                    N_bonds[prim.j],
                    N_bonds[prim.k],
                )
            elif type(prim) is Improper:
                h_ii = self.improper(
                    xyzs[prim.i],
                    xyzs[prim.j],
                    xyzs[prim.k],
                    xyzs[prim.l],
                    zs[prim.i],
                    zs[prim.j],
                    zs[prim.k],
                    zs[prim.l],
                )
            Bi = prim.derivative(xyzs)
            H += np.outer(Bi, h_ii * Bi)

        # Add translation force constants for all three Cartesian displacements
        for coord in range(3):
            # The outer product of a displacement vector along the x-axis:
            # [1, 0, 0, 1, 0, 0, ..., 1, 0, 0] corresponds to the
            # indexing [0::3, 0::3]. The factor 1/N is used to normalize the
            # displacement vector.
            H[coord::3, coord::3] += self.h_trans / N

        # Add force constants for rotation about the three Cartesian axes
        center = np.mean(xyzs, axis=0)
        for ax in np.eye(3):
            # A rotation about the geometric center is characterized by the
            # following B vector.
            B = np.cross(xyzs - center, ax).flatten()
            norm_B = np.linalg.norm(B)
            if norm_B > 0.0:
                B /= norm_B
            H += self.h_rot * np.outer(B, B)
        return H

    def distance(self, xyz_i, xyz_j, z_i, z_j):
        return 0.5 * ase.units.Hartree / ase.units.Bohr**2

    def angle(self, xyz_i, xyz_j, xyz_k, z_i, z_j, z_k):
        return 0.2 * ase.units.Hartree

    def linear_angle(self, xyz_i, xyz_j, xyz_k, z_i, z_j, z_k):
        return self.angle(xyz_i, xyz_j, xyz_k, z_i, z_j, z_k)

    def dihedral(
        self, xyz_i, xyz_j, xyz_k, xyz_l, z_i, z_j, z_k, z_l, bonds_j, bonds_k
    ):
        return 0.1 * ase.units.Hartree

    def improper(self, xyz_i, xyz_j, xyz_k, xyz_l, z_i, z_j, z_k, z_l):
        return 0.1 * ase.units.Hartree


class SchlegelHessian(TrivialGuessHessian):
    """
    Schlegel, Theoret. Chim. Acta 66, 333-340 (1984).
    https://doi.org/10.1007/BF00554788
    """

    def get_B_str(self, z_1, z_2):
        """
        Returns the B parameter for stretch coordinates given two atomic
        numbers
        """
        # Sort for simplicity
        z_1, z_2 = min(z_1, z_2), max(z_1, z_2)
        if z_1 <= 2:  # First period
            if z_2 <= 2:  # first-first
                return -0.244
            elif z_2 <= 10:  # first-second
                return 0.352
            else:  # first-third+
                return 0.660
        elif z_1 <= 10:  # Second period
            if z_2 <= 10:  # second-second
                return 1.085
            else:  # second-third+
                return 1.522
        else:  # third+-third+
            return 2.068

    def distance(self, xyz_i, xyz_j, z_i, z_j):
        r = np.linalg.norm(xyz_j - xyz_i)
        r_cov = ase.data.covalent_radii[z_i] + ase.data.covalent_radii[z_j]
        if r < self.threshold * r_cov:
            B = self.get_B_str(z_i, z_j)
            return (
                1.734
                * ase.units.Hartree
                * ase.units.Bohr
                / (r - B * ase.units.Bohr) ** 3
            )
        else:
            # Not covalently bonded atoms (from fragment connection).
            # Following geomeTRIC those are assigned a fixed value:
            return 0.1 * ase.units.Hartree / ase.units.Bohr**2

    def angle(self, xyz_i, xyz_j, xyz_k, z_i, z_j, z_k):
        if z_i == 1 or z_k == 1:
            # "either or both terminal atoms hydrogen"
            return 0.160 * ase.units.Hartree
        # "all three heavy atom bends"
        return 0.250 * ase.units.Hartree

    def dihedral(
        self, xyz_i, xyz_j, xyz_k, xyz_l, z_i, z_j, z_k, z_l, bonds_j, bonds_k
    ):
        r = np.linalg.norm(xyz_j - xyz_k)
        r_cov = ase.data.covalent_radii[z_j] + ase.data.covalent_radii[z_k]
        # Follows the implementation in Psi4: line 177 in
        # https://github.com/psi4/psi4/blob/d9093c75c71c2b33fbe86f32b25d138675ac22eb/psi4/src/psi4/optking/frag_H_guess.cc
        A = 0.0023 * ase.units.Hartree
        B = 0.07 * ase.units.Hartree / ase.units.Bohr
        if r < r_cov + A / B:
            return A - B * (r - r_cov)
        else:
            return A

    def improper(self, xyz_i, xyz_j, xyz_k, xyz_l, z_i, z_j, z_k, z_l):
        r1 = xyz_j - xyz_i
        r2 = xyz_k - xyz_i
        r3 = xyz_l - xyz_i
        # Additional np.abs() since we do not know the orientation of r1
        # with respect to r2 x r3.
        d = 1 - np.abs(np.dot(r1, np.cross(r2, r3))) / (
            np.linalg.norm(r1) * np.linalg.norm(r2) * np.linalg.norm(r3)
        )
        return 0.045 * ase.units.Hartree * d**4


class FischerAlmloefHessian(TrivialGuessHessian):
    """
    Fischer and Almloef, J. Phys. Chem. 1992, 96, 24, 9768-9774
    https://doi.org/10.1021/j100203a036
    """

    def distance(self, xyz_i, xyz_j, z_i, z_j):
        # "Bond stretch between atoms a and b"
        r_ab = np.linalg.norm(xyz_j - xyz_i)
        r_ab_cov = ase.data.covalent_radii[z_i] + ase.data.covalent_radii[z_j]
        A = 0.3601 * ase.units.Hartree / ase.units.Bohr**2
        B = 1.944 / ase.units.Bohr
        if r_ab < self.threshold * r_ab_cov:
            return A * np.exp(-B * (r_ab - r_ab_cov))
        else:
            # Not covalently bonded atoms (from fragment connection).
            # Following geomeTRIC those are assigned a fixed value
            return 0.1 * ase.units.Hartree / ase.units.Bohr**2

    def angle(self, xyz_i, xyz_j, xyz_k, z_i, z_j, z_k):
        # "Valence angle bend formed by atoms b-a-c"
        r_ab = np.linalg.norm(xyz_i - xyz_j)
        r_ac = np.linalg.norm(xyz_k - xyz_j)
        r_ab_cov = ase.data.covalent_radii[z_i] + ase.data.covalent_radii[z_j]
        r_ac_cov = ase.data.covalent_radii[z_k] + ase.data.covalent_radii[z_j]
        A = 0.089 * ase.units.Hartree
        B = 0.11 * ase.units.Hartree
        C = 0.44 / ase.units.Bohr
        D = -0.42
        return A + B / (r_ab_cov * r_ac_cov / ase.units.Bohr**2) ** D * np.exp(
            -C * (r_ab + r_ac - r_ab_cov - r_ac_cov)
        )

    def dihedral(
        self, xyz_i, xyz_j, xyz_k, xyz_l, z_i, z_j, z_k, z_l, bonds_j, bonds_k
    ):
        # "Torsion about the central bond between atoms a and b"
        r_ab = np.linalg.norm(xyz_j - xyz_k)
        r_ab_cov = ase.data.covalent_radii[z_j] + ase.data.covalent_radii[z_k]
        # "L, number of bonds connected to atom a and b (except the
        # central bond)"
        L = bonds_j - 1 + bonds_k - 1
        A = 0.0015 * ase.units.Hartree
        B = 14.0 * ase.units.Hartree
        C = 2.85 / ase.units.Bohr
        D = 0.57
        E = 4.0
        return A + B * L**D / (r_ab * r_ab_cov / ase.units.Bohr**2) ** E * np.exp(
            -C * (r_ab - r_ab_cov)
        )

    def improper(self, xyz_i, xyz_j, xyz_k, xyz_l, z_i, z_j, z_k, z_l):
        # "Out-of-plane bend of atom x and the plane formed by atoms a, b,
        # and c where atom x is connected to atom a"
        r_ax = np.linalg.norm(xyz_j - xyz_i)
        r_ax_cov = ase.data.covalent_radii[z_i] + ase.data.covalent_radii[z_j]
        r_ab_cov = ase.data.covalent_radii[z_j] + ase.data.covalent_radii[z_k]
        r_ac_cov = ase.data.covalent_radii[z_l] + ase.data.covalent_radii[z_k]

        n_1 = np.cross(xyz_i - xyz_j, xyz_k - xyz_j)
        n_2 = np.cross(xyz_j - xyz_k, xyz_l - xyz_k)
        cos_phi = np.dot(n_1, n_2) / np.linalg.norm(n_1) / np.linalg.norm(n_2)

        A = 0.0025 * ase.units.Hartree
        B = 0.0061 * ase.units.Hartree
        C = 3.00 / ase.units.Bohr
        D = 4.0
        E = 0.8
        return A + B * (
            r_ab_cov * r_ac_cov / ase.units.Bohr**2
        ) ** E * cos_phi**D * np.exp(-C * (r_ax - r_ax_cov))


class LindhHessian:
    def __init__(self, threshold=1e-5, h_trans=0.0, h_rot=0.0):
        """
        Parameters
        ----------
        threshold : float
            Only pairs of atoms with a rho value above this threshold are
            considered bonded. Increasing this threshold can significantly
            speed up the construction of the Hessian by reducing the
            amount of primitives that are used for the guess.
        h_trans : float
            Force constant for translations.
        h_rot : float
            Force constant for rotations.
        """
        self.threshold = threshold
        self.h_trans = h_trans
        self.h_rot = h_rot

    def build(self, atoms):
        """
        Parameters
        ----------
        atoms : ase.atoms.Atoms
            Arrangement of atoms.

        Returns
        -------
        H : numpy.ndarray
            Guess Hessian in cartesian coordinates and ase units (eV/Ang**2)
        """

        N = len(atoms)
        zs = atoms.get_atomic_numbers()
        xyzs = atoms.get_positions()

        rho = np.zeros((N, N))
        bonds = []
        # All atoms are considered connected
        for i in range(N):
            for j in range(i + 1, N):
                r = np.linalg.norm(xyzs[i] - xyzs[j]) / ase.units.Bohr
                alpha, r_ref = self.get_parameters(zs[i], zs[j])
                rho[i, j] = rho[j, i] = np.exp(alpha * (r_ref**2 - r**2))
                if rho[i, j] > self.threshold:
                    bonds.append((i, j))

        bends, _, torsions, _ = find_primitives(
            xyzs, bonds, linear_flag=False, planar_threshold=1.0
        )

        primitives = (
            [Distance(*b) for b in bonds]
            + [Angle(*a) for a in bends]
            + [Dihedral(*d) for d in torsions]
        )

        # Initialize Hessian in Cartesian coordinates
        H = np.zeros((3 * N, 3 * N))

        k_r = 0.45 * ase.units.Hartree / ase.units.Bohr**2
        k_phi = 0.15 * ase.units.Hartree
        k_tau = 0.005 * ase.units.Hartree
        for prim in primitives:
            # Default for "unknown" primitives
            h_ii = 0.0
            if type(prim) is Distance:
                h_ii = k_r * rho[prim.i, prim.j]
            elif type(prim) is Angle:
                h_ii = k_phi * rho[prim.i, prim.j] * rho[prim.j, prim.k]
            elif type(prim) is Dihedral:
                h_ii = (
                    k_tau
                    * rho[prim.i, prim.j]
                    * rho[prim.j, prim.k]
                    * rho[prim.k, prim.l]
                )
            Bi = prim.derivative(xyzs)
            H += np.outer(Bi, h_ii * Bi)

        # Add translation force constants for all three Cartesian displacements
        for coord in range(3):
            # The outer product of a displacement vector along the x-axis:
            # [1, 0, 0, 1, 0, 0, ..., 1, 0, 0] corresponds to the
            # indexing [0::3, 0::3]. The factor 1/N is used to normalize the
            # displacement vector.
            H[coord::3, coord::3] += self.h_trans / N

        # Add force constants for rotation about the three Cartesian axes
        center = np.mean(xyzs, axis=0)
        for ax in np.eye(3):
            # A rotation about the geometric center is characterized by the
            # following B vector.
            B = np.cross(xyzs - center, ax).flatten()
            norm_B = np.linalg.norm(B)
            if norm_B > 0.0:
                B /= norm_B
            H += self.h_rot * np.outer(B, B)
        return H

    def get_parameters(self, z_1, z_2):
        # Sort for simplicity
        z_1, z_2 = min(z_1, z_2), max(z_1, z_2)
        if z_1 <= 2:  # First period
            if z_2 <= 2:  # first-first
                return 1.0, 1.35
            elif z_2 <= 10:  # first-second
                return 0.3949, 2.10
            else:  # first-third+
                return 0.3949, 2.53
        elif z_1 <= 10:  # Second period
            if z_2 <= 10:  # second-second
                return 0.28, 2.87
            else:  # second-third+
                return 0.28, 3.4
        else:  # third+-third+
            return 0.28, 3.4


def xtb_hessian(atoms, method):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write .xyz file
        ase.io.write(os.path.join(tmpdir, "tmp.xyz"), atoms, plain=True)
        with open(os.path.join(tmpdir, "xtb.inp"), "w") as fout:
            fout.write("$symmetry\n")
            fout.write("  maxat=0\n")
            fout.write("$end\n")
        os.path.join(tmpdir, "tmp.xyz")
        try:
            output = subprocess.run(
                ["xtb", "--input", "xtb.inp", "--hess", "tmp.xyz"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise ChildProcessError(
                "Could not find subprocess xtb. Ensure xtb"
                " is installed and properly configured."
            )
        if output.returncode != 0:
            print(output)
            raise ChildProcessError("XTB calculation failed")
        H = read_xtb_hessian(os.path.join(tmpdir, "hessian"))
    return H


def read_xtb_hessian(file):
    with open(file, "r") as fin:
        content = fin.read()
    values = np.array([float(f) for f in content.split()[1:]])
    N = int(np.sqrt(values.size))
    return values.reshape(N, N) * ase.units.Hartree / ase.units.Bohr**2


def numerical_hessian(atoms, step=1e-5, symmetrize=True, method="forward"):
    N = len(atoms)
    x0 = atoms.get_positions()
    H = np.zeros((3 * N, 3 * N))

    if method not in ["forward", "central"]:
        raise NotImplementedError(f"Unknown method {method}")

    if method == "forward":
        g0 = -atoms.get_forces().flatten()

    for i in range(N):
        for c in range(3):
            logger.debug(f"Displacements for coordinate {3*i+c}/{3*N}")
            x = x0.copy()
            x[i, c] += step
            atoms.set_positions(x)
            g_plus = -atoms.get_forces().flatten()

            if method == "central":
                x = x0.copy()
                x[i, c] -= step
                atoms.set_positions(x)
                g_minus = -atoms.get_forces().flatten()
                H[3 * i + c, :] = (g_plus - g_minus) / (2 * step)
            elif method == "forward":
                H[3 * i + c, :] = (g_plus - g0) / step
    atoms.set_positions(x0)
    if symmetrize:
        return 0.5 * (H + H.T)
    return H


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
