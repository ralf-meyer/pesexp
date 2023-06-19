import pathlib
import logging
import numpy as np
import ase.io
import ase.units

from pesexp.calculators import TeraChem

logger = logging.getLogger(__name__)


def read_molecule(terachem_file):
    # Convert to path. This allows to finds the path of the xyz file later.
    terachem_file = pathlib.Path(terachem_file)
    with open(terachem_file, "r") as fin:
        lines = fin.readlines()
    # Set defaults
    charge = 0.0
    spin = 1.0
    for line in lines:
        if line.startswith("coordinates"):
            xyz_path = terachem_file.parent.joinpath(line.split()[1])
        elif line.startswith("charge"):
            charge = float(line.split()[1])
        elif line.startswith("spinmult"):
            spin = float(line.split()[1])
    logger.debug(
        f"Read terachem file with xyz: {xyz_path}, " f"charge: {charge}, spin: {spin}"
    )
    atoms = ase.io.read(xyz_path)
    # Assign spin and charge to first atom
    q = np.zeros_like(atoms.get_initial_charges())
    q[0] = charge
    atoms.set_initial_charges(q)
    s = np.zeros_like(atoms.get_initial_magnetic_moments())
    # Number of unpaired electrons is multiplicity - 1
    s[0] = spin - 1
    atoms.set_initial_magnetic_moments(s)
    return atoms, xyz_path


def read_terachem_input(terachem_file):
    atoms, _ = read_molecule(terachem_file)
    terachem_file = pathlib.Path(terachem_file)
    with open(terachem_file, "r") as fin:
        lines = fin.readlines()
    params = {}
    for line in lines:
        if line.startswith("end"):
            break
        elif not line.startswith("#"):
            key, val = line.split()[:2]
            if key not in ["run", "coordinates"]:
                params[key] = val
    atoms.calc = TeraChem(**params)
    return atoms


def read_terachem_hessian(binary_file):
    # TODO: Unfinished/Untested first implementation
    # Not actually what the second value is
    N, dim = np.fromfile(binary_file, dtype=np.int32, count=2)
    # The next N * 4 entries are values for x y z and n_electron (?)
    # Followed by the actual Hessian matrix, offset is multiplied by 8 since
    # 64 bits -> 8 bytes
    H = (
        np.fromfile(binary_file, count=N * N * 3 * 3, offset=(2 + N * 4) * 8).reshape(
            3 * N, 3 * N
        )
        * ase.units.Hartree
        / ase.units.Bohr**2
    )
    # 3 x N x 3 Matrix at the end are the dipole moment derivatives
    return H


def read_terachem_frequencies(frequencies_file):
    with open(frequencies_file, "r") as fin:
        lines = fin.readlines()
    n_atoms = int(lines[0].split()[2])
    n_modes = int(lines[1].split()[3])

    # The following double list contraction gets all the frequencies from
    # the headers by slicing with a stride of 3*n_atoms + 4, where the + 4 is
    # necessary to account for one header, the actual frequencies,
    # one separator and one footer line.
    frequencies = np.array(
        [float(sp) for line in lines[4 :: 3 * n_atoms + 4] for sp in line.split()]
    )

    # The modes are read row by row in exactly the same way
    modes = np.zeros((3 * n_atoms, n_modes))
    for i in range(3 * n_atoms):
        modes[i, :] = [
            float(sp)
            for line in lines[6 + i :: 3 * n_atoms + 4]
            for sp in line[6:].split()
        ]
    return frequencies, modes


def read_orca_hessian(hessian_file):
    with open(hessian_file, "r") as fin:
        lines = fin.readlines()
    start_ind = lines.index("$hessian\n")
    n_coords = int(lines[start_ind + 1])
    H = np.zeros((n_coords, n_coords))
    # The Hessian is divided into blocks of 5 coordinates
    block_len = 5
    n_blocks = n_coords // block_len + 1
    hessian_lines = lines[start_ind + 2 : start_ind + 2 + (n_coords + 1) * n_blocks]
    # Filter out the header line at the beginning of every block
    hessian_lines = [
        hessian_lines[i] for i in range(len(hessian_lines)) if i % (n_coords + 1) != 0
    ]
    for i, line in enumerate(hessian_lines):
        block = i // n_coords
        row = i % n_coords
        values = [float(col) for col in line.split()[1:]]
        H[row, block_len * block : block_len * block + len(values)] = values
    # Convert from Hartee / Bohr^2 to eV / Ang^2
    H *= ase.units.Hartree / ase.units.Bohr**2
    return H


def read_orca_frequencies(orca_output):
    with open(orca_output, "r") as fin:
        lines = fin.readlines()

    line_iter = iter(lines)
    for line in line_iter:
        if line.startswith("VIBRATIONAL FREQUENCIES"):
            # Skip 4 lines:
            for _ in range(4):
                line = next(line_iter)
            # Parse lines until we hit a blank line:
            frequencies = []
            for line in line_iter:
                if not line.strip():
                    break
                frequencies.append(float(line.split()[1]))

    frequencies = np.array(frequencies) * ase.units.invcm
    # TODO Implement for modes
    modes = None
    return frequencies, modes
