import pathlib
import logging
import numpy as np
import ase.io

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
