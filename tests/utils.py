import os
import shutil
import tempfile
import pytest
import ase.collections
import ase.build
import ase.io
import geometric.molecule

g2_molecules = {}
for name in ase.collections.g2.names:
    atoms = ase.build.molecule(name)
    if len(atoms) == 1:  # Skip single atom systems
        continue
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "tmp.xyz")
        ase.io.write(tmp_file, atoms, plain=True)
        mol = geometric.molecule.Molecule(tmp_file)
    g2_molecules[name] = dict(atoms=atoms, mol=mol)


# Decorator to skip test is xtb is not installed
xtb_installed = pytest.mark.skipif(
    shutil.which("xtb") is None, reason="Could not find xtb installation"
)
