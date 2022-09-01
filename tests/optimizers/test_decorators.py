import ase.build
import ase.calculators
from pesexp.geometry.coordinate_systems import ApproximateNormalCoordinates
from pesexp.optimizers import RFO
from pesexp.optimizers.decorators import anc_rebuilding
from rmsd import kabsch_rmsd


def test_anc_rebuilding():

    atoms = ase.build.molecule("C2H6")
    atoms.calc = ase.calculators.emt.EMT()
    coord_set = ApproximateNormalCoordinates(atoms)

    atoms_ref = atoms.copy()
    atoms_ref.calc = ase.calculators.emt.EMT()

    opt = anc_rebuilding(RFO, n_rebuild=5)(atoms, coord_set)
    opt_ref = RFO(atoms_ref, coord_set)

    opt.run(fmax=0.001, steps=100)
    opt_ref.run(fmax=0.001, steps=100)
    assert opt.nsteps <= opt_ref.nsteps

    assert opt.converged()
    assert (
        kabsch_rmsd(atoms.get_positions(), atoms_ref.get_positions(), translate=True)
        < 1e-2
    )
