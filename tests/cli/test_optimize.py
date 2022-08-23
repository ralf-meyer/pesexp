import ase.build
import ase.calculators.emt
from pesexp.cli.optimize import run_optimization


def test_run_optimization(tmp_path):
    class TerachemDummy(ase.calculators.emt.EMT):
        def __init__(self, *args, **kwargs):
            ase.calculators.emt.EMT.__init__(self, *args, **kwargs)
            self.parameters["method"] = "ub3lyp"
            self.parameters["scrdir"] = "./scr"

    atoms = ase.build.molecule("C2H6")
    atoms.calc = TerachemDummy()

    run_optimization(atoms, coords="anc", name=tmp_path / "test")

    ref_params = {
        "guess": "./scr/ca0 ./scr/cb0",
        "convthre": "3.0e-6",
        "threall": "1.0e-13",
    }
    for key, val in ref_params.items():
        assert atoms.calc.parameters[key] == val

    # Check that the trajectory file was saved
    assert (tmp_path / "test_optim.traj").exists()
