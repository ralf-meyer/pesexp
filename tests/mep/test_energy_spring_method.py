import numpy as np
from pesexp.mep.neb import NEB
from pesexp.calculators.calculators import TwoDCalculator
from ase.atoms import Atoms
from ase.optimize import FIRE


class SineSquaredCalculator(TwoDCalculator):

    def energy(self, x, y):
        if 0. < x < 1.:
            return np.sin(np.pi * x)**2
        return 0.

    def gradient(self, x, y):
        if 0. < x < 1.:
            return np.pi * np.sin(2 * np.pi * x), 0.
        return 0., 0.


def test_energy_spring_method_1D():
    images = [Atoms(['H'], positions=[[x, 0., 0.]]) for
              x in [0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]]
    for im in images:
        im.calc = SineSquaredCalculator()
    neb = NEB(images, k=0.01, method='energyspring', mu=1.0)
    opt = FIRE(neb)
    opt.run(fmax=1e-5)

    # Since mu >> k the images should be equally distibuted in energy space:
    energies = np.array([im.get_potential_energy() for im in images])
    np.testing.assert_allclose(np.diff(energies),
                               [0.25]*4 + [-0.25]*4, atol=5e-3)

    # Switch k and mu to achieve equal distribution in space
    neb.k = [1.0] * (neb.nimages - 1)
    neb.neb_method.mu = 0.01
    opt = FIRE(neb)
    opt.run(fmax=1e-5)
    path = np.array([im.get_positions()[0, 0] for im in images])
    print(path)
    np.testing.assert_allclose(np.diff(path), [0.125]*8, atol=5e-3)
