from pesexp.mep.neb import EnergyWeightedSpringConstant, NEB
from ase.atoms import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE


def test_energy_weighted_k():
    k = EnergyWeightedSpringConstant(k_lower=0.01, k_upper=0.1, e_ref=0.0)

    def spring(e1, e2, emax=2.0):
        im1 = Atoms(["H"])
        im1.calc = SinglePointCalculator(im1, energy=e1)
        im2 = im1.copy()
        im2.calc = SinglePointCalculator(im2, energy=e2)
        im3 = im1.copy()
        im3.calc = SinglePointCalculator(im3, energy=emax)
        neb = NEB([im1, im2, im3], k=k)
        return neb.k[0]

    # Assert symmetry.
    assert spring(0.3, 0.7) == spring(0.7, 0.3)
    # Assert that only higher energy matters.
    assert spring(0.8, 0.4) == spring(0.8, 0.2)
    # Assert k is never lower than kl.
    # Note, a similar test for upper bound would fail since
    # both energies are expected to be lower than emax.
    assert spring(-1, -1) == k.k_lower


def test_tetrahedron():
    # Build a tetrahedral cluster of 4 LennardJones atoms with side
    # lengths 2^(1/6) corresponding to r0 of the LJ potential
    a = 2 ** (1 / 6)
    initial = Atoms(
        ["H"] * 4,
        [
            [a / 3**0.5, 0, 0],
            [-a / (2 * 3**0.5), a / 2, 0],
            [-a / (2 * 3**0.5), -a / 2, 0],
            [0, 0, a * (2 / 3) ** 0.5],
        ],
    )

    final = initial.copy()
    final.positions[-1, :] = -initial.positions[-1, :]

    images = [initial] + [final.copy() for _ in range(7)] + [final]
    for image in images:
        image.calc = LennardJones()

    k = EnergyWeightedSpringConstant(
        k_lower=0.1, k_upper=1.0, e_ref=images[0].get_potential_energy()
    )
    neb = NEB(images, k=k, method="improvedtangent")
    neb.interpolate()

    opt = FIRE(neb)
    opt.run(fmax=0.05, steps=200)

    for i, val in enumerate(neb.k):
        if i == neb.imax or i == neb.imax - 1:
            # Spring constants next to minimum should equal ku
            assert val == k.k_upper
        else:
            # Others should between k_lower and k_upper
            assert k.k_lower < val < k.k_upper
