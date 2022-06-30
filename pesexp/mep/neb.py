import ase.neb
from typing import List
from collections import namedtuple


EnergyWeightedSpringConstant = namedtuple(
    'EnergyWeightedSpringConstant', 'k_lower k_upper e_ref')


class EnergySpringMethod(ase.neb.ImprovedTangentMethod):

    def __init__(self, neb, mu=0.1):
        super().__init__(neb)
        self.mu = mu

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        imgforce -= tangential_force * tangent
        energies = state.energies
        spring_force_1 = (self.mu * abs(energies[i - 1] - energies[i])
                          + spring1.nt * spring1.k)
        spring_force_2 = (self.mu * abs(energies[i + 1] - energies[i])
                          + spring2.nt * spring2.k)
        imgforce += (spring_force_2 - spring_force_1) * tangent


class NEB(ase.neb.NEB):

    def __init__(self, images, k=0.1, climb=False, parallel=False,
                 remove_rotation_and_translation=False, world=None,
                 method='aseneb', allow_shared_calculator=False,
                 precon=None, mu=0.1, **kwargs):
        # Implements two new behaviors:
        # - Energy weighted springs:
        #   achieved by implementing k as a @property and initializing
        #   ase.neb.NEB with a fixed constant of k = 0.1. As last step the
        #   actual requested value of k is asigned to the new instance
        #   (Necesarry to avoid type checking in ase.neb.NEB).
        # - Adding new neb methods:
        #   Again type checking has to be avoided by initilializing with an
        #   default value 'aseneb' and assigning the proper method after
        #   the instance is created.
        if method == 'energyspring':
            # Initialize using ase neb and replace by chosen method
            super().__init__(
                images, k=0.1, climb=climb, parallel=parallel,
                remove_rotation_and_translation=remove_rotation_and_translation,
                world=world, method='aseneb',
                allow_shared_calculator=allow_shared_calculator,
                precon=precon, **kwargs)
            self.method == 'energyspring'
            self.neb_method = EnergySpringMethod(self, mu=mu)
        else:
            super().__init__(
                images, k=0.1, climb=climb, parallel=parallel,
                remove_rotation_and_translation=remove_rotation_and_translation,
                world=world, method=method,
                allow_shared_calculator=allow_shared_calculator,
                precon=precon, **kwargs)
        # Assign actual spring constant
        self.k = k

    @property
    def k(self) -> List[float]:
        # Newly added functionality
        if isinstance(self._k, EnergyWeightedSpringConstant):
            k = []
            # Unfortunately self.energies is not guaranteed to be populated
            # properly (for aseneb method), therefore, the energies are
            # reevaluated here (should not need actual calculations)
            energies = [im.get_potential_energy() for im in self.images]
            emax = max(energies)
            for i in range(self.nimages - 1):
                ei = max(energies[i], energies[i+1])
                if ei > self._k.e_ref:
                    alpha = (emax - ei) / (emax - self._k.e_ref)
                    k.append((1 - alpha) * self._k.k_upper
                             + alpha * self._k.k_lower)
                else:
                    k.append(self._k.k_lower)
            return k
        else:  # Default case
            return list(self._k)

    @k.setter
    def k(self, value):
        # Default behavior:
        if isinstance(value, (float, int)):
            value = [value] * (self.nimages - 1)
        self._k = value
