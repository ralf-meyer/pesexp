from ase.calculators.calculator import (Calculator, all_changes)
from abc import abstractmethod


class MLCalculator(Calculator):
    """Base class for all machine learning calculators"""

    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None,
                 ignore_bad_restart_file=Calculator._deprecated,
                 label=None, atoms=None, C1=1e8, C2=1e8, **kwargs):
        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)

        self.C1 = C1
        self.C2 = C2
        self.atoms_train = []

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results['energy'], self.results['forces'] = self.predict(atoms)

    def add_data(self, atoms):
        self.atoms_train.append(atoms)

    @property
    def n_samples(self):
        return len(self.atoms_train)

    @abstractmethod
    def fit(self):
        """Fit on data datapoints in self.atoms_train"""

    @abstractmethod
    def predict(self, atoms):
        """Predict energy and forces for given atoms object"""

    @abstractmethod
    def get_params(self):
        """Return dictionary of the (fitted) model parameters"""

    @abstractmethod
    def set_params(self, **params):
        """Set the model parameters from a given dictionary"""
