import io
import numpy as np
import ase.io
import ase.units
from ase.calculators.calculator import (Calculator, FileIOCalculator,
                                        all_changes)
from openbabel import openbabel


class TeraChem(FileIOCalculator):
    """
    TeraChem calculator
    """
    name = 'TeraChem'

    implemented_properties = ['energy', 'forces']
    command = 'terachem PREFIX.inp > PREFIX.out'

    # Following the minimal requirements given in
    # http://www.petachem.com/doc/userguide.pdf
    default_parameters = {'run': 'gradient',
                          'method': 'blyp',
                          'basis': '6-31g*',
                          'charge': 0}

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='terachem', atoms=None, **kwargs):
        """
        """

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def read(self, label):
        raise NotImplementedError()

    def read_results(self):
        with open(f'{self.label}.out', 'r') as fileobj:
            N_atoms = 0
            lineiter = iter(fileobj)
            for line in lineiter:
                if 'Total atoms:' in line:
                    N_atoms = int(line.split()[2])
                elif 'FINAL ENERGY:' in line:
                    self.results['energy'] = float(
                        line.split()[2]) * ase.units.Hartree
                elif 'Gradient units are Hartree/Bohr' in line:
                    gradient = np.zeros((N_atoms, 3))
                    # Skip next two lines
                    next(lineiter)
                    next(lineiter)
                    for i in range(N_atoms):
                        gradient[i, :] = list(
                            map(float, next(lineiter).split()))
                    # Convert gradient to ASE forces
                    self.results['forces'] = (
                        - gradient * ase.units.Hartree / ase.units.Bohr)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # Write geometry as .xyz
        ase.io.write(f'{self.label}.xyz', atoms, plain=True)

        with open(f'{self.label}.inp', 'w') as fileobj:
            fileobj.write('# ASE generated input file\n')

            # Add xyz path to input file
            fileobj.write(f'{"coordinates":25s}   {self.prefix}.xyz\n')

            for key, value in self.parameters.items():
                fileobj.write(f'{key:25s}   {value}\n')

            fileobj.write('end\n')


class OpenbabelFF(Calculator):

    implemented_properties = ['energy', 'forces']

    nolabel = True

    default_parameters = {'ff': 'UFF'}

    ob_units = {'kcal/mol': ase.units.kcal/ase.units.mol,
                'kJ/mol': ase.units.kJ/ase.units.mol}

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.outputname = 'openbabelff'

    def initialize(self, atoms):
        obMol = OpenbabelFF.atoms2OBmol(atoms)
        self.ff = openbabel.OBForceField.FindForceField(self.parameters.ff)
        if not self.ff.Setup(obMol):
            for atom in openbabel.OBMolAtomIter(obMol):
                print(atom.GetAtomicNum(), atom.GetFormalCharge())
            raise RuntimeError(
                f'Could not setup force field {self.parameters.ff}')
        self.energy_unit = self.ob_units[self.ff.GetUnit()]

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'numbers' in system_changes:
            self.initialize(self.atoms)

        obMol = OpenbabelFF.atoms2OBmol(self.atoms)
        self.ff.SetCoordinates(obMol)
        self.results['energy'] = self.ff.Energy()*self.energy_unit
        grad = np.zeros((len(atoms), 3))
        for i, atom_i in enumerate(openbabel.OBMolAtomIter(obMol)):
            grad_i = self.ff.GetGradient(atom_i)
            grad[i, :] = (grad_i.GetX(), grad_i.GetY(), grad_i.GetZ())
        # Note, GetGradient() returns the negative gradient, so no sign
        # inversion needed here.
        self.results['forces'] = grad * self.energy_unit/ase.units.Ang

    @staticmethod
    def atoms2OBmol(atoms):
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat('xyz')
        obMol = openbabel.OBMol()

        with io.StringIO() as stream:
            ase.io.write(stream, atoms, format='xyz')
            obConversion.ReadString(obMol, stream.getvalue())
        return obMol


class TwoDCalculator(Calculator):
    """Base class for two dimensional benchmark systems."""

    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        xyzs = self.atoms.get_positions()
        x = xyzs[0, 0]
        dx = np.zeros_like(xyzs)
        dx[0, 0] = 1.0
        y = xyzs[0, 1]
        dy = np.zeros_like(xyzs)
        dy[0, 1] = 1.0

        self.results['energy'] = self.energy(x, y)
        gx, gy = self.gradient(x, y)
        self.results['forces'] = -gx * dx - gy * dy


class CerjanMillerSurface(TwoDCalculator):
    a = 1
    b = 1
    c = 1

    def energy(self, x, y):
        return ((self.a - self.b * y**2) * x**2 * np.exp(-x**2)
                + (self.c/2) * y**2)

    def gradient(self, x, y):
        exp = np.exp(-x**2)
        return (2 * (self.a - self.b * y**2) * exp * x * (1 - x**2),
                - 2 * self.b * y * exp * x**2 + self.c * y)


class AdamsSurface(TwoDCalculator):

    def energy(self, x, y):
        return (2 * x**2 * (4 - x) + y**2 * (4 + y)
                - x*y * (6 - 17 * np.exp(-(x**2 + y**2)/4)))

    def gradient(self, x, y):
        dx1 = (-17/2 * (x**2 - 2) * y * np.exp(-(x**2 + y**2)/4)
               - 6 * x**2 + 16 * x - 6 * y)
        dx2 = (-17/2 * (y**2 - 2) * x * np.exp(-(x**2 + y**2)/4)
               + 3 * y**2 + 8 * y - 6 * x)
        return dx1, dx2


class MuellerBrownSurface(TwoDCalculator):
    min_A = (-0.55822363,  1.44172584)
    min_B = (-0.05001083,  0.4666941)
    min_C = (0.6234994,  0.02803776)

    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    def _v(self, x, y):
        return self.A * np.exp(self.a*(x - self.x0)**2
                               + self.b*(x - self.x0)*(y - self.y0)
                               + self.c*(y - self.y0)**2)

    def energy(self, x, y):
        return np.sum(self._v(x, y))

    def gradient(self, x, y):
        v = self._v(x, y)
        return ((2*self.a*(x - self.x0) + self.b*(y - self.y0)) @ v,
                (self.b*(x - self.x0) + 2*self.c*(y - self.y0)) @ v)


class LEPSPotential(TwoDCalculator):
    """From https://theory.cm.utexas.edu/henkelman/pubs/jonsson98_385.pdf"""
    a = 0.05
    b = 0.30
    c = 0.05
    d_AB = 4.746
    d_BC = 4.746
    d_AC = 3.445
    r0 = 0.742
    alpha = 1.942

    def energy(self, r_AB, r_BC):
        r_AC = r_AB + r_BC
        Q_AB = self.d_AB / 2 * (1.5 * np.exp(-2*self.alpha*(r_AB - self.r0))
                                - np.exp(-self.alpha*(r_AB - self.r0)))
        Q_BC = self.d_BC / 2 * (1.5 * np.exp(-2*self.alpha*(r_BC - self.r0))
                                - np.exp(-self.alpha*(r_BC - self.r0)))
        Q_AC = self.d_AC / 2 * (1.5 * np.exp(-2*self.alpha*(r_AC - self.r0))
                                - np.exp(-self.alpha*(r_AC - self.r0)))
        J_AB = self.d_AB / 4 * (np.exp(-2*self.alpha*(r_AB - self.r0))
                                - 6*np.exp(-self.alpha*(r_AB - self.r0)))
        J_BC = self.d_BC / 4 * (np.exp(-2*self.alpha*(r_BC - self.r0))
                                - 6*np.exp(-self.alpha*(r_BC - self.r0)))
        J_AC = self.d_AC / 4 * (np.exp(-2*self.alpha*(r_AC - self.r0))
                                - 6*np.exp(-self.alpha*(r_AC - self.r0)))
        return (Q_AB / (1 + self.a) + Q_BC / (1 + self.b) + Q_AC / (1 + self.c)
                - np.sqrt(J_AB**2 / (1 + self.a)**2 + J_BC**2 / (1 + self.b)**2
                          + J_AC**2 / (1 + self.c)**2
                          - (J_AB * J_BC) / ((1 + self.a)*(1 + self.b))
                          - (J_BC * J_AC) / ((1 + self.b)*(1 + self.c))
                          - (J_AB * J_AC) / ((1 + self.a)*(1 + self.c)))
                )

    def gradient(self, x, y):
        raise NotImplementedError()
        dx1 = 0.
        dx2 = 0.
        return dx1, dx2


class ThreeDCalculator(Calculator):
    """Base class for three dimensional benchmark systems."""

    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        xyzs = self.atoms.get_positions()
        x = xyzs[0, 0]
        dx = np.zeros_like(xyzs)
        dx[0, 0] = 1.0
        y = xyzs[0, 1]
        dy = np.zeros_like(xyzs)
        dy[0, 1] = 1.0
        z = xyzs[0, 2]
        dz = np.zeros_like(xyzs)
        dz[0, 2] = 1.0

        self.results['energy'] = self.energy(x, y, z)
        gx, gy, gz = self.gradient(x, y, z)
        self.results['forces'] = -gx * dx - gy * dy - gz * dz


class SecondOrderSaddlePointCalculator(ThreeDCalculator):

    def energy(self, x, y, z):
        return x**2 - y**2 + (z**4 - z**2)

    def gradient(self, x, y, z):
        return 2*x, -2*y, (4*z**3 - 2*z)
