import numpy as np
from pesexp.utils.io import (read_molecule, read_terachem_frequencies,
                             read_terachem_input)
from pesexp.calculators import TeraChem
from pkg_resources import resource_filename, Requirement


def test_read_molecule():
    in_file = resource_filename(
        Requirement.parse('molSimplify'),
        'tests/optimize/inputs/acac/acac.inp')
    atoms, xyz_file = read_molecule(in_file)
    assert xyz_file.name == 'fe_oct_2_acac_3_s_5_conf_1.xyz'
    assert atoms.get_initial_magnetic_moments().sum() == 4.
    assert atoms.get_initial_charges().sum() == -1.


def test_read_terachem_input():
    in_file = resource_filename(
        Requirement.parse('molSimplify'),
        'tests/optimize/inputs/acac/acac.inp')
    atoms = read_terachem_input(in_file)
    assert atoms.get_initial_magnetic_moments().sum() == 4.
    assert atoms.get_initial_charges().sum() == -1.
    calc = atoms.calc
    assert type(calc) is TeraChem

    params = {'timings': 'yes',
              'maxit': '500',
              'scrdir': './scr',
              'method': 'ub3lyp',
              'basis': 'lacvps_ecp',
              'spinmult': '5',
              'charge': '-1',
              'gpus': '1',
              'scf': 'diis+a',
              'levelshift': 'yes',
              'levelshiftvala': '0.25',
              'levelshiftvalb': '0.25'}
    for key, val in params.items():
        assert calc.parameters[key] == val


def test_read_terachem_frequencies():
    in_file = resource_filename(
        Requirement.parse('molSimplify'),
        'tests/optimize/inputs/Frequencies.dat')
    freq, modes = read_terachem_frequencies(in_file)

    assert len(freq) == 66
    # Test is sorted
    assert np.all(freq[:-1] <= freq[1:])
    # Test actual values
    np.testing.assert_allclose(freq[:4], [-814.6149685777, -23.6415668389,
                                          60.2874900162, 71.6998539466])
    np.testing.assert_allclose(freq[-4:], [3542.3649619929, 3543.6785363150,
                                           3544.3846928982, 3545.9819577459])

    # Test modes
    np.testing.assert_allclose(modes[:4, 0], [-0.0111435207, 0.0184712098,
                                              0.0094254281, -0.0104267842])

    np.testing.assert_allclose(modes[:4, -1], [0.0000243849, 0.0000085852,
                                               -0.0000221754, -0.0004790454])
