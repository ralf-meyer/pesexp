import pytest
import numpy as np
import geometric.internal
import ase.atoms
import ase.io
from utils import g2_molecules
from pesexp.geometry.connectivity import (find_connectivity,
                                          find_primitives,
                                          get_primitives)


@pytest.mark.parametrize('name', g2_molecules.keys())
def test_connectivity(name):
    # For some reason the current verion of geomeTRIC uses a covalent radius
    # of zero for Na. Therefore, Na containing molecules have to be skipped.
    if 'Na' in name:
        return
    atoms = g2_molecules[name]['atoms']
    mol = g2_molecules[name]['mol']
    mol.build_topology()
    # geomeTRIC uses a threshold of 1.2 on the unsquared distances.
    # This correspondes to using 1.2^2 in the Billeter et al. alogrithm.
    bonds = find_connectivity(atoms, threshold=1.2**2, connect_fragments=False)

    bonds_ref = list(mol.topology.edges())
    assert bonds == bonds_ref


@pytest.mark.parametrize('name', g2_molecules.keys())
def test_find_primitives(name):
    atoms = g2_molecules[name]['atoms']
    mol = g2_molecules[name]['mol']
    mol.build_topology()
    # geomeTRIC uses a threshold of 1.2 on the unsquared distances.
    # This correspondes to using 1.2^2 in the Billeter et al. alogrithm.
    bonds = find_connectivity(atoms, threshold=1.2**2, connect_fragments=True)
    # geoemTRIC uses a threshold of cos(theta) = 0.95 for linear angles.
    linear_threshold = np.arccos(0.95) * 180 / np.pi
    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds, linear_threshold=linear_threshold,
        planar_threshold=0.95, planar_method='billeter')

    coords_ref = geometric.internal.PrimitiveInternalCoordinates(
        mol, connect=True)
    # Compare bonds
    bonds_ref = [(ic.a, ic.b) for ic in coords_ref.Internals
                 if isinstance(ic, geometric.internal.Distance)]
    assert bonds == bonds_ref
    # Compare dihedrals
    torsions_ref = [(ic.a, ic.b, ic.c, ic.d) for ic in coords_ref.Internals
                    if isinstance(ic, geometric.internal.Dihedral)]
    assert len(torsions) == len(torsions_ref)
    for t in torsions:
        assert t in torsions_ref or t[::-1] in torsions_ref
    # Compare linear bends. Note that geomeTRIC does not use the
    # "connected to exactly two atoms" rule by Billeter et al for linear bends
    if name != 'ClF3':  # Central atom bound to three neighbors
        linear_bends_ref = [(ic.a, ic.b, ic.c) for ic in coords_ref.Internals
                            if isinstance(ic, geometric.internal.LinearAngle)]
        # Every linear bend appears twice in the reference
        assert linear_bends == linear_bends_ref[::2]
    # Compare planars. For 'C3H4_C2v the methods do not agree on the ordering
    # and for 'methylenecyclopropane' geomeTRIC does not find the second OOP.
    # Second list of exclusions is for a bug / implementation detail how
    # non-planar systems are recognized in geomeTRIC (dot product of the normal
    # vectors of angles a-i-j and i-j-k instead of a-i-j and a-j-k)
    if (name not in ['methylenecyclopropane', 'C3H4_C2v'] and
            name not in ['C3H9C', 'ClF3', 'CH2NHCH2', 'CH2OCH2', 'CH3CONH2',
                         'C3H7', 'C5H8']):
        planars_ref = [(ic.a, ic.b, ic.c, ic.d) for ic in coords_ref.Internals
                       if isinstance(ic, geometric.internal.OutOfPlane)]
        assert planars == planars_ref
        # Compare angles.
        bends_ref = [(ic.a, ic.b, ic.c) for ic in coords_ref.Internals
                     if isinstance(ic, geometric.internal.Angle)]
        assert sorted(bends) == sorted(bends_ref)


def test_find_primitives_on_pentatetraene():
    """
    Test proper detection of linear chains on pentatetraene:
    H(7)                            H(5)
     \\                            //
      C(4) - C(3) - C(0) - C(1) - C(2)
     //                            \\
    H(8)                            H(6)
    Second methyl isocyanide:
         H(4)
          \\
    H(3) - C(2) - N(1) - C(0)
          //
         H(5)
    """
    ri = 1.4
    atoms = ase.atoms.Atoms(
        ['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
        positions=np.array([[0., 0., 0.],
                            [ri, 0., 0.],
                            [2*ri, 0., 0.],
                            [-ri, 0., 0.],
                            [-2*ri, 0., 0.],
                            [2.5*ri, 0.5*ri, 0.],
                            [2.5*ri, -0.5*ri, 0.],
                            [-2.5*ri, 0.5*ri, 0.],
                            [-2.5*ri, -0.5*ri, 0.]]))
    bonds = find_connectivity(atoms)
    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds)
    assert bends == [(1, 2, 6), (5, 2, 6), (3, 4, 8), (7, 4, 8)]
    assert linear_bends == [(1, 0, 3), (0, 1, 2), (0, 3, 4)]
    assert torsions == [(5, 2, 4, 7), (5, 2, 4, 8), (6, 2, 4, 7), (6, 2, 4, 8)]
    assert planars == [(2, 1, 5, 6), (4, 3, 7, 8)]


def test_find_primitives_on_methyl_isocyanide():
    """
    Test proper detection of linear chains on methyl isocyanide:
                    H(4)
                   //
    C(0) - N(1) - C(2) - H(3)
                   \\
                    H(5)
    """
    ri = 1.4
    atoms = ase.atoms.Atoms(
        ['C', 'N', 'C', 'H', 'H', 'H'],
        positions=np.array([[0., 0., 0.],
                            [ri, 0., 0.],
                            [2*ri, 0., 0.],
                            [2.5*ri, -0.5*ri, 0.],
                            [2.5*ri, 0.5*ri, 0.5*ri],
                            [2.5*ri, 0.5*ri, -0.5*ri]]))
    bonds = find_connectivity(atoms)
    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds)
    assert bends == [(1, 2, 3), (1, 2, 4), (1, 2, 5),
                     (3, 2, 4), (3, 2, 5), (4, 2, 5)]
    assert linear_bends == [(0, 1, 2)]
    assert torsions == []
    assert planars == []


def test_find_primitives_on_isocyanic_acid():
    """
    Test proper detection of linear chains on isocyanic acid:
                    H(3)
                   //
    O(0) - C(1) - N(2)
    """
    ri = 1.4
    atoms = ase.atoms.Atoms(
        ['O', 'C', 'N', 'H'],
        positions=np.array([[0., 0., 0.],
                            [ri, 0., 0.],
                            [2*ri, 0., 0.],
                            [2.5*ri, 0.5*ri, 0.]]))
    bonds = find_connectivity(atoms)
    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds)
    assert bends == [(1, 2, 3)]
    assert linear_bends == [(0, 1, 2)]
    assert torsions == []
    assert planars == []


def test_find_primitives_on_simple_octahedron():
    ri = 2.8
    atoms = ase.atoms.Atoms(
        ['Fe'] + ['Cl']*6,
        positions=np.array([[0., 0., 0.],
                            [ri, 0., 0.],
                            [0., ri, 0.],
                            [-ri, 0., 0.],
                            [0., -ri, 0.],
                            [0., 0., ri],
                            [0., 0., -ri]]))

    bonds = find_connectivity(atoms)
    assert bonds == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds, planar_method='billeter')
    # (1, 0, 2) is removed for a planar
    assert bends == [(1, 0, 4), (1, 0, 5), (1, 0, 6), (2, 0, 3), (2, 0, 5),
                     (2, 0, 6), (3, 0, 4), (3, 0, 5), (3, 0, 6), (4, 0, 5),
                     (4, 0, 6)]
    assert linear_bends == []
    assert torsions == []
    assert planars == [(0, 1, 2, 3)]

    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds, planar_method='molsimplify')

    assert bends == [(1, 0, 2), (1, 0, 4), (1, 0, 5), (1, 0, 6), (2, 0, 3),
                     (2, 0, 5), (2, 0, 6), (3, 0, 4), (3, 0, 5), (3, 0, 6),
                     (4, 0, 5), (4, 0, 6)]
    assert linear_bends == []
    assert torsions == []
    assert planars == []


def test_planar_methods_on_Fe_CO_6(resource_path_root):
    atoms = ase.io.read(
        resource_path_root / 'homoleptic_octahedrals/Co_II_co.xyz')
    bonds = find_connectivity(atoms)
    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds, planar_method='billeter')

    assert bends == [(1, 0, 7), (1, 0, 9), (1, 0, 11),
                     (3, 0, 5), (3, 0, 9), (3, 0, 11), (5, 0, 7),
                     (5, 0, 9), (5, 0, 11), (7, 0, 9), (7, 0, 11)]
    assert linear_bends == [(0, 1, 2), (0, 3, 4), (0, 5, 6),
                            (0, 7, 8), (0, 9, 10), (0, 11, 12)]
    assert torsions == []
    assert planars == [(0, 1, 3, 5)]

    bends, linear_bends, torsions, planars = find_primitives(
        atoms.get_positions(), bonds, planar_method='molsimplify')

    assert bends == [(1, 0, 3), (1, 0, 7), (1, 0, 9), (1, 0, 11),
                     (3, 0, 5), (3, 0, 9), (3, 0, 11), (5, 0, 7),
                     (5, 0, 9), (5, 0, 11), (7, 0, 9), (7, 0, 11)]
    assert linear_bends == [(0, 1, 2), (0, 3, 4), (0, 5, 6),
                            (0, 7, 8), (0, 9, 10), (0, 11, 12)]
    assert torsions == []
    assert planars == []

    with pytest.raises(NotImplementedError):
        find_primitives(atoms.get_positions(), bonds, planar_method='foo')


@pytest.mark.parametrize('ligand', ['co', 'misc', 'water', 'pyr', 'furan'])
def test_find_primitives_on_homoleptic_octahedrals(resource_path_root, ligand):
    """Inspired by Janet et al. Inorg. Chem. 2019, 58, 10592-10606"""
    xyz_file = (
        resource_path_root / f'homoleptic_octahedrals/Co_II_{ligand}.xyz')
    atoms = ase.io.read(xyz_file)
    mol = geometric.molecule.Molecule(xyz_file)

    # False is necessary here because geometric (erroneously) connects atoms
    # 50 and 53 (a hydrogen and a carbon atom) in the furan example.
    coords_ref = geometric.internal.PrimitiveInternalCoordinates(
        mol, connect=False)

    # geomeTRIC uses a threshold of 1.2 on the unsquared distances.
    # This correspondes to using 1.2^2 in the Billeter et al. alogrithm.
    bonds = find_connectivity(atoms, threshold=1.2**2, connect_fragments=True)
    # Compare bonds
    bonds_ref = [(ic.a, ic.b) for ic in coords_ref.Internals
                 if isinstance(ic, geometric.internal.Distance)]

    assert bonds == bonds_ref

    bends, linear_bends, dihedrals, planars = find_primitives(
        atoms.get_positions(), bonds, planar_method='molsimplify')
    linear_bends_ref = [(ic.a, ic.b, ic.c) for ic in coords_ref.Internals
                        if isinstance(ic, geometric.internal.LinearAngle)]
    # Geometric adds 6 linear bends on the octahedral core. The factor 2
    # is to account for the two possible axes of a linear bend.
    assert 2*len(linear_bends) == len(linear_bends_ref) - 6

    planars_ref = [(ic.a, ic.b, ic.c, ic.d) for ic in coords_ref.Internals
                   if isinstance(ic, geometric.internal.OutOfPlane)]
    # Geometric add 12 planar bends on the octahedral core.
    assert len(planars) == len(planars_ref) - 12

    dihedrals_ref = []
    for ic in coords_ref.Internals:
        if isinstance(ic, geometric.internal.Dihedral):
            # Remove the dihedrals that appear because of the additional
            # linear bends
            skip = False
            for lin in linear_bends_ref:
                if lin not in linear_bends:
                    if ((ic.b, ic.c) == (lin[0], lin[2])
                            or (ic.c, ic.b) == (lin[0], lin[2])):
                        skip = True
                        break
            if skip:
                continue
            if ligand == 'misc':
                # Current version of geometric also adds dihedrals stretching
                # over the metal center. Skip those!
                if (ic.b, ic.c) in [(3, 15), (9, 21), (27, 33)]:
                    continue
            dihedrals_ref.append((ic.a, ic.b, ic.c, ic.d))
    assert sorted(dihedrals) == sorted(dihedrals_ref)


def test_connectivity_furan_6_failure(resource_path_root):
    """Tests the get primitives function on a calculation that has failed
    during production runs."""

    atoms = ase.io.read(
        resource_path_root / 'previous_failures'
        / 'co_oct_2_furan_6_s_2.xyz')
    prims = get_primitives(atoms)
    # Check if the set of primitives is complete:
    xyzs = atoms.get_positions()
    B = np.zeros((len(prims), xyzs.size))
    for i, prim in enumerate(prims):
        B[i] = prim.derivative(xyzs)
    P = np.linalg.pinv(B) @ B
    _, s, _ = np.linalg.svd(P)
    # Count how many degrees of freedom are lost by transforming to the
    # internal space and back again (should be 6) characterized by close to
    # zeros singular values
    lost_df = np.sum(s < 1e-10)
    assert lost_df == 6
