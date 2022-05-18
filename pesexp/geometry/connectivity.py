import numpy as np
import ase.data
import itertools
from molSimplify.optimize.primitives import (Distance, Angle, LinearAngle,
                                             Dihedral, Improper)


def find_connectivity(atoms, threshold=1.25, connect_fragments=True):
    """
    Follows the algorithm outlined in Section II A of
    Billeter et al. Phys. Chem. Chem. Phys., 2000, 2, 2177-2186
    """
    N = len(atoms)
    xyzs = atoms.get_positions()
    types = atoms.get_atomic_numbers()

    bonds = []
    # Array used to save the squared distances. The diagonal is set to
    # np.inf to avoid bonding atoms to themselves. TODO: There should be
    # a "nicer" way of avoiding this problem.
    r2 = np.zeros((N, N))
    np.fill_diagonal(r2, np.inf)
    neighbors_per_atom = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i+1, N):
            r2[i, j] = r2[j, i] = np.sum((xyzs[i, :] - xyzs[j, :])**2)
            # "A connection is made whenever the square of the interatomic
            # distance is less than 1.25 times the square of the sum of the
            # corresponding covalent atomic radii."
            if (r2[i, j] < threshold * (
                    ase.data.covalent_radii[types[i]]
                    + ase.data.covalent_radii[types[j]])**2):
                bonds.append((i, j))
                neighbors_per_atom[i].append(j)
                neighbors_per_atom[j].append(i)

    if connect_fragments:
        # "This procedure may lead to insulated fragments which are avoided
        # by inserting the missing connections from the shortest-distance
        # branched path."
        def depth_first_search(fragment, i, visited):
            visited[i] = True
            fragment.append(i)
            for j in neighbors_per_atom[i]:
                if not visited[j]:
                    fragment = depth_first_search(fragment, j, visited)
            return fragment

        visited = [False] * N
        fragments = []
        for i in range(N):
            if not visited[i]:
                fragment = []
                fragments.append(depth_first_search(fragment, i, visited))

        # If there are more than 1 fragment connect shortest distance atoms
        while len(fragments) > 1:
            # Merge first fragment with nearest fragment by adding a bond
            r_min = np.inf
            bond_to_add = ()
            for frag_ind, other_frag in enumerate(fragments[1:]):
                for i in fragments[0]:
                    for j in other_frag:
                        if r2[i, j] < r_min:
                            r_min = r2[i, j]
                            # Plus one because of iterating fragments[1:]
                            fragment_to_merge = frag_ind + 1
                            bond_to_add = tuple(sorted([i, j]))
            bonds.append(bond_to_add)
            fragments[0].extend(fragments.pop(fragment_to_merge))
    return bonds


def cos_angle(r1, r2):
    """
    Helper function that calculates the cosine of the angle between
    two vectors
    """
    return np.dot(r1, r2)/(np.linalg.norm(r1)*np.linalg.norm(r2))


def find_planars_billeter(a, neighbors, xyzs, planar_threshold):
    # "lies in the centre of a planar system" characterized by
    # a central atom 'a' and the plane spanned by three neighboring
    # atoms (ai, aj, ak)
    for (ai, aj, ak) in itertools.combinations(neighbors[a], 3):
        r_ai = xyzs[ai, :] - xyzs[a, :]
        r_aj = xyzs[aj, :] - xyzs[a, :]
        r_ak = xyzs[ak, :] - xyzs[a, :]
        n1 = np.cross(r_ai, r_aj)
        if np.sum(n1**2) > 0:  # Linear cases lead to division by zero
            n1 /= np.linalg.norm(n1)
        n2 = np.cross(r_aj, r_ak)
        if np.sum(n2**2) > 0:  # Linear cases lead to division by zero
            n2 /= np.linalg.norm(n2)
        n3 = np.cross(r_ak, r_ai)
        if np.sum(n3**2) > 0:  # Linear cases lead to division by zero
            n3 /= np.linalg.norm(n3)
        # Not sure if actually all three possible values need to be
        # checked. For an actual planar case the angles are dependent
        # since they add to 360 degrees.
        if (np.abs(n1.dot(n2)) > planar_threshold
                or np.abs(n2.dot(n3)) > planar_threshold
                or np.abs(n3.dot(n1)) > planar_threshold):
            # Try to find an improper (b, a, c, d)
            # such neither the angle t1 between (b, a, c)
            # nor t2 between (a, c, d) is close to linear
            for (b, c, d) in itertools.permutations([ai, aj, ak], 3):
                r_ab = xyzs[b, :] - xyzs[a, :]
                r_ac = xyzs[c, :] - xyzs[a, :]
                r_cd = xyzs[d, :] - xyzs[c, :]
                cos_t1 = cos_angle(r_ab, r_ac)
                cos_t2 = cos_angle(r_ac, r_cd)
                if np.abs(cos_t1) < 0.95 and np.abs(cos_t2) < 0.95:
                    # Remove bend (ai, a, aj)
                    # TODO: Better heuristic of which bend to remove
                    # Return after a first improper has been added
                    return [(ai, a, aj)], [(a, b, c, d)]
    # If no planar structure is found, return the empty lists
    return [], []


def find_planars_molsimplify(a, neighbors, xyzs, planar_threshold):
    max_area = 0.
    for (ai, aj, ak) in itertools.combinations(neighbors[a], 3):
        r_ai = xyzs[ai, :] - xyzs[a, :]
        n_ai = r_ai / np.linalg.norm(r_ai)
        r_aj = xyzs[aj, :] - xyzs[a, :]
        n_aj = r_aj / np.linalg.norm(r_aj)
        r_ak = xyzs[ak, :] - xyzs[a, :]
        n_ak = r_ak / np.linalg.norm(r_ak)
        n1 = np.cross(r_ai, r_aj)
        if np.sum(n1**2) > 0:  # Linear cases lead to division by zero
            n1 /= np.linalg.norm(n1)
        n2 = np.cross(r_aj, r_ak)
        if np.sum(n2**2) > 0:  # Linear cases lead to division by zero
            n2 /= np.linalg.norm(n2)
        n3 = np.cross(r_ak, r_ai)
        if np.sum(n3**2) > 0:  # Linear cases lead to division by zero
            n3 /= np.linalg.norm(n3)
        if not (np.abs(n1.dot(n2)) > planar_threshold
                or np.abs(n2.dot(n3)) > planar_threshold
                or np.abs(n3.dot(n1)) > planar_threshold):
            # If a structure is identified as non-planar return empty lists
            return [], []
        area = 0.5 * (np.linalg.norm(np.cross(n_ai, n_aj))
                      + np.linalg.norm(np.cross(n_aj, n_ak))
                      + np.linalg.norm(np.cross(n_ak, n_ai)))
        if area > max_area:
            max_area = area
            planar = (a, ai, aj, ak)
            # TODO better heuristic of which bend to remove
            bend = (ai, a, aj)
    return [bend], [planar]


def find_primitives(xyzs, bonds, linear_flag=True, linear_threshold=5.,
                    planar_threshold=0.95, planar_method='molsimplify'):
    """
    Finds primitive internals given a reference geometry and connectivity list.
    Follows the algorithm outlined in Section II A of
    Billeter et al. Phys. Chem. Chem. Phys., 2000, 2, 2177-2186.
    Throughout the code literal citations of this paper appear in quotes "".

    Parameters
    ----------
    xyzs : np.ndarray
        Cartesian coordinates.
    bonds : list of integer pairs
        List of pairs of indices for the atoms that are considered bonded.
    linear_flag : bool, optional
        Use linear bends for bends exceeding the linear_threshold angle.
    linear_threshold : float, optional
        Threshold angle in degrees to determine when to replace
        close to linear bends with a coplanar and perpendicular
        bend coordinate (default 5 degrees).
    planar_threshold : float, optional
        Threshold value used to detect planar structures (default 0.95).
    planar_method : str, optional
        Method use to detect planar structures (default 'billeter').

    Returns
    -------
    tuple of lists
        bends, linear_bends, torsions, planars
    """

    neighbors = [[] for _ in range(len(xyzs))]
    for b in bonds:
        neighbors[b[0]].append(b[1])
        neighbors[b[1]].append(b[0])
    neighbors = list(map(sorted, neighbors))
    bends = []
    linear_bends = []
    torsions = []
    planars = []
    # Transform the threshold angle to cos(angle) to avoid repeated
    # error prone calls to arccos.
    cos_lin_thresh = np.abs(np.cos(linear_threshold*np.pi/180.))

    for a in range(len(xyzs)):
        for i, ai in enumerate(neighbors[a]):
            for aj in neighbors[a][i+1:]:
                r_ai = xyzs[ai, :] - xyzs[a, :]
                r_aj = xyzs[aj, :] - xyzs[a, :]
                cos_theta = np.abs(cos_angle(r_ai, r_aj))
                # "Almost linear bends are dropped." Implemented by comparsion
                # to a threshold angle.
                if cos_theta <= cos_lin_thresh:
                    bends.append((ai, a, aj))
                    for ak in neighbors[ai]:
                        if (ak != a and ak != aj
                                and (aj, a, ai, ak) not in torsions
                                and (ak, ai, a, aj) not in torsions):
                            # Check if (ak, ai, a) is linear
                            r_ik = xyzs[ak, :] - xyzs[ai, :]
                            cos_phi = cos_angle(r_ik, r_ai)
                            if np.abs(cos_phi) < 0.99:
                                # Convention is to use accending index order
                                # for the central two atoms (here a < ai).
                                torsions.append((aj, a, ai, ak))
                    for ak in neighbors[aj]:
                        if (ak != a and ak != ai
                                and (ak, aj, a, ai) not in torsions
                                and (ai, a, aj, ak) not in torsions):
                            # Check if (a, aj, ak) is linear
                            r_jk = xyzs[ak, :] - xyzs[aj, :]
                            cos_phi = cos_angle(r_jk, r_aj)
                            if np.abs(cos_phi) < 0.99:
                                torsions.append((ai, a, aj, ak))
                elif linear_flag:
                    # "If the centre of such an angle is connected to exactly
                    # two atoms, the dropped angle is replaced by..." a linear
                    # bend.
                    if len(neighbors[a]) == 2:
                        # Add linear bend
                        linear_bends.append((ai, a, aj))
                        # Do not add torsions on linear bends, instead try
                        # using the ends of a linear chain as central bond.

                        def find_non_linear_neighbors(o, p):
                            # Helper function to find the ends of a linear
                            # chain. o is the starting index and the search
                            # is towards the direction of index p.
                            if len(neighbors[p]) == 2:
                                # Find the neighbor other than o
                                q = neighbors[p][0]
                                if q == o:
                                    q = neighbors[p][1]
                                # Check if the angle (o, p, q) is linear
                                r_po = xyzs[o, :] - xyzs[p, :]
                                r_pq = xyzs[q, :] - xyzs[p, :]
                                cos_t = np.abs(cos_angle(r_po, r_pq))
                                if cos_t < 0.99:
                                    # Return p as one end of a chain and its
                                    # single neighbor q
                                    return p, [q]
                                # Else: recursively call this function:
                                return find_non_linear_neighbors(
                                    p, q)
                            # Return p as end of chain and all neighbors
                            # that are not p and where the angle (o, p, q)
                            # is not linear:
                            neighbors_p = []
                            for q in neighbors[p]:
                                if q != o:
                                    r_po = xyzs[o, :] - xyzs[p, :]
                                    r_pq = xyzs[q, :] - xyzs[p, :]
                                    cos_t = np.abs(cos_angle(r_po, r_pq))
                                    if cos_t < 0.99:
                                        neighbors_p.append(q)
                            return p, neighbors_p

                        # Use find_non_linear_neightbors to append all torsions
                        # on the linear chain.
                        p, neighbors_p = find_non_linear_neighbors(a, ai)
                        q, neighbors_q = find_non_linear_neighbors(a, aj)
                        for o in neighbors_p:
                            for r in neighbors_q:
                                if ((o, p, q, r) not in torsions
                                        and (r, q, p, o) not in torsions):
                                    torsions.append((o, p, q, r))
        # "If an atom is connected to more than two atoms" and the threshold
        # is small enough to be exceeded.
        if len(neighbors[a]) > 2 and planar_threshold < 1.0:
            if planar_method.lower() == 'billeter':
                bends_to_remove, planars_to_append = find_planars_billeter(
                    a, neighbors, xyzs, planar_threshold)
            elif planar_method.lower() == 'molsimplify':
                bends_to_remove, planars_to_append = find_planars_molsimplify(
                    a, neighbors, xyzs, planar_threshold)
            else:
                raise NotImplementedError(
                    f'Unknown planar_method {planar_method}')
            for b in bends_to_remove:
                bends.remove(b)
            planars.extend(planars_to_append)
    return bends, linear_bends, torsions, planars


def get_primitives(atoms, threshold=1.25, connect_fragments=True,
                   linear_flag=True, linear_threshold=5.,
                   planar_threshold=0.95, planar_method='molsimplify'):

    bonds = find_connectivity(atoms, threshold=threshold,
                              connect_fragments=connect_fragments)
    xyzs = atoms.get_positions()
    bends, linear_bends, torsions, planars = find_primitives(
        xyzs, bonds, linear_flag=linear_flag,
        linear_threshold=linear_threshold, planar_threshold=planar_threshold,
        planar_method=planar_method)

    primitives = ([Distance(*b) for b in bonds]
                  + [Angle(*a) for a in bends]
                  + [LinearAngle(*a, axis=axis)
                     for a in linear_bends for axis in (0, 1)]
                  + [Dihedral(*d) for d in torsions]
                  + [Improper(*p) for p in planars])
    return primitives
