import pytest
import numpy as np
from scipy.linalg import eigh
from ase.atoms import Atoms
from pesexp.optimizers import RFO
from pesexp.geometry.coordinate_systems import ApproximateNormalCoordinates


@pytest.mark.parametrize("s", [1.0, 0.5, 2.0])
def test_RFO_step_calculations(s):
    opt = RFO(Atoms(), s=s)
    # Based on the third step in a Baker system 01 optimization
    H = np.array(
        [
            [-10.75201976, -6.42166332, 5.57618174],
            [-6.42166332, 8.23167565, 14.23864386],
            [5.57618174, 14.23864386, 283.73471308],
        ]
    )
    omega, V = np.linalg.eigh(H)
    f = np.array([-1.04432943, 1.98562983, 1.97730308])

    opt.H = H
    H_ext = np.zeros((len(f) + 1, len(f) + 1))
    H_ext[: len(f), : len(f)] = H
    H_ext[:-1, -1] = -f
    H_ext[-1, :-1] = -f
    S = s * np.eye(len(f))
    S_ext = np.zeros_like(H_ext)
    S_ext[:-1, :-1] = S
    S_ext[-1, -1] = 1
    # Equation (6) in Baker 1986
    vals, vecs = eigh(H_ext, S_ext)

    for mu in range(0, len(f) + 1):
        np.testing.assert_allclose(
            opt.calc_shift_parameter(
                np.dot(f, V), omega, s * np.ones_like(omega), root=mu
            ),
            vals[mu],
            atol=1e-6,
        )

        opt.mu = mu
        opt.s = s * np.ones_like(omega)
        step = opt.internal_step(f)
        step_ref = vecs[:-1, mu] / vecs[-1, mu]

        np.testing.assert_allclose(step, step_ref, atol=1e-6)


@pytest.mark.parametrize(
    ["s_update", "reference"],
    [("nr_scalar", 90.7264245), ("nr_vector", [1317.383826, 97.993153, 17160.815268])],
)
def test_s_update_methods(s_update, reference):
    atoms = Atoms(
        ["C", "N", "H"],
        [
            [0.00000, 0.00000, 0.00000],
            [1.14838, 0.00000, 0.00000],
            [1.14838, 0.00000, 1.58536],
        ],
    )
    V = np.array(
        [
            [0.0861, 0.0, 0.2461, 0.0923, 0.0, -0.7794, -0.1783, 0.0, 0.5332],
            [-0.2277, 0.0, -0.6137, -0.2168, 0.0, 0.0437, 0.4446, 0.0, 0.57],
            [-0.7058, 0.0, 0.0036, 0.7084, 0.0, 0.0031, -0.0026, 0.0, -0.0067],
        ]
    )
    coord_set = ApproximateNormalCoordinates(atoms)
    coord_set.BTinv = V
    # Based on the third step in a Baker system 01 optimization
    H = np.array(
        [
            [-9.00694082, 10.75415352, 16.01601457],
            [10.75415352, 11.08937814, -30.0838256],
            [16.01601457, -30.0838256, 282.12316583],
        ]
    )
    omega, V = np.linalg.eigh(H)
    f0 = np.array(
        [
            [1.47218388, 0.0, 1.33021951],
            [-0.71917117, -0.0, 2.05629584],
            [-0.7530127, 0.0, -3.38651534],
        ]
    )
    r0 = np.array(
        [
            [7.99898402e-02, 0.0, 2.25904716e-01],
            [1.23202773e00, 0.0, -1.15761055e-01],
            [9.84742430e-01, 0.0, 1.47521634e00],
        ]
    )
    f = np.array(
        [
            [1.74939786, -0.0, 0.09690256],
            [-1.42186529, 0.0, 0.9253995],
            [-0.32753257, 0.0, -1.02230206],
        ]
    )
    r = np.array(
        [
            [1.28050270e-01, 0.0, 3.27413744e-01],
            [1.25749693e00, 0.0, -7.65719671e-02],
            [9.11212798e-01, 0.0, 1.33451822e00],
        ]
    )

    opt = RFO(atoms, coord_set, s_update=s_update)
    opt.H = H
    opt.update_S_matrix(r, f, r0, f0)
    np.testing.assert_allclose(opt.s, reference)


def test_calc_shift_parameter_baker09_1():
    """Previously failed run (in baker system 09)"""
    opt = RFO(Atoms())
    mu = 0
    f_trans = np.array(
        [
            3.68775770e-11,
            1.05375341e-10,
            -1.20748881e-01,
            5.48624614e-01,
            5.92905657e-11,
            5.38204980e-12,
            -7.99705331e-01,
            4.59421296e-10,
            -4.45934288e-01,
            8.73219017e-01,
            -2.51387251e-09,
            8.13179569e-01,
            -1.17214343e00,
            -6.67969207e-10,
            -3.29227128e-01,
            -9.15673770e-10,
            6.10238387e-01,
            -2.16298396e-09,
            -5.62462279e00,
            2.76142805e00,
            -2.53603641e-10,
            -7.57262198e-10,
            5.22090598e-11,
            2.16793822e-01,
            9.75830491e-11,
            5.40589951e00,
            -1.29761317e-10,
            -1.21735665e00,
            6.59374740e-11,
            -2.79901419e-10,
            2.04419038e00,
            4.13824959e-11,
            1.73426477e00,
            3.93522512e-11,
            1.35274923e00,
            -2.53813335e-10,
            -2.54335590e00,
            -2.57989576e00,
            6.15056456e-11,
            -3.23049102e-02,
            -3.45327705e-11,
        ]
    )
    omega = np.array(
        [
            -1.38915854e-01,
            6.41664043e-01,
            1.31719956e00,
            1.97945483e00,
            2.90270581e00,
            3.12529275e00,
            3.20574159e00,
            3.57539033e00,
            3.88147744e00,
            4.66006282e00,
            4.80687342e00,
            5.76611164e00,
            6.81946006e00,
            7.15672276e00,
            8.06881342e00,
            8.44245196e00,
            8.97604510e00,
            1.03322897e01,
            1.09667840e01,
            1.37992786e01,
            1.45515217e01,
            1.57474795e01,
            1.64242099e01,
            1.83299578e01,
            1.92985090e01,
            2.47407212e01,
            2.78104878e01,
            2.90883245e01,
            3.11896977e01,
            3.85245278e01,
            4.54799314e01,
            5.36886388e01,
            7.00415987e01,
            7.37464558e01,
            7.58371867e01,
            9.79243588e01,
            1.02943997e02,
            1.03495540e02,
            1.07533774e02,
            1.25473638e02,
            1.41082404e02,
        ]
    )
    shift = opt.calc_shift_parameter(f_trans, omega, 1.0, mu)
    ref = -4.28947
    assert abs(shift - ref) < 1e-5


def test_calc_shift_parameter_baker09_2():
    """Previously failed run (in baker system 09)"""
    opt = RFO(Atoms())
    mu = 0
    f_trans = np.array(
        [
            -4.98587374e-09,
            6.83299550e-09,
            7.25540051e-02,
            -1.89599173e-01,
            -5.61167802e-10,
            -2.86372902e-09,
            -7.47304780e-02,
            -1.45636319e-09,
            3.52321996e-02,
            5.02186719e-09,
            4.50223568e-02,
            -1.12921385e-01,
            -1.69593634e-01,
            -1.63503922e-09,
            -1.02615733e-01,
            -9.29002497e-02,
            -2.76625404e-09,
            9.20788230e-02,
            -2.61009260e-09,
            1.39931439e-02,
            9.81943066e-11,
            8.32911149e-10,
            6.79629396e-10,
            -1.72235021e-01,
            -1.60213407e-09,
            5.72039240e-01,
            3.02099420e-09,
            -1.32015807e-01,
            -2.49875927e-09,
            -5.49100653e-09,
            3.88177504e-01,
            7.27099999e-10,
            -7.24221033e-01,
            4.33696182e-09,
            7.00488683e-02,
            -7.62774602e-09,
            9.43431236e-01,
            -3.96126164e-01,
            1.27254789e-09,
            -3.88577751e-01,
            8.79305157e-09,
        ]
    )
    omega = np.array(
        [
            -1.38915854e-01,
            6.41664043e-01,
            1.36585225e00,
            1.90058734e00,
            2.90270581e00,
            3.12529275e00,
            3.32022109e00,
            3.57539033e00,
            3.91895067e00,
            4.80687342e00,
            5.01908700e00,
            5.76908283e00,
            6.06934852e00,
            7.15672276e00,
            7.82720138e00,
            8.41041299e00,
            8.44245196e00,
            9.44290024e00,
            1.03322897e01,
            1.38927234e01,
            1.45515217e01,
            1.57474795e01,
            1.64242099e01,
            1.83551908e01,
            1.92985090e01,
            2.30249538e01,
            2.78104878e01,
            2.91819558e01,
            3.11896977e01,
            3.85245278e01,
            4.59225138e01,
            5.36886388e01,
            6.97777838e01,
            7.37464558e01,
            7.57251962e01,
            9.79243588e01,
            1.02584034e02,
            1.03632558e02,
            1.07533774e02,
            1.25240083e02,
            1.41082404e02,
        ]
    )
    shift = opt.calc_shift_parameter(f_trans, omega, 1.0, mu)
    ref = -0.138916
    assert abs(shift - ref) < 1e-5


def test_calc_shift_parameter_baker09_3():
    """Previously failed run (in baker system 09)"""
    opt = RFO(Atoms())
    mu = 0
    f_trans = np.array(
        [
            4.22618270e-10,
            2.20203350e-10,
            3.06436497e-02,
            2.44336088e-01,
            4.48424234e-11,
            1.01525981e-10,
            3.01598995e-01,
            6.78262504e-12,
            -1.44819985e-01,
            2.46897300e-01,
            -4.96595344e-11,
            1.84284387e-01,
            3.49770126e-01,
            9.48845727e-12,
            -8.64736938e-02,
            -4.56742658e-11,
            3.47341871e-01,
            2.68835108e-11,
            -1.15097027e00,
            -5.17647956e-01,
            -1.19544820e-11,
            -2.85069975e-12,
            -4.22560361e-12,
            -4.98101484e-02,
            -6.02385417e-12,
            7.45664523e-01,
            -1.25413482e-11,
            -1.48132639e-01,
            1.66263431e-12,
            -4.01748416e-12,
            -1.78103202e-01,
            3.01432483e-12,
            1.45985837e-01,
            -6.70007310e-12,
            -1.15311742e-01,
            1.34600099e-11,
            -9.64986130e-02,
            -1.79596037e-01,
            2.93353211e-12,
            9.96367936e-03,
            -8.98757330e-12,
        ]
    )
    omega = np.array(
        [
            -2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
    shift = opt.calc_shift_parameter(f_trans, omega, 1.0, mu)
    ref = -2.0
    assert abs(shift - ref) < 1e-5


def test_calc_shift_parameter_baker10_1():
    """Previously failed run (in baker system 10)"""
    opt = RFO(Atoms())
    mu = 1
    f_trans = np.array([-1.76696862e-13])
    omega = np.array([0.72761467])
    shift = opt.calc_shift_parameter(f_trans, omega, 1.0, mu)
    ref = 0.72761
    assert abs(shift - ref) < 1e-5


def test_calc_shift_parameter_baker10_2():
    """Previously failed run (in baker system 10)"""
    opt = RFO(Atoms())
    mu = 1
    f_trans = np.array([1.03235612e-08])
    omega = np.array([0.72736632])
    shift = opt.calc_shift_parameter(f_trans, omega, 1.0, mu)
    ref = 0.72737
    assert abs(shift - ref) < 1e-5
