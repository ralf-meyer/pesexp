import numpy as np
from ase.atoms import Atoms
from pesexp.optimizers import RFO


def test_calc_shift_parameter_fail1():
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
    shift = opt.calc_shift_parameter(f_trans, omega, mu)
    ref = -4.28947
    assert abs(shift - ref) < 1e-5


def test_calc_shift_parameter_fail2():
    """Previously failed run (in baker system 10)"""
    opt = RFO(Atoms())
    mu = 1
    f_trans = np.array([-1.76696862e-13])
    omega = np.array([0.72761467])
    shift = opt.calc_shift_parameter(f_trans, omega, mu)
    ref = 0.72761
    assert abs(shift - ref) < 1e-5


def test_calc_shift_parameter_fail3():
    """Previously failed run (in baker system 10)"""
    opt = RFO(Atoms())
    mu = 1
    f_trans = np.array([1.03235612e-08])
    omega = np.array([0.72736632])
    shift = opt.calc_shift_parameter(f_trans, omega, mu)
    ref = 0.72737
    assert abs(shift - ref) < 1e-5
