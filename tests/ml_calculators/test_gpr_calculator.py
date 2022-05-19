import numpy as np

from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from pesexp.ml_calculators import GPRCalculator
from pesexp.ml_calculators.kernels import RBFKernel


def test_co_potential_curve():
    direction = np.array([1., 2., 3.])
    direction /= np.linalg.norm(direction)
    r_train = np.linspace(0.5, 7, 11)
    energies_train = []
    images_train = []
    for ri in r_train:
        image = Atoms(
            ['C', 'O'],
            positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
        image.set_calculator(EMT())
        energies_train.append(image.get_potential_energy())
        images_train.append(image)

    kernel = RBFKernel(constant=100.0, length_scale=1e-1)
    calc = GPRCalculator(kernel=kernel, C1=1e8, C2=1e8, opt_restarts=2,
                         verbose=1)

    [calc.add_data(im) for im in images_train]
    calc.fit()
    np.testing.assert_allclose(
        energies_train,
        [calc.predict(im)[0] for im in images_train])


def test_prediction_variance():
    direction = np.array([1., 2., 3.])
    direction /= np.linalg.norm(direction)
    r_train = [0.7, 1.7]
    images_train = []
    for ri in r_train:
        image = Atoms(
            ['C', 'O'],
            positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
        image.set_calculator(EMT())
        images_train.append(image)

    r_test = np.linspace(0.5, 1.9, 101)
    images_test = []
    for ri in r_test:
        image = Atoms(
            ['C', 'O'],
            positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
        image.set_calculator(EMT())
        images_test.append(image)

    kernel = RBFKernel(constant=100., length_scale=.1)
    calc = GPRCalculator(kernel=kernel, C1=1E8, C2=1E8, opt_restarts=0)
    [calc.add_data(im) for im in images_train]

    calc.fit()
    prediction_var = [calc.predict_var(im)[0] for im in images_test]
    max_var = np.argmax(prediction_var)
    np.testing.assert_equal(r_test[max_var], 1.2)


def test_log_marginal_likelihood_derivative():

    kernel = RBFKernel(factor=3.1, length_scale=2.1)
    theta0 = np.exp(kernel.theta)
    calc = GPRCalculator(kernel=kernel, C1=1e8, C2=1e8, opt_restarts=0)
    atoms = molecule('C2H6')
    atoms.set_calculator(EMT())

    calc.add_data(atoms)
    calc.fit()

    dt = 1e-5
    L, dL = calc.log_marginal_likelihood()
    dL_num = np.zeros_like(dL)
    for i in range(len(theta0)):
        dti = np.zeros(len(theta0))
        dti[i] = dt
        calc.kernel.theta = np.log(theta0 + dti)
        L_plus = calc.log_marginal_likelihood()[0]
        calc.kernel.theta = np.log(theta0 - dti)
        L_minus = calc.log_marginal_likelihood()[0]
        calc.kernel.theta = np.log(theta0)
        dL_num[i] = (L_plus - L_minus)/(2*dt)

    np.testing.assert_allclose(dL, dL_num)


def test_log_predictive_probability_derivative():

    kernel = RBFKernel(length_scale=2.1)
    theta0 = np.exp(kernel.theta)
    calc = GPRCalculator(kernel=kernel, C1=1e8, C2=1e8, opt_restarts=0)
    atoms = molecule('C2H6')
    atoms.set_calculator(EMT())

    calc.add_data(atoms)
    calc.fit()

    dt = 1e-5
    L, dL = calc.log_predictive_probability()
    dL_num = np.zeros_like(dL)
    for i in range(len(theta0)):
        dti = np.zeros(len(theta0))
        dti[i] = dt
        calc.kernel.theta = np.log(theta0 + dti)
        L_plus = calc.log_predictive_probability()[0]
        calc.kernel.theta = np.log(theta0 - dti)
        L_minus = calc.log_predictive_probability()[0]
        calc.kernel.theta = np.log(theta0)
        dL_num[i] = (L_plus - L_minus)/(2*dt)

    np.testing.assert_allclose(dL, dL_num)
