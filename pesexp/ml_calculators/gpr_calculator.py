from pesexp.ml_calculators.ml_calculator import MLCalculator
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize


class GPRCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, kernel=None, verbose=0,
                 opt_fun='marginal_likelihood', opt_method='L-BFGS-B',
                 opt_restarts=0, normalize_y=False, mean_model=None, **kwargs):
        MLCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                              atoms, **kwargs)

        self.kernel = kernel
        self.verbose = verbose
        self.opt_fun = opt_fun
        self.opt_method = opt_method
        self.opt_restarts = opt_restarts
        self.normalize_y = normalize_y
        self.mean_model = mean_model

    def add_data(self, atoms):
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.n_atoms = len(atoms)
            self.n_dim = 3*self.n_atoms
            self.x_train = np.zeros((0, self.n_dim))
            self.E_train = np.zeros(0)
            self.F_train = np.zeros(0)
        # else: check if the new atoms object has the same length as previous
        else:
            if not 3*len(atoms) == self.n_dim:
                raise ValueError('New data does not have the same number of '
                                 'atoms as previously added data.')

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        self.x_train = np.append(
            self.x_train, self._transform_input(atoms), axis=0)
        # Call forces first in case forces and energy are calculated at the
        # same time by the calculator
        if self.mean_model is None:
            F = atoms.get_forces().flatten()
            E = atoms.get_potential_energy()
        else:
            F = (atoms.get_forces().flatten()
                 - self.mean_model.get_forces(atoms=atoms).flatten())
            E = (atoms.get_potential_energy()
                 - self.mean_model.get_potential_energy(atoms=atoms))
        self.E_train = np.append(self.E_train, E)
        self.F_train = np.append(self.F_train, F)

    def delete_data(self, indices=None):
        if indices is None:
            indices = slice(len(self.atoms_train))
        del self.atoms_train[indices]
        self.x_train = np.delete(self.x_train, indices, 0)
        self.E_train = np.delete(self.E_train, indices, 0)
        self.F_train = np.delete(self.F_train.reshape(-1, self.n_dim),
                                 indices, 0).reshape(-1)

    def _transform_input(self, atoms):
        return atoms.get_positions().reshape((1, -1))

    def _normalize_input(self, x):
        return x

    def _update_intercept(self):
        try:
            self.intercept = float(self.normalize_y)
        except (ValueError, TypeError):
            if self.normalize_y == 'mean':
                self.intercept = np.mean(self.E_train)
            elif self.normalize_y == 'min':
                self.intercept = np.min(self.E_train)
            elif self.normalize_y == 'max':
                self.intercept = np.max(self.E_train)
            elif self.normalize_y.startswith('max+'):
                self.intercept = (
                    np.max(self.E_train) + float(self.normalize_y[4:]))
            elif self.normalize_y is False or self.normalize_y is None:
                self.intercept = 0.
            else:
                raise NotImplementedError(
                    'Unknown option: %s' % self.normalize_y)

    def fit(self):
        if self.verbose > 0:
            print('Fit called with %d geometries.' % len(self.atoms_train))

        self._update_intercept()

        self._target_vector = np.concatenate(
            [self.E_train - self.intercept, -self.F_train.flatten()])
        # TODO: highly flawed implementation: The recalculation of the kernel
        # matrix after the optimization loop has to be avoided somehow!
        if self.opt_restarts > 0:
            # TODO: Maybe it would be better to start from the same
            # initial_hyper_parameters (given at kernel initialization),
            # every time...

            # Lists to hold the results of the hyperparameter optimizations
            opt_hyper_parameter = []
            # List of values of the marginal log likelihood
            value = []
            for ii in range(self.opt_restarts):
                # First run: start from the current hyperparameters
                if ii == 0:
                    initial_hyper_parameters = self.kernel.theta
                # else: draw from log uniform distribution (drawing from
                # uniform but get and set hyper_parameter work with log values)
                else:
                    bounds = self.kernel.bounds
                    initial_hyper_parameters = np.zeros(len(bounds))
                    for bi, (lower_bound, upper_bound) in enumerate(bounds):
                        initial_hyper_parameters[bi] = np.random.uniform(
                            lower_bound, upper_bound, 1)
                if self.verbose > 0:
                    print(f'Starting hyperparameter optimization {ii+1}/'
                          f'{self.opt_restarts} with parameters: ',
                          np.exp(initial_hyper_parameters))
                try:
                    opt_res = self._opt_routine(initial_hyper_parameters)
                    opt_hyper_parameter.append(opt_res.x)
                    value.append(opt_res.fun)
                    if self.verbose > 0:
                        print('Finished hyperparameter optimization after '
                              f'{opt_res.nit} iterations with value: '
                              f'{opt_res.fun} and parameters:',
                              np.exp(opt_res.x))
                except np.linalg.LinAlgError as E:
                    print('Cholesky factorization failed for parameters:',
                          self.kernel.theta)
                    print(E)

            if len(value) == 0:
                raise ValueError('No successful optimization')
            # Find the optimum among all runs:
            min_idx = np.argmin(value)
            self.kernel.theta = opt_hyper_parameter[min_idx]

        k_mat = self.build_kernel_matrix()
        # Copy original k_mat (without regularization) for later calculation of
        # trainings error
        pure_k_mat = k_mat.copy()
        k_mat[:self.n_samples, :self.n_samples] += np.eye(
            self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(
            self.n_samples * self.n_dim)/self.C2

        self.L, self.alpha = self._cholesky(k_mat)

        y = self.alpha.dot(pure_k_mat)
        E = y[:self.n_samples] + self.intercept
        F = -y[self.n_samples:]
        if self.verbose > 0:
            print('Fit finished. Final RMSE energy = %f, RMSE force = %f.' % (
                np.sqrt(np.mean((E - self.E_train)**2)),
                np.sqrt(np.mean((F - self.F_train)**2))))

    def _cholesky(self, kernel):
        """
        save routine to evaluate the cholesky factorization and weights
        :param kernel: kernel matrix
        :return: lower cholesky matrix, weights.
        """
        L = cholesky(kernel, lower=True)
        alpha = cho_solve((L, True), self._target_vector)
        return L, alpha

    def _opt_routine(self, initial_hyper_parameter):
        if self.opt_method in ['L-BFGS-B', 'SLSQP', 'TNC']:
            opt_res = minimize(self._opt_fun, initial_hyper_parameter,
                               method=self.opt_method, jac=True,
                               bounds=self.kernel.bounds)
        else:
            raise NotImplementedError(
                'Method is not implemented or does not support the use of '
                'bounds use method=L-BFGS-B.')

        return opt_res

    def _opt_fun(self, hyper_parameter):
        """
        Function to optimize kernels hyper parameters
        :param hyper_parameter: new kernel hyper parameters
        :return: negative log marignal likelihood, derivative of the negative
                log marignal likelihood
        """
        self.kernel.theta = hyper_parameter
        if self.opt_fun == 'marginal_likelihood':
            log_marginal_likelihood, d_log_marginal_likelihood = (
                self.log_marginal_likelihood())
            return -log_marginal_likelihood, -d_log_marginal_likelihood
        elif self.opt_fun in ['predictive_probability', 'LOO_CV']:
            log_pred_prob, d_log_pred_prob = self.log_predictive_probability()
            return -log_pred_prob, -d_log_pred_prob
        else:
            raise NotImplementedError(
                'Method is not implemented. Use "marginal_likelihood" or '
                '"predictive_probability"')

    def log_marginal_likelihood(self):
        """
        calculate the log marignal likelihood
        :return: log marinal likelihood, derivative of the log marignal
                 likelihood w.r.t. the hyperparameters
        """
        # gives vale of log marginal likelihood with the gradient
        k_mat, k_grad = self.build_kernel_matrix(eval_gradient=True)
        k_mat[:self.n_samples, :self.n_samples] += np.eye(
            self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(
            self.n_samples*self.n_dim)/self.C2
        try:
            L, alpha = self._cholesky(k_mat)
        except np.linalg.LinAlgError:
            print('excepted LinAlgError for theta: ',
                  np.exp(self.kernel.theta))
            # In order to avoid errors in the hyperparameter optimization
            # some large value and gradient has to be returned
            return -1e60, -1e60*np.sign(self.kernel.theta)
        # Following Rasmussen Algorithm 2.1 the determinant in 2.30 can be
        # expressed as a sum over the Cholesky decomposition L
        log_marginal_likelihood = (- 0.5 * self._target_vector.dot(alpha)
                                   - np.log(np.diag(L)).sum()
                                   - 0.5 * L.shape[0] * np.log(2 * np.pi))

        # summation inspired form scikit-learn Gaussian process regression
        temp = (np.multiply.outer(alpha, alpha) -
                cho_solve((L, True), np.eye(L.shape[0])))[:, :, np.newaxis]
        d_log_marginal_likelihood = 0.5 * np.einsum("ijl,ijk->kl",
                                                    temp, k_grad)
        d_log_marginal_likelihood = d_log_marginal_likelihood.sum(-1)

        return log_marginal_likelihood, d_log_marginal_likelihood

    def log_predictive_probability(self):
        """Leave one out log predictive probability according to Rasmussen
        and Williams"""
        k_mat, k_grad = self.build_kernel_matrix(eval_gradient=True)
        k_mat[:self.n_samples, :self.n_samples] += np.eye(
            self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(
            self.n_samples*self.n_dim)/self.C2
        L, alpha = self._cholesky(k_mat)

        # From scikit-learn implementation for the variances:
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        sigma2 = 1.0/K_inv.diagonal()
        log_pred_prob = np.sum(- 0.5*np.log(sigma2)
                               - 0.5*(alpha/K_inv.diagonal())**2/(sigma2)
                               - 0.5*np.log(2*np.pi))
        Z = K_inv.dot(k_grad)
        d_log_pred_prob = np.sum(
            (alpha[:, np.newaxis]*alpha.dot(Z)
             - 0.5*(1 + alpha**2/K_inv.diagonal())[:, np.newaxis]
             * K_inv.dot(Z).diagonal().T
             )/K_inv.diagonal()[:, np.newaxis], axis=0)
        return log_pred_prob, d_log_pred_prob

    def predict(self, atoms):
        # Prediction
        X_star = self._normalize_input(self._transform_input(atoms))
        y = self.alpha.dot(self.build_kernel_matrix(X_star=X_star))
        E = y[0] + self.intercept
        F = -y[1:].reshape((-1, 3))
        if self.mean_model is not None:
            E += self.mean_model.get_potential_energy(atoms=atoms)
            F += self.mean_model.get_forces(atoms=atoms)
        return E, F

    def predict_var(self, atoms):
        X_star = self._normalize_input(self._transform_input(atoms))
        K_star = self.build_kernel_matrix(X_star=X_star)

        # Scikit-learn implementation for the variances:
        # L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        # K_inv = L_inv.dot(L_inv.T)
        # y_var = self.build_kernel_diagonal(X_star)
        # y_var -= np.einsum('ij,ij->j', K_star, K_inv.dot(K_star))
        # Rasmussen implementation seems numerically more stable:
        v = solve_triangular(self.L, K_star, lower=True)
        y_var = self.build_kernel_diagonal(X_star)
        y_var -= np.einsum('ij,ij->j', v, v)

        E_var = y_var[0]
        F_var = y_var[1:].reshape((-1, 3))
        return E_var, F_var

    def get_params(self):
        return {'atoms_train': self.atoms_train,
                'x_train': self.x_train,
                'alpha': self.alpha,
                'L': self.L,
                'intercept': self.intercept,
                'mean_model': self.mean_model,
                'hyperparameters': self.kernel.theta}

    def set_params(self, **params):
        self.atoms_train = params['atoms_train']
        self.x_train = params['x_train']
        self.n_dim = self.x_train.shape[1]
        self.n_atoms = len(self.atoms_train[0])
        self.alpha = params['alpha']
        self.L = params['L']
        self.intercept = params['intercept']
        self.mean_model = params['mean_model']
        self.kernel.theta = params['hyperparameters']

    def build_kernel_matrix(self, X_star=None, eval_gradient=False):
        """Builds the kernel matrix K(X,X*) of the trainings_examples and
        X_star. If X_star==None the kernel of the trainings_examples with
        themselves K(X,X)."""
        if X_star is None:
            return self.kernel(self.x_train, self.x_train, dx=True, dy=True,
                               eval_gradient=eval_gradient)
        else:
            return self.kernel(self.x_train, X_star, dx=True, dy=True,
                               eval_gradient=eval_gradient)

    def build_kernel_diagonal(self, X_star):
        """Evaluates the diagonal of the kernel matrix which can be done
        significantly faster than evaluating the whole matrix and is needed for
        the uncertainty prediction.
        """
        return self.kernel.diag(X_star)
