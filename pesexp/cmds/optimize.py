import sys
import logging
from pesexp.utils.io import read_terachem_input
from pesexp.cmds.params import parse_opt_args
from pesexp.geometry.coordinate_systems import get_coordinate_system
from pesexp.hessians.hessian_guesses import get_hessian_guess, filter_hessian
from pesexp.optimizers.optimizers import RFO
from pesexp.optimizers.convergence import TerachemConvergence
logger = logging.getLogger(__name__)


def run_optimization(atoms, coords='cart', steps=300,
                     hessian_guess='trivial',
                     hessian_thresh=None, name='molsimp'):

    # Build optimizer with terachem convergence criteria
    class MolSimplifyOpt(TerachemConvergence, RFO):
        pass

    coord_set = get_coordinate_system(atoms, coords)
    H0 = None
    if hessian_guess is not None:
        H0 = get_hessian_guess(atoms, hessian_guess)
        if hessian_thresh is not None:
            H0 = filter_hessian(H0, hessian_thresh)

    opt = MolSimplifyOpt(atoms, coordinate_set=coord_set, H0=H0,
                         trajectory=f'{name}_optim.traj')
    opt.run(steps=steps)


def main():
    args = parse_opt_args(sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG)
    logger.info(f'sys.argv[1:]: {sys.argv[1:]}')

    atoms = read_terachem_input(args['input'])

    # Preprocessing not implemented
    # run_preprocessing(atoms, args['preopt'])

    run_optimization(atoms,
                     coords=args['coords'],
                     steps=args['maxiter'],
                     hessian_guess=args['hessian_guess'],
                     hessian_thresh=args['hessian_thresh'])


if __name__ == '__main__':
    main()
