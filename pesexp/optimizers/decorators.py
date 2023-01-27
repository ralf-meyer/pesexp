import logging


logger = logging.getLogger(__name__)


def anc_rebuilding(opt, n_rebuild=10):
    """Creates a new subclass of the provided optimizer that rebuilds
    the approximate normal coordinates every n_rebuild steps"""

    class Optimizer(opt):
        def step(self, f=None):
            if self.nsteps > 0 and self.nsteps % self.n_rebuild == 0:
                logger.debug("Rebuilding approximate normal coordinates")
                xyzs = self.atoms.get_positions()
                H_cart = self.coord_set.hessian_to_cartesians(xyzs, self.H)
                self.coord_set.build(self.atoms, H_cart)
                self.H = self.coord_set.hessian_to_internals(xyzs, H_cart)
            super().step(f=f)

    Optimizer.n_rebuild = n_rebuild
    return Optimizer
