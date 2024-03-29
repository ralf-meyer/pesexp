import time
import numpy as np
import ase.optimize
from pesexp.optimizers.optimizers import InternalCoordinatesOptimizer


class ConvergenceBase:
    """Mixin class used to replace the convergence criteria of an
    molSimplify.optimize.Optimizer class. Usage:
    >>> class NewOptimizer(ConvergenceMixin, OldOptimizer):
            pass
    >>> opt = NewOptimizer(atoms)
    Note: Because of Pythons method resolution order the mixin class
    needs to come first!
    """

    threshold_energy = 1e-4
    threshold_max_grad = 1e-3
    threshold_rms_grad = 1e-3
    threshold_max_step = 1e-3
    threshold_rms_step = 5e-2
    use_predicted_step = False

    def convergence_condition(
        self, energy_change, max_grad, rms_grad, max_step, rms_step
    ):
        """This is separated out as some convergence criteria might
        want to implement a more sophisticated logic than just all True"""
        conditions = (
            energy_change < self.threshold_energy,
            max_grad < self.threshold_max_grad,
            rms_grad < self.threshold_rms_grad,
            max_step < self.threshold_max_step,
            rms_step < self.threshold_rms_step,
        )
        return all(conditions), conditions

    def irun(self, steps=None):
        """remove fmax from the argument list"""
        return InternalCoordinatesOptimizer.irun(self, fmax=0.0, steps=steps)

    def run(self, steps=None):
        """remove fmax from the argument list"""
        return InternalCoordinatesOptimizer.run(self, fmax=0.0, steps=steps)

    def get_atomwise_step(self):
        r = self.atoms.get_positions()
        if self.use_predicted_step:
            step = self.next_step
        else:
            r0 = self.r0 if self.r0 is not None else r
            step = r - r0
        return np.sqrt(np.sum(step**2, axis=1))

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()

        atomwise_forces = np.sqrt(np.sum(forces**2, axis=1))
        max_grad = np.max(atomwise_forces)
        rms_grad = np.sqrt(np.mean(atomwise_forces**2))

        e = self.atoms.get_potential_energy()
        e0 = self.e0 if self.e0 is not None else e
        energy_change = abs(e - e0)

        atomwise_step = self.get_atomwise_step()
        max_step = np.max(atomwise_step)
        rms_step = np.sqrt(np.mean(atomwise_step**2))

        return self.convergence_condition(
            energy_change, max_grad, rms_grad, max_step, rms_step
        )[0]

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()

        atomwise_forces = np.sqrt(np.sum(forces**2, axis=1))
        max_grad = np.max(atomwise_forces)
        rms_grad = np.sqrt(np.mean(atomwise_forces**2))

        e = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
        e0 = self.e0 if self.e0 is not None else e
        delta_e = e - e0

        atomwise_step = self.get_atomwise_step()
        max_step = np.max(atomwise_step)
        rms_step = np.sqrt(np.mean(atomwise_step**2))

        conditions = self.convergence_condition(
            abs(delta_e), max_grad, rms_grad, max_step, rms_step
        )[1]
        conv = ["*" if c else " " for c in conditions]

        time_now = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (
                    " " * len(name),
                    "Step",
                    "Time",
                    "Energy",
                    "delta_e",
                    "grad_max",
                    "grad_rms",
                    "step_max",
                    "step_rms",
                )
                msg = "%s  %4s %8s %15s %15s  %15s  %15s  %15s  %15s\n" % args
                self.logfile.write(msg)

            msg = (
                f"{name}:  {self.nsteps:3d} {time_now[3]:02d}:{time_now[4]:02d}"
                f":{time_now[5]:02d} {e:15.6f} {delta_e:15.6f}{conv[0]} "
                f"{max_grad:15.6f}{conv[1]} {rms_grad:15.6f}{conv[2]} "
                f"{max_step:15.6f}{conv[3]} {rms_step:15.6f}{conv[4]}\n"
            )
            self.logfile.write(msg)

            self.logfile.flush()


def custom_convergence(opt):
    class CustomConvergence(ConvergenceBase, opt):
        pass

    return CustomConvergence


def terachem_convergence(opt):
    class TerachemConvergence(ConvergenceBase, opt):
        threshold_energy = 1e-6 * ase.units.Hartree
        threshold_max_grad = 4.5e-4 * ase.units.Hartree / ase.units.Bohr
        threshold_rms_grad = 3.0e-4 * ase.units.Hartree / ase.units.Bohr
        threshold_max_step = 1.8e-3 * ase.units.Bohr
        threshold_rms_step = 1.2e-3 * ase.units.Bohr

    return TerachemConvergence


def baker_convergence(opt):
    class BakerConvergence(ConvergenceBase, opt):
        threshold_energy = 1e-6 * ase.units.Hartree
        threshold_max_grad = 3.0e-4 * ase.units.Hartree / ase.units.Bohr
        threshold_max_step = 3.0e-4 * ase.units.Bohr
        use_predicted_step = True

        def convergence_condition(
            self, energy_change, max_grad, rms_grad, max_step, rms_step
        ):
            """This is separated out as some convergence criteria might
            want to implement a more sophisticated logic than just all True"""
            c_energy = energy_change < self.threshold_energy
            c_max_grad = max_grad < self.threshold_max_grad
            c_max_step = max_step < self.threshold_max_step
            conditions = (
                c_energy,
                c_max_grad,
                False,
                c_max_step,
                False,
            )
            return c_max_grad and (c_energy or c_max_step), conditions

    return BakerConvergence
