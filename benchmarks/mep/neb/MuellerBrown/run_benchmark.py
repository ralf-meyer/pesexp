from ase.atoms import Atoms
from ase.neb import NEBTools
from ase.optimize import LBFGS
from pesexp.mep.neb import NEB
from pesexp.calculators import MuellerBrownSurface
from utils import get_MB_surface
import numpy as np
import matplotlib.pyplot as plt

initial = Atoms(["H"], positions=[[*MuellerBrownSurface.min_A, 0]])
final = Atoms(["H"], positions=[[*MuellerBrownSurface.min_C, 0]])

images = [initial] + [initial.copy() for _ in range(11)] + [final]
for image in images:
    image.calc = MuellerBrownSurface()

neb = NEB(images, k=1e-1, method="improvedtangent")
neb.interpolate()

opt = LBFGS(neb, alpha=1e3)
opt.run(fmax=0.05)

xyzs = np.array([im.get_positions() for im in images])

x = np.linspace(-1.6, 1.1, 41)
y = np.linspace(-0.3, 2.0, 41)
surface = get_MB_surface(x, y)
levels = np.linspace(surface.min(), 0.0, 31)

fig, ax = plt.subplots(ncols=2)
ax[0].contourf(x, y, surface, levels=levels, extend="max")
ax[0].plot(xyzs[:, 0, 0], xyzs[:, 0, 1], "o-k")

NEBTools(images).plot_band(ax=ax[1])

plt.show()
