import numpy as np
from pesexp.calculators import MuellerBrownSurface


def get_MB_surface(x, y):
    surface = np.zeros((len(y), len(x)))

    calc = MuellerBrownSurface()

    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            surface[i, j] = calc.energy(xj, yi)
    return surface
