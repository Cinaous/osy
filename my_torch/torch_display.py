from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def himmenlblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


ax = Axes3D(figure())
X = np.arange(-6, 6, 0.25)
X, Y = np.meshgrid(X, X)
Z = himmenlblau([X, Y])
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
show()
