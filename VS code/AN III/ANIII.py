import sympy as sp
import numpy as np

h = 6.626e-34
c = 3e8
eV = 1.6e-19
me = 9.11e-31
mp = 1.67e-27
mn = 1.67e-27
epsilon0 = 8.85e-12
k_coul = 9e9
hbar = sp.N(h / (2 * sp.pi))
u = 1.66e-27
kb = 1.38e-23
pi = sp.N(sp.pi)
mu0 = 4e-7 * pi

v = sp.sqrt(eV**2 / (4 * sp.pi * epsilon0 * me * 5e-11))
r = 5e-11

P = (mu0 * eV**2 * v**4) / (6 * pi * c * r**2)

T = 1/2 * me * v**2
V = 1/(4 * pi * epsilon0) * eV**2 / r
print(sp.N((T+V) / P))