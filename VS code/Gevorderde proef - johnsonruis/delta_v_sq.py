import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

f = 100e3 - 100
T = 273+25
R = 100e3
kb = 1.380649e-23
Vsq = 4 * kb * T * R * f

print(Vsq)