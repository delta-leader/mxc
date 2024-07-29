import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import sys

csv = sys.argv[1]
pdf = sys.argv[2]

iters = np.loadtxt(csv, dtype=float, delimiter=',', usecols=22)
resid_h2 = np.loadtxt(csv, dtype=float, delimiter=',', skiprows = 0, max_rows = 1, usecols=np.arange(25, 26+iters[0], dtype=int))
resid_hss = np.loadtxt(csv, dtype=float, delimiter=',', skiprows = 1, max_rows = 1, usecols=np.arange(25, 26+iters[1], dtype=int))
resid_none = np.loadtxt(csv, dtype=float, delimiter=',', skiprows = 2, max_rows = 1, usecols=np.arange(25, 26+iters[2], dtype=int))

plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots()
ax.set_xlabel('GMRES iterations')
ax.set_ylabel('Residual')

ax.plot(np.arange(0, 20*resid_h2.size, 20), resid_h2, '^', label = "H2 (SC22)", linestyle="-", color='blue')
ax.plot(np.arange(0, 20*resid_hss.size, 20), resid_hss, '^', label = "HSS", linestyle="-", color='red')
ax.plot(np.arange(0, 20*resid_none.size, 20), resid_none, '^', label = "No Precondition", linestyle="-", color='yellow')

ax.axis([0, 1000, 1.e-14, 1])
try:
    ax.set_yscale('log', base=10)
except ValueError:
    ax.set_yscale('log', basey=10)

ax.grid(True, which='both')
ax.legend(fontsize=11)

plt.subplots_adjust(left=0.2,right=0.9,bottom=0.15)
fig.savefig(pdf)
