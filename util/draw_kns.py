import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

exact_sod = pd.read_fwf('../calculated_data/exact_sod/waf.dat')

Kn0009 = pd.read_csv('../calculated_data/labubuKolgan/n_x:60_xi:(-10,10,40)_t:0.28300000000000003_CFL:0.45_Kn:9e-05.dat')
Kn009 = pd.read_csv('../calculated_data/labubuKolgan/n_x:60_xi:(-10,10,40)_t:0.28300000000000003_CFL:0.45_Kn:0.009.dat')
Kn09 = pd.read_csv('../calculated_data/labubuKolgan/n_x:60_xi:(-10,10,40)_t:0.28300000000000003_CFL:0.45_Kn:0.9.dat')

fig, axs = plt.subplots(1, 1)
fig.suptitle(f't=0.28, Kolgan')

csh = 0.9

axs.set_title('n (density)')
axs.scatter((Kn09['x']-0.5)*csh + 0.5, Kn09['n'], color='red', linewidth=0.01, label='Kn=0.9')
axs.plot((Kn09['x']-0.5)*csh + 0.5, Kn09['n'], color='red')


axs.scatter((Kn09['x']-0.5)*csh + 0.5, Kn0009['n'], color='blue', linewidth=0.01, label='Kn=0.00009')
axs.plot((Kn09['x']-0.5)*csh + 0.5, Kn0009['n'], color='blue')

axs.plot(exact_sod['X'], exact_sod['RHO'], color='black', label='exact')
axs.grid()
axs.legend()
"""
axs[1].set_title('u (velocity)')
axs[1].scatter(Kn0009['x'], Kn0009['u'], color='blue', linewidth=0.01, label='Kn=0.009')
axs[1].plot(Kn0009['x'], Kn0009['u'], color='blue')
#axs[1].scatter(Kn009['x'], Kn009['u'], color='green', linewidth=0.01, label='Kn=0.09')
#axs[1].plot(Kn009['x'], Kn009['u'], color='green')
axs[1].scatter(Kn09['x'], Kn09['u'], color='red', linewidth=0.01, label='Kn=0.9')
axs[1].plot(Kn09['x'], Kn09['u'], color='red')
axs[1].plot(exact_sod['X'], exact_sod['U'] / 2**0.5, color='black', label='exact')
axs[1].grid()
axs[1].legend()

axs[2].set_title('T (temperature)')
axs[2].scatter(Kn0009['x'], Kn0009['T'], color='blue', linewidth=0.01, label='Kn=0.009')
axs[2].plot(Kn0009['x'], Kn0009['T'], color='blue')
#axs[2].scatter(Kn009['x'], Kn009['T'], color='green', linewidth=0.01, label='Kn=0.09')
#axs[2].plot(Kn009['x'], Kn009['T'], color='green')
axs[2].scatter(Kn09['x'], Kn09['T'], color='red', linewidth=0.01, label='Kn=0.9')
axs[2].plot(Kn09['x'], Kn09['T'], color='red')
axs[2].plot(exact_sod['X'], exact_sod['P']/exact_sod['RHO'], color='black', label='exact')
axs[2].grid()
axs[2].legend()

axs[3].set_title('q (termal_flow)')
axs[3].scatter(Kn0009['x'], Kn0009['q'], color='blue', linewidth=0.01, label='Kn=0.009')
axs[3].plot(Kn0009['x'], Kn0009['q'], color='blue')
axs[3].scatter(Kn009['x'], Kn009['q'], color='green', linewidth=0.01, label='Kn=0.09')
axs[3].plot(Kn009['x'], Kn009['q'], color='green')
axs[3].scatter(Kn09['x'], Kn09['q'], color='red', linewidth=0.01, label='Kn=0.9')
axs[3].plot(Kn09['x'], Kn09['q'], color='red')
#axs[3].plot(exact_sod['X'], exact_sod['P']/exact_sod['RHO'], color='black', label='exact')
axs[3].grid()
axs[3].legend()
"""

l, r = -0.5, 1.5
axs.axis(xmin=l,xmax=r)



plt.show()