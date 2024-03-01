import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = np.linspace(200, 20000, 100)

# Constants
C = 1.13
D = 0.2
u = 50  # m/s
r = 1  # m
x = 0.25  # m
h = 0.024  # NACA0012 so 12% of the chord length

span = 0.2
b = span / 2

m1 = 20 / 1000
m2 = 25 / 1000
a1 = 5 / 1000
a2 = 7 / 1000

M = u / 343

gammax1 = a1 * D * np.sqrt(x / a1)
gammax2 = a1 * D * np.sqrt(x / a1)

Tu1 = C * (x / a1)**(-5 / 7)
Tu2 = C * (x / a2)**(-5 / 7)

Tu1perc = Tu1 * 100
Tu2perc = Tu2 * 100

print('Turbulence Intensity with Grid 1: ', Tu1perc, '%')
print('Turbulence Intensity with Grid 2: ', Tu2perc, '%')

Kappax1 = (8 * np.pi * f * Tu1) / (3 * u)
Kappax2 = (8 * np.pi * f * Tu2) / (3 * u)

# for leading noise produced

term1 = ((gammax1 * b) / (r**2)) * (M**5 * Tu1**2) * ((Kappax1**3) / ((1 + Kappax1)**(7 / 3))) * np.exp(
    (-np.pi * f * h) / (u))

lp_dash1 = 10 * np.log10(term1)
Lp1 = lp_dash1 + 181.3

term2 = ((gammax2 * b) / (r**2)) * (M**5 * Tu1**2) * ((Kappax2**3) / ((1 + Kappax2)**(7 / 3))) * np.exp(
    (-np.pi * f * h) / (u))

lp_dash2 = 10 * np.log10(term2)
Lp2 = lp_dash2 + 181.3


plt.plot(f, Lp2,'bo-', markersize=4, label='Grid 1')
plt.xscale('log')
plt.xlabel(r'$f_{c}$ in Hz')  # Increase fontsize
plt.ylabel(r'$L_{p}$ in dB')  # Increase fontsize
plt.title('Turbulence Grid 1: m = 20mm, a = 5mm')  # Increase fontsize
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
desired_dpi = 300
plt.savefig('Lp.pdf', dpi=desired_dpi)
plt.show()
