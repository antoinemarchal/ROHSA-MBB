import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

import mod_optimize as mod_opt
reload(mod_opt)

wavelength = np.arange(90,900,1) * u.micron
freq = (const.c / wavelength).to(u.GHz)

sigma = np.arange(10) *u.cm**2
beta = np.linspace(1.,2.5,10)
T = np.linspace(8.2,20.,10) *u.K
NHI = 1. * u.cm**-2

I_l_beta = [mod_opt.MBB_l(wavelength,NHI,np.mean(sigma),b,np.mean(T),(const.c/(353*u.GHz)).to(u.micron)).to(u.W*u.m**-3) for b in beta]
I_l_T = [mod_opt.MBB_l(wavelength,NHI,np.mean(sigma),np.mean(beta),tdust,(const.c/(353*u.GHz)).to(u.micron)).to(u.W*u.m**-3) for tdust in T]

I_l_beta_adim = [mod_opt.MBB_l_adim(wavelength,NHI,np.mean(sigma),b,np.mean(T),(const.c/(353*u.GHz)).to(u.micron)) for b in beta]
I_l_T_adim = [mod_opt.MBB_l_adim(wavelength,NHI,np.mean(sigma),np.mean(beta),tdust,(const.c/(353*u.GHz)).to(u.micron)) for tdust in T]

# fig, (ax1, ax2) = plt.subplots(1, 2)
# for I in I_l_beta: ax1.plot(wavelength, I)
# for I in I_l_T: ax2.plot(wavelength, I)

fig, (ax1, ax2) = plt.subplots(1, 2)
for I in I_l_beta_adim: ax1.plot(wavelength, I)
for I in I_l_T_adim: ax2.plot(wavelength, I)
ax1.set(xlabel=r"$\lambda$ [micron]", ylabel='I [adim]')
ax2.set(xlabel=r"$\lambda$ [micron]", ylabel='I [adim]')
ax1.set_title('Variation Beta')
ax2.set_title('Variation T')





