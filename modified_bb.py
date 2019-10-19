import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

import mod_optimize
reload(mod_optimize)

import turbulence as tb

shape = (64,64,64)
sig_conv = 1.
gamma = 3.

cube = tb.fBmnd(shape, tb.Pkgen(gamma,0.,np.inf), seed=31, unit_length=1)
NHI = 1.e19 * u.cm**-2

wavelength = np.array([np.full((shape[1],shape[2]),i) for i in np.arange(100,900,1)]) * u.micron
freq = (const.c / wavelength).to(u.GHz)

# freq = np.array([np.full((shape[1],shape[2]),i) for i in np.array([353,545,857,3000])])*u.GHz
# wavelength = (const.c / freq).to(u.micron)

beta = np.full((shape[1],shape[2]),1.77)
sigma = np.full((shape[1],shape[2]),1.e-25) * u.cm**2
T = np.full((shape[1],shape[2]),16.47) * u.K

#Compute the mean spectrum
mean_spectrum = np.mean(cube,(1,2))

#Simulate Planck+IRAS MBB
I_nu = mod_optimize.MBB_nu(freq,NHI,sigma,beta,T,353*u.GHz).to(u.mJy) 
I_l = mod_optimize.MBB_l(wavelength,NHI,sigma,beta,T,(const.c/(353*u.GHz)).to(u.micron)).to(u.W*u.m**-3) 

I_nu_adim = mod_optimize.MBB_nu_adim(freq,NHI,sigma,beta,T,353*u.GHz)
I_l_adim = mod_optimize.MBB_l_adim(wavelength,NHI,sigma,beta,T,(const.c/(353*u.GHz)).to(u.micron))










