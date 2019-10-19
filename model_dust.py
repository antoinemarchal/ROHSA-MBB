import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

def planck_nu(nu, T):
    return 2. * const.h * nu**3. / const.c**2 * 1./(np.exp(const.h*nu/const.k_B/T) - 1.)

def intensity(nu, beta, tau, T, nu0):
    return tau * (nu/nu0)**beta * planck_nu(nu,T)


wavelength = np.arange(40,3000,1) * u.micron
freq = (const.c / wavelength).to(u.GHz)

I = intensity(freq,1.77,1.e-10,16.47*u.K,353*u.GHz).to(u.mJy)


