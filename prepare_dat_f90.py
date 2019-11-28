import numpy as np
from astropy.io import fits
from astropy import constants as const
from astropy import units as u

from ROHSApy import ROHSA
import turbulence as tb

#Open data GNILC_IRIS-SFD_adim
path = "/data/amarchal/GNILC_IRIS-SFD_adim/"
fitsname = "GNILC_IRIS-SFD_adim_G86.fits"
hdu = fits.open(path+fitsname)
hdr = hdu[0].header
cube = hdu[0].data[:,44-32:44+32,44-32:44+32] *1.e6 #Attention rescale

freq = np.array([np.full((cube.shape[1],cube.shape[2]),i) for i in np.array([353,545,857,3000])])*u.GHz
wavelength = (const.c / freq).to(u.micron)
wavelength_full = np.array([np.full((cube.shape[1],cube.shape[2]),i) for i in np.arange(100,900,1)]) * u.micron

#Open HI model CNM WNM with ROHSA
path_HI = "/data/amarchal/G86/model/"
fitsname_HI = "GHIGLS_G86_LVC_IVC_CNM_WNM.fits"
hdu_HI = fits.open(path_HI+fitsname_HI)
hdr_HI = hdu_HI[0].header
NHI = hdu_HI[0].data[:,44-32:44+32,44-32:44+32] *u.cm**-2 # 1.e18

# NHI = np.zeros((1,cube.shape[1],cube.shape[2]))
# NHI[0,:,:] = np.sum(hdu_HI[0].data[:,44-32:44+32,44-32:44+32],0)
# NHI = NHI*u.cm**-2

rms_map = np.ones((cube.shape[1],cube.shape[2]))

core_cube = ROHSA(cube)
core_NHI = ROHSA(NHI.value)

core_cube.cube2dat(filename="/home/amarchal/ROHSA/src/SED.dat")
core_NHI.cube2dat(filename="/home/amarchal/ROHSA/src/NHI.dat")
core_cube.rms_map(rms_map, filename="/home/amarchal/ROHSA/src/rms.dat")


