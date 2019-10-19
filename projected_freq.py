import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy import wcs
from astropy import constants as const
import FITS_tools
import healpy as hp
from reproject import reproject_from_healpix, reproject_to_healpix

from mylibrary.CGD import mod_tools
from mylibrary.PROJ import mod_projection
from mylibrary.PLOT import mod_plot

import turbulence as tb

plt.ion()

#Resolution
arcmin2rad = u.arcmin.to(u.radian)
resolution_GNILC = 5. #arcmin                                        
resolution_GBT = 9.55 #arcmin
kernel = np.sqrt(resolution_GBT**2 - resolution_GNILC**2)

#Load Planck GNILC frequency maps + IRIS/SFR combined
# path_pl = "/data/mmiville/Planck/PR3/SKYMAP/"
# path_iras = "/data/mmiville/IRIS_Healpix/"
path_pl = "/data/amarchal/PLANCK/"
path_iras = "/data/amarchal/IRAS/"

# pl_353, hdr_353 = hp.read_map(path_pl + "HFI_SkyMap_353-field-Int_2048_R3.00_full.fits", h=True)
# pl_545, hdr_545 = hp.read_map(path_pl + "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits", h=True)
# pl_857, hdr_857 = hp.read_map(path_pl + "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits", h=True)

# iris_3000, hdr_3000 = hp.read_map(path_iras + "IRIS_combined_SFD_really_nohole_4_2048.fits", h=True)

pl_353, hdr_353 = hp.read_map(path_pl + "COM_CompMap_Dust-GNILC-F353_2048_R2.00.fits", h=True)
pl_545, hdr_545 = hp.read_map(path_pl + "COM_CompMap_Dust-GNILC-F545_2048_R2.00.fits", h=True)
pl_857, hdr_857 = hp.read_map(path_pl + "COM_CompMap_Dust-GNILC-F857_2048_R2.00.fits", h=True)

iris_3000, hdr_3000 = hp.read_map(path_iras + "IRIS_combined_SFD_really_nohole_nosource_4_2048.fits", h=True)

pl_353_smoothed = hp.smoothing(pl_353, fwhm=kernel*arcmin2rad)
pl_545_smoothed = hp.smoothing(pl_545, fwhm=kernel*arcmin2rad)
pl_857_smoothed = hp.smoothing(pl_857, fwhm=kernel*arcmin2rad)
iris_3000_smoothed = hp.smoothing(iris_3000, fwhm=kernel*arcmin2rad)

#Projection parameters
nside = 2048
reso = np.sqrt(hp.nside2pixarea(nside, degrees=True))/2.

# sizex = 1200
# Glon = 174.1331
# Glat = -13.4523

# #Define WCS + header
# target_wcs = mod_projection.set_wcs(sizex, 'GLON-TAN', 'GLAT-TAN', reso, Glon, Glat)
# target_hdr = target_wcs.to_header()

# size=(1200,1200)

#Or open wcs from FITS file (HI for exemple)
path = "/data/amarchal/G86/"
fitsname = "data/GHIGLS_G86_Tb.fits"
hdu = fits.open(path+fitsname)
hdr = hdu[0].header
cube = hdu[0].data[0]
size=(cube.shape[1],cube.shape[2])
target_wcs = tb.proj.wcs2D(hdr)
target_hdr = target_wcs.to_header()

freq = np.array([np.full((size[0],size[1]),i) for i in np.array([353,545,857,3000])])*u.GHz
wavelength = (const.c / freq).to(u.micron)

#Projection
proj_353, foo = reproject_from_healpix((pl_353_smoothed,'g'), target_hdr, shape_out=size, nested=False)
proj_545, foo = reproject_from_healpix((pl_545_smoothed,'g'), target_hdr, shape_out=size, nested=False)
proj_857, foo = reproject_from_healpix((pl_857_smoothed,'g'), target_hdr, shape_out=size, nested=False)
proj_3000, foo = reproject_from_healpix((iris_3000_smoothed,'g'), target_hdr, shape_out=size, nested=False)

cube = np.array([proj_353,proj_545,proj_857,proj_3000]) * u.MJy / (2. * const.h * freq**3. / const.c**2)
cube = cube.decompose()

#Write FITS file
# pathout = "/data/amarchal/PLANCK_IRIS-SFD_adim/"
pathout = "/data/amarchal/GNILC_IRIS-SFD_adim/"

hdu = fits.PrimaryHDU(cube.value)
hdu.header["CRPIX1"] = target_hdr["CRPIX1"]
hdu.header["CRVAL1"] = target_hdr["CRVAL1"]
hdu.header["CDELT1"] = target_hdr["CDELT1"]
hdu.header["CTYPE1"] = target_hdr["CTYPE1"]

hdu.header["CRPIX2"] = target_hdr["CRPIX2"]
hdu.header["CRVAL2"] = target_hdr["CRVAL2"]
hdu.header["CDELT2"] = target_hdr["CDELT2"]
hdu.header["CTYPE2"] = target_hdr["CTYPE2"]

hdulist = fits.HDUList([hdu])
hdulist.writeto(pathout + "Planck_GNILC_IRIS-SFD_adim_{}_{}_{}.fits".format(Glon, Glat, sizex), clobber=True)
# hdulist.writeto(pathout + "PLANCK_IRIS-SFD_adim_G86.fits", overwrite=True)
