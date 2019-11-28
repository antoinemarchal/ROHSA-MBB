import numpy as np
from astropy.io import fits
from astropy.io import fits as pyfits
import astropy.table as pytabs
import matplotlib.pyplot as plt
import warnings
from astropy.modeling import models, fitting
from scipy import optimize
from astropy import units as u
from astropy import constants as const

plt.ion()

def n_coeff(degree):
    k = 0
    for i in np.arange(degree):
        for j in np.arange(degree+1-i):
            k += 1
    return k+1


def f(beta, xx, yy, degree, data):
    model = 0.

    k = 0
    for i in np.arange(degree):
        for j in np.arange(degree+1-i):
            model += beta[k] * xx**i * yy**j
            k += 1
    return (model - data).ravel()

                
def poly(coef, xx, yy, degree):
    model = 0.
    k = 0
    for i in np.arange(degree):
        for j in np.arange(degree+1-i):
            model += coef[k] * xx**i * yy**j
            k += 1
    return model


def fit_poly(x, y, degree, data):
    beta = np.zeros(n_coeff(degree))
    bounds = [(-1, 1) for i in np.arange(n_coeff(degree))]
    
    result = optimize.leastsq(f, beta, args=(x, y, degree, data))
    return result[0]
    
    
path = "/data/amarchal/PLANCK/"
fitsname = "col_cor_hfi_iras_dirbe_DX9v2.fits"

cat = pyfits.getdata(path + fitsname)
table = pytabs.Table(cat)

beta, Td = np.meshgrid(table["BETA"].data, table["TD"].data)
freq = np.array([3000,857,545,353]) * u.GHz

IRAS4 = table["IRAS4"].data[0]
DIRBE8 = table["DIRBE8"].data[0]
DIRBE9 = table["DIRBE9"].data[0]
DIRBE10 = table["DIRBE10"].data[0]
HFI1 = table["HFI1"].data[0]
HFI2 = table["HFI2"].data[0]
HFI3 = table["HFI3"].data[0]

degree = 5 
n = n_coeff(degree)

fit_IRAS4 = fit_poly(beta, Td, degree, IRAS4)
fit_DIRBE8 = fit_poly(beta, Td, degree, DIRBE8)
fit_DIRBE9 = fit_poly(beta, Td, degree, DIRBE9)
fit_DIRBE10 = fit_poly(beta, Td, degree, DIRBE10)
fit_HFI1 = fit_poly(beta, Td, degree, HFI1)
fit_HFI2 = fit_poly(beta, Td, degree, HFI2)
fit_HFI3 = fit_poly(beta, Td, degree, HFI3)

model_IRAS4 = poly(fit_IRAS4, beta, Td, degree)
model_DIRBE8 = poly(fit_DIRBE8, beta, Td, degree)
model_DIRBE9 = poly(fit_DIRBE9, beta, Td, degree)
model_DIRBE10 = poly(fit_DIRBE10, beta, Td, degree)
model_HFI1 = poly(fit_HFI1, beta, Td, degree)
model_HFI2 = poly(fit_HFI2, beta, Td, degree)
model_HFI3 = poly(fit_HFI3, beta, Td, degree)

# polynome = np.zeros((7, n))
# polynome[0,:] = fit_IRAS4
# polynome[1,:] = fit_DIRBE8
# polynome[2,:] = fit_DIRBE9
# polynome[3,:] = fit_DIRBE10
# polynome[4,:] = fit_HFI1
# polynome[5,:] = fit_HFI2
# polynome[6,:] = fit_HFI3

polynome = np.zeros((4, n))
polynome[0,:] = fit_IRAS4
polynome[1,:] = fit_HFI1
polynome[2,:] = fit_HFI2
polynome[3,:] = fit_HFI3

#Write output local
hdu0 = fits.PrimaryHDU(polynome)
hdulist = fits.HDUList([hdu0])
hdulist.writeto(path + "col_cor_iras_hfi_DX9v2_poly.fits", overwrite=True)

stop

# # Fit the data using astropy.modeling
# degree = 3

# p_init = models.Polynomial2D(degree=degree)
# fit_p = fitting.LevMarLSQFitter()

# with warnings.catch_warnings():
#     # Ignore model linearity warning from the fitter
#     warnings.simplefilter('ignore')
#     p = fit_p(p_init, x, y, data)

# model = p(x,y)

# reindex = np.zeros(len(p.parameters))
# k = 0
# for i in np.arange(degree):
#     for j in np.arange(degree+1-i):
#         name = 'c{0}_{1}'.format(i, j)
#         reindex[k] = p.parameters[p.param_names.index(name)]
#         print (i,j), p.parameters[p.param_names.index(name)]

#         f +=  p.parameters[p.param_names.index(name)] * x**j * y**i
#         k += 1
        
# print "_________________"










