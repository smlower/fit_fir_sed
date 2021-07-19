import numpy as np
from fitting_functions import B_nu
from astropy.constants import c
from astropy.cosmology import Planck15
import astropy.units as u

def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

def get_dustmass(lum,wav,Td,fit_beta,nu=3000):
    """
    Quick dust mass calculator, using Greve+12 formula (Eq. 2)
    
    Parameters:
    Snu: float or array of floats or Quantity's
        Flux density at observed frequency nu, assumed mJy if no unit
    nu: float or Quantity; default 345GHz
        Observed frequency, assumed GHz if no unit
    
    Returns:
    Md: float
        Apparent dust mass in solar masses
    """
    beta = fit_beta          # Dust emissivity index
    beta_array = np.array([1.5, 1.7,1.75,1.8,2.0]) #these and kappa are from emissivities.txt
    kappa870_array = np.array([0.0414,0.07,0.078,0.0484,0.0413])
    kappa_fit = kappa870_array[find_nearest(beta_array, beta)]* (u.m**2 * u.kg**(-1))
    Snu_mJy = (lum / ((10 * u.pc).to(u.cm)**2 * (c / wav.to(u.m)))).to(u.mJy)
    #nu=3000Ghz = 100 micron
    DL = 10 * u.pc
    Tcmb = 2.73 # z = 0
    denom = B_nu(nu*u.GHz,Td*u.K) - B_nu(nu*u.GHz,Tcmb*u.K)
    Md = (DL.to(u.m)**2. * Snu_mJy[find_nearest(wav.to(u.micron).value,100)] / (kappa_fit * denom))
    Md = Md.to(u.M_sun).value
    
    return Md
