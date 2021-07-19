import numpy as np
from fitting_functions import B_nu
from astropy.cosmology import Planck15
import astropy.units as u

def get_dustmass(Snu,Td,fit_beta,nu=345.0):
    """
    Quick dust mass calculator, using Greve+12 formula (Eq. 2)
    
    Parameters:
    Snu: float or array of floats or Quantity's
        Flux density at observed frequency nu, assumed mJy if no unit
    z: float
        Redshift of object
    nu: float or Quantity; default 345GHz
        Observed frequency, assumed GHz if no unit
    
    Returns:
    Md: float
        Apparent dust mass in solar masses
    """
    
    # Do some unit handling
    if hasattr(Snu,'unit'):
        Snu = Snu.to('mJy').decompose().value
    else: Snu = u.Quantity(Snu,'mJy').decompose().value
    if hasattr(nu,'unit'):
        nu = (nu.to('GHz')).value

    nu_rest = nu
    beta = fit_beta          # Dust emissivity index
    # Dust opacity; see Hildebrand83,Kruegel&Siebenmorgen94, in m**2 / kg
    # Below value used with beta = 2 in Greve+12
    kappa_nu = 0.045 * (nu_rest/250.)**beta
    # Updated emissivity, reproduces MW dust, see Bianchi+13, beta = 1.5
    #kappa_nu = 0.34 * (nu_rest/1200.)**beta
    # Scoville+14 version, beta = 1.8, from Planck maps, d_GDR = 100
    #kappa_nu = 4.84e-2 * (nu_rest/345.)**beta
    # From Draine&Li07 models / Li&Draine01; beta = 1.7
    #kappa_nu = 0.0431 * (nu_rest/352.)**beta
    #kappa_nu = 3.13 * (nu_rest/3000.)**beta
    
    #DL = u.Quantity(cosmo.luminosity_distance(z),'Mpc').to('m').value
    DL = 10 * u.pc
    Tcmb = 2.73 # z = 0
    denom = B_nu(1e9*nu_rest,Td) - B_nu(1e9*nu_rest,Tcmb)
    Md = (DL.value**2. * Snu / (kappa_nu * denom)) * u.kg
    Md = Md.to(u.M_sun).value
    
    return Md
