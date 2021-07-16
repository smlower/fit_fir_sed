import numpy as np
from astropy.constants import c, k_B, h
from astropy.cosmology import Planck15


__all__ = ['B_lambda','B_nu','greybody','greybody_powerlaw',
            'greybody_sourcesize']
            
# This file just holds various possible fitting functions, etc.

# Ditch astropy's units, they're all in SI
c, k_B, h = c.value, k_B.value, h.value

def B_lambda(lambdas,T):
      """
      Return the Planck blackbody function in SI wavelength units,
      W sr^-1 m^-3.
      
      Parameters:
      lambdas: float or array of floats, units of m
            Wavelength(s) where B_lambdas is to be evaluated
      T: float, units of K
            Blackbody temperature. 
      
      Returns:
      B_lambda: float or array of floats, units of W sr^-1 m^-3
            Planck function evaluated at given lambdas and T
      """
      return 2.0 * h * c**2 / (lambdas**5 * (np.exp(h*c/(lambdas * k_B * T)) - 1.0))
      
def B_nu(nu,T):
      """
      Return the Planck blackbody function in SI frequency units,
      W sr^-1 m^-2 Hz^-1
      
      Parameters:
      nu: float or array of floats, units of Hz
            Frequency(ies) where B_nu is to be evaluated
      T: float, units of K
            Blackbody temperature. 
      
      Returns:
      bb: float or array of floats, units of W sr^-1 m^-2 Hz^-1
            Planck function evaluated at given nu and T
      """
      
      return 2.0 * h * nu**3. * c**-2. / (np.exp(h*nu/(k_B * T)) - 1.0)
      
def greybody(lambdas,amp,Tdust=35.,L0=1e-4,beta=2.0,z=0.,cosmo=Planck15):
      """
      Calculate the modified blackbody function ('greybody').
      Note that this equation does match da Cunha et al 2013 eq 17 (2nd line).
      
      Parameters:
      lambdas: float or array of floats, units of m
            Wavelengths to evaluate
      amp: float, vaguely unitless
            Arbitrary normalizing scale factor
      Tdust: float, units of K
            Assumed dust temperature
      L0: float, units of m
            Wavelength at which the dust opacity is 1, ranges from ~40-400um.
      beta: float, unitless
            Dust emissivity spectral index, should range from ~0 to 4.
      z: float, unitless
            Source redshift, to subtract the CMB emission. This is really
            only relevant for very high redshifts (z>~6) and/or cold dust
            temperatures (~20K)
      cosmo: astropy cosmology thing
            Default cosmology is Planck 2015 (paper XIII), but can be changed
            by passing some other astropy cosmology FlatLambdaCDM object here.
            Choice of cosmology has exceedingly little effect on anything.
      
      Returns:
      gbb: array of floats, units related to Jy (scaled by ::amp::)
      """
      Tcmb = 2.73 #[Kelvin] since we're dealing with z=0
      gbb = (B_lambda(lambdas,Tdust)-B_lambda(lambdas,Tcmb))
      taudust = (L0/lambdas)**beta
      gbb = amp * gbb * (lambdas/L0)**2 * (1-np.exp(-taudust))
      
      return gbb
      
def greybody_powerlaw(lambdas,amp,Tdust=35.,L0=1e-4,beta=2.0,alpha=2.0,z=0.,cosmo=Planck15):
      """
      Same as a regular greybody at long wavelengths (>~100um), but instead
      joins the short wavelength Wien side with a power-law of slope alpha.
      The powerlaw is joined to the greybody following Casey2012
      
      Parameters:
      
      Parameters:
      lambdas: float or array of floats, units of m
            Wavelengths to evaluate
      amp: float, vaguely unitless
            Arbitrary normalizing scale factor
      Tdust: float, units of K
            Assumed dust temperature
      L0: float, units of m
            Wavelength at which the dust opacity is 1, ranges from ~40-400um.
      beta: float, unitless
            Dust Rayleigh-Jeans tail emissivity spectral index, 
            should range from ~0 to 4.
      alpha: float, unitless
            Short wavelength powerlaw index (positive alpha is a falling
            spectrum to shorter wavelengths, which you want).
      z: float, unitless
            Source redshift, to subtract the CMB emission. This is really
            only relevant for very high redshifts (z>~6) and/or cold dust
            temperatures (~20K)
      cosmo: astropy cosmology thing
            Default cosmology is Planck 2015 (paper XIII), but can be changed
            by passing some other astropy cosmology FlatLambdaCDM object here.
            Choice of cosmology has exceedingly little effect on anything.
      
      Returns:
      gbbpl: array of floats, units related to Jy (sacled by ::amp::)
      """
      
      gbb = greybody(lambdas,amp,Tdust,L0,beta,z,cosmo)
      
      # Casey2012 calculations for powerlaw scaling and matchup wavelength
      # Solves amppl * Lc**alpha = greybody(Lc,...)
      Lc = 0.75 * 1e-6 * ((26.68 + 6.246*alpha)**-2. + Tdust*(1.905e-4 + 7.243e-5*alpha))**-1.
      amppl = amp * Lc**-alpha * (1-np.exp(-(L0/Lc)**beta)) * (B_lambda(Lc,Tdust)-B_lambda(Lc,cosmo.Tcmb(z).value)) * (Lc/L0)**2.
      powerlaw = amppl * lambdas**alpha * np.exp(-(lambdas/Lc)**2.)
      return gbb + powerlaw
      
def greybody_sourcesize(lambdas,reff,amp=None,Tdust=35.,L0=1e-4,beta=2.0,
      z=0.,cosmo=Planck15):
      """
      Same as the regular greybody, but in this case, instead of using an
      arbitrary normalizing scale factor, we assume we know the actual emitting
      area of the source, which is directly related to the greybody normalization.
      This should be used by sedfit_mcmc if lensmodelpars is passed to the likelihood
      function.
      
      Parameters:
      lambdas: float or array of floats, units of m
            Wavelengths to evaluate
      reff: float, units of kpc
            Source half-light radius.
      amp: ignored
            This parameter is ignored, it's only here so we can use the *exploder
            in the likelihood function.
      Tdust: float, units of K
            Assumed dust temperature
      L0: float, units of m
            Wavelength at which the dust opacity is 1, ranges from ~40-400um.
      beta: float, unitless
            Dust emissivity spectral index, should range from ~0 to 4.
      z: float, unitless
            Source redshift, to subtract CMB emission and rescale source
            ::reff:: into steradians from kpc.
      cosmo: astropy cosmology thing
            Default cosmology is Planck 2015 (paper XIII), but can be changed
            by passing some other astropy cosmology FlatLambdaCDM object here.
      
      Returns:
      gbb: array of floats, SI units of W m^-2 Hz^-1 [ie, mult by 1e29 to mJy]
      """
      
      nu = c/lambdas
      DA = 1e3*cosmo.angular_diameter_distance(z).value
      tau = (L0/lambdas)**beta
      deltaB = B_nu(nu,Tdust) - B_nu(nu,cosmo.Tcmb(z).value)
      return 2*np.pi*reff**2. * deltaB * (1-np.exp(-tau)) / (DA**2 * (1+z)**3)
      
