import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u

__all__ = ['B_lambda','B_nu','greybody','greybody_powerlaw']

def B_lambda(wav, T):
      from astropy.modeling.blackbody import blackbody_lambda
      flux = blackbody_lambda(wav, T)
      return flux * (4. * np.pi * u.sr) * (10*u.pc).to(u.cm)**2 * wav.to(u.Angstrom)
def B_nu(nu,T):
      #for dust mass calculation
      from astropy.modeling.blackbody import blackbody_nu
      flux = blackbody_nu(nu, T)
      return (flux * (4. * np.pi * u.sr)).to(u.mJy)

def greybody(lambdas,amp,Tdust=35.,L0=1e-4,beta=2.0):
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
      L0 = L0 * u.micron
      Tcmb = 2.73 #[Kelvin] since we're dealing with z=0                                                                                                      
      gbb = (B_lambda(lambdas,Tdust)-B_lambda(lambdas,Tcmb))
      taudust = (L0/lambdas)**beta
      gbb = amp * gbb.value * (lambdas/L0)**2 * (1-np.exp(-taudust))
      #multiply by amp to fix any weird DL issues
      return gbb * (u.erg / u.s)
      
def greybody_powerlaw(lambdas,amp,Tdust=35.,L0=1e-4,beta=2.0,alpha=2.0,z=0.,cosmo=Planck15):
      # Not cleaned yet !

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
      

      
