import numpy as np
import emcee
from tqdm.auto import tqdm
from scipy.special import erfc
from fitting_functions import *
from get_luminosity import get_luminosity
from get_dustmass import get_dustmass
import astropy.units as u

__all__ = ['calc_prior','calc_likelihood','sedfit_mcmc']

def calc_prior(p,fitfunction, params):
      """
      We can impose some priors on parameters here. In this case, 
      I just use some loose flat priors to exclude unphysical values.
      This depends a bit on which function we're fitting, exactly.
      In principle can easily alter this, eg to have a gaussian prior
      on Tdust for sources w/o spec-z's.
      
      This is a bit kludgy, eventually should just have the fitfunctions
      be classes instead of only functions, and implement priors directly
      there.
      
      Parameters:
      p: parameter array
      fitfunction: The function being fitted.
      
      Returns:
      lnlike_prior: float
            The log-likelihood from the priors.
      """
      
      plike = 0.
      
      if fitfunction.__name__ == 'greybody':
            # param ordering is [amp,Tdust,L0,beta]
            # setting priors as uniform
            if p[params.index('amp')] < 0: plike -= np.inf
            if (p[params.index('Td')]<15) or (p[params.index('Td')]>300): plike -= np.inf
            if (p[params.index('L0')]<50) or (p[params.index('L0')]>500): plike -= np.inf
            if (p[params.index('beta')]<0.) or (p[params.index('beta')]>4.): plike -= np.inf
            return plike
      elif fitfunction.func_name == 'greybody_powerlaw':
            # param ordering is[amp,Tdust,L0,beta,alpha,z]
            if p[0] < 0: plike -= np.inf
            if (p[1]<15) or (p[1]>300): plike -= np.inf
            if (p[2]<50e-6) or (p[2]>500e-6): plike -= np.inf
            if (p[3]<0.) or (p[3]>4.): plike -= np.inf
            if (p[4]<0.25) or (p[4]>3.): plike -= np.inf
            if (p[5]<0.) or (p[5]>10.): plike -= np.inf
            return plike
      

def calc_likelihood(p,fitfunction,um_data,mjy_data,mjy_errors,params,fixed):
      """
      Calculate the log-likelihood of a model given some data.
      
      Parameters:
      p: array of function values
            The proposed values for each parameter being fit for. If
            any are actually being held fixed (see below), those params
            are ignored.
      fitfunction: python function
            The function we're fitting to. See fitting_functions for the
            currently available choices.
      um_data,mjy_data,mjy_errors: arrays, units of um, mJy, and mJy
            Three arrays containing the data point wavelengths (in um),
            flux densities (in mJy), and uncertainties (in mJy). Upper limits
            can be indicated by setting the flux at that wavelength to be
            <= 0, in which case we use the uncertainty to calculate whether
            the proposed model agrees with the non-detection.
      fixed: array
            Array describing which parameter(s) should be held fixed
            during fitting. Should be same size as ::p::. If the i'th parameter
            is free, fixed[i]=False. If the i'th parameter is held fixed
            to the value x, fixed[i]=x. For example, if fitfunction is the
            greybody and we want to fix beta and z_spec, we should have
            fixed = [False,False,False,beta,z_spec].
      Returns:
      lnlike: float
            The log-likelihood of the model given the data, divided by the
            number of degrees of freedom (Ndatapoints - Nfreeparams). Sums the 
            likelihoods of the detected, undetected, and lens model points.
      """
      
      # Fix any values we're fixing to the given values.
      for i,val in enumerate(fixed):
            if val: p[i] = val
      
      # Figure out if any parameters are excluded by priors
      priorlik = calc_prior(p,fitfunction, params)
      # if prior likelihood is -inf, don't need to do anything else
      if ~np.isfinite(priorlik): return priorlik 
      modeldata = fitfunction(um_data, *p)
      # Calculate likelihood for detected points, usual chi-squared definition
      detlik = np.sum((mjy_data - modeldata)**2./(mjy_errors)**2.)
      
      freepars = np.where(np.asarray(fixed)==0)[0]
      dof = float(um_data.size-len(freepars))
      return -(1./dof)*(priorlik + detlik)
      
      
      
def sedfit_mcmc(fitfunc,um_data,mjy_data,mjy_errors,p0=None,params=None,fixed=None,
                mcmcsteps=[100,100,100],nthreads=2,pool=None):

      """
      Perform the actual SED fitting, and save/output the results.
      
      Parameters:
      fitfunc: python function
            The function we're going to fit to, see fitting_functions for
            options. Currently have greybody, greybody_powerlaw.      
      um_data,mjy_data,mjy_errors: arrays, units of um, mJy, and mJy
            Three arrays containing the data point wavelengths (in um),
            flux densities (in mJy), and uncertainties (in mJy). Upper limits
            can be indicated by setting the flux at that wavelength to be
            <= 0, in which case we use the uncertainty to calculate whether
            the proposed model agrees with the non-detection.
        p0: array 
      fixed: array
            Array describing which parameter(s) should be held fixed
            during fitting. Should be same size as ::p::. If the i'th parameter
            is free, fixed[i]=False. If the i'th parameter is held fixed
            to the value x, fixed[i]=x. For example, if fitfunction is the
            greybody and we want to fix beta and z_spec, we should have
            fixed = [False,False,False,beta,z_spec].
        """

      if p0 is None:
            if fitfunc.__name__ == 'greybody': p0 = np.array([1.,40.,150,2])
            elif  fitfunc.__name__ == 'greybody_powerlaw': p0 = np.array([1.,40.,150e-6,2,2,3]) 
      else: p0 = np.asarray(p0)
      
            
      if fixed is None:
            fixed = np.array([False for p in p0])
      
      # Some setup for emcee
      nwalkers,nburn,nstep = mcmcsteps
      ndim = p0.size
      initials = np.abs(emcee.utils.sample_ball(p0,0.1*p0,int(nwalkers)))
      
      sampler = emcee.EnsembleSampler(nwalkers,ndim,calc_likelihood,
            args=[fitfunc,um_data,mjy_data,mjy_errors,params,fixed],
            threads=nthreads,pool=pool)
      

      for i,res in enumerate(sampler.sample(initials,iterations=nburn,storechain=False)):
            if (i+1)%50==0: print("step ",i+1,'/',nburn)
            pos,prob,rstate = res
            
      sampler.reset()
      
      for i,res in enumerate(sampler.sample(pos,rstate0=rstate,iterations=nstep,storechain=True)):
            if (i+1)%50==0: print('Chain step ',i+1,'/',nstep)
            
            
      chains,lnlike = sampler.flatchain, sampler.flatlnprobability
      print(f'log likelihood: {lnlike}')
      # Overwrite values which were fixed:
      for i,p in enumerate(fixed):
            if p: chains[:,i] = p
      DL = 10*u.pc
      um_rest_ir = np.linspace(8.,1000.,200) * u.micron
      LIR,Mdust = [],[]
      print('getting LFIR')
      for i in tqdm(range(int(chains.shape[0]))):
            Sfit = fitfunc(um_rest_ir,*chains[i,:])
            LIR.append(get_luminosity(Lambdas=um_rest_ir, lum=Sfit).value)
            Mdust.append(get_dustmass(Sfit, um_rest_ir,chains[i,params.index('Td')], chains[i,params.index('beta')]))
      imax = np.argmax(lnlike)
      Sfit_best = fitfunc(um_rest_ir,*chains[imax,:])
      chains = np.c_[chains,LIR,Mdust]
      
      a = np.percentile(chains,[15.87,84.13],axis=0)
      stds = np.abs(a[0,:] - a[1,:])/2.
      
      return chains,stds,Sfit_best,um_rest_ir
