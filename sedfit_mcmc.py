import numpy as np
import emcee
from scipy.special import erfc
from fitting_functions import *
from get_luminosity import *
from get_dustmass import *
from plotting_utils import *
from astropy.cosmology import Planck15
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
                plotfile='triangle.png',mcmcsteps=[100,100,100],nthreads=2,pool=None):

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
      
      print("Running sampler")
      for i,res in enumerate(sampler.sample(initials,iterations=nburn,storechain=False)):
            if (i+1)%50==0: print("step ",i+1,'/',nburn)
            pos,prob,rstate = res
            
      sampler.reset()
      
      for i,res in enumerate(sampler.sample(pos,rstate0=rstate,iterations=nstep,storechain=True)):
            if (i+1)%50==0: print('Chain step ',i+1,'/',nstep)
            
            
      print("Mean acceptance fraction: {0:.3f}".format(100*sampler.acceptance_fraction.mean()))
      
      chains,lnlike = sampler.flatchain, sampler.flatlnprobability
      
      # Overwrite values which were fixed:
      for i,p in enumerate(fixed):
            if p: chains[:,i] = p
      DL = 10*u.pc
      um_rest_ir = np.linspace(8.,1000.,400) * u.micron
      fir = ((um_rest_ir.value>42.5) & (um_rest_ir.value<122.5))
      LIR,LFIR,Tpeak,Mdust = [],[],[], []
      print('getting LFIR')
      for i in range(chains.shape[0]):
            Sfit = fitfunc(um_rest_ir,*chains[i,:])
            LIR.append(get_luminosity(Lambdas=um_rest_ir, lum=Sfit))
            LFIR.append(get_luminosity(Lambdas=um_rest_ir[fir], lum=Sfit[fir]))
            Tpeak.append(3e14/um_rest_ir[Sfit.value.argmax()]/58.79e9) # this version for Sfit in per-Hz mJy, courtesy dpm
            Sfit_mJy = Sfit/((4. * np.pi * u.sr) * (10*u.pc).to(u.cm)**2 * (c / um_rest_ir.to(u.m)))
            Mdust.append(get_dustmass((Sfit_mJy.value)*1e-13 *u.mJy, chains[i,params.index('Td')], chains[i,params.index('beta')]))
      chains = np.c_[chains,LIR,LFIR,Tpeak,Mdust]
      
      fits = np.median(chains,axis=0)
      Sfit = []
      for i in (um_rest_ir):
                        Sfit.append(fitfunc(i,*fits[:len(params)]))
      a = np.percentile(chains,[15.87,84.13],axis=0)
      stds = np.abs(a[0,:] - a[1,:])/2.
      if plotfile is not None:
            freepars = np.where(np.asarray(fixed)==0)[0]
            freepars = np.append(freepars,[-1, -4]) # plot luminosities and mass posteriors
            if fitfunc.__name__ == 'greybody':
                  labels = np.asarray(['Amplitude','$T_{dust}$','$\\lambda_0$ ($\\mu$m)','$\\beta$','L$_{IR}$ (10$^{12}$L$_\\odot$)','L$_{FIR}$ (10$^{12}$L$_\\odot$)','T$_{peak}$', 'M$_{dust}$'])
            elif fitfunc.__name__ == 'greybody_powerlaw':
                  labels = np.asarray(['Amplitude','$T_{dust}$','$\\lambda_0$ ($\\mu$m)','$\\beta$','$\\alpha_{pl}$','z','L$_{IR}$ (10$^{12}$L$_\\odot$)','L$_{FIR}$ (10$^{12}$L$_\\odot$)','T$_{peak}$'])    
            make_outputplot(plotfile,chains[:,freepars],labels[freepars],plotsed=True,um_rest=um_rest_ir,
                            Sfit=Sfit,um_data=um_data,mjy_data=mjy_data,mjy_errors=mjy_errors, z=z)
      
      return chains,fits,stds,Sfit
