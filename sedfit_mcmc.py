import numpy as np
import emcee
from scipy.special import erfc
from fitting_functions import *
from get_luminosity import *
from plotting_utils import *
from astropy.cosmology import Planck15
import astropy.units as u
__all__ = ['calc_prior','calc_likelihood','sedfit_mcmc']

def calc_prior(p,fitfunction):
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
            # param ordering is [amp,Tdust,L0,beta,z]
            if p[0] < 0: plike -= np.inf
            if (p[1]<15) or (p[1]>300): plike -= np.inf
            if (p[2]<50e-6) or (p[2]>500e-6): plike -= np.inf
            if (p[3]<0.) or (p[3]>4.): plike -= np.inf
            if (p[4]<0.) or (p[4]>10.): plike -= np.inf
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
      

def calc_likelihood(p,fitfunction,um_data,mjy_data,mjy_errors,fixed,
      lensmodelpars=None,fracbw=None,cosmo=Planck15):
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
      lensmodelpars: six-element list
            Optionally, constraints on the source size and brightness from
            independent lens models can be incorporated. To be used, should
            have six elements: the function used to evaluate the lens model
            portion of the model, the wavelength the model was fit at (in um),
            the source flux density and uncertainty at that wavelength (in mJy),
            and the source effective radius and uncertainty (in kpc).
      fracbw: array of length um_data
            To quasi-account for the instrumental response, the model can
            be averaged at the central wavelength, as well as +/- the fracbw
            of that point. Should be in the form, eg, ['10%','15%','10%',...]
      cosmo: astropy cosmology FlatLambdaCDM instance
            Default cosmology is Planck2015 (Paper XIII), but any other astropy
            cosmology thing can be used instead.
            
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
      priorlik = calc_prior(p,fitfunction)
      # if prior likelihood is -inf, don't need to do anything else
      if ~np.isfinite(priorlik): return priorlik 
      
      if fracbw is None: fracbw = ['']*um_data.size # don't do anything
      
      modeldata = np.zeros(um_data.size)
      for i,point in enumerate(um_data):
            if '%' in fracbw[i]:
                  percent = float(fracbw[i][:-1])/100.
                  bandwidth = 1e-6*np.linspace((1-percent)*point,(1+percent)*point,3)
                  modeldata[i] = np.average(fitfunction(bandwidth/(1+p[-1]),*p,cosmo=cosmo))
            else: modeldata[i] = fitfunction(1e-6*point/(1+p[-1]),*p,cosmo=cosmo)
      
      # Calculate likelihood for detected points, usual chi-squared definition
      det = mjy_data > 0
      detlik = np.sum((mjy_data[det] - modeldata[det])**2./(mjy_errors[det])**2.)
      
      # ... and the contribution for undetected points. see monhanty+13, appendix c
      undetlik = np.sum(0.5*erfc((1*mjy_errors[~det] - modeldata[~det])/mjy_errors[~det]))
      
      if lensmodelpars is not None:
            # also calculate the likelihood of the proposed parameters being correct,
            # given the information we know from the lens model.
            # here we sum uncertainties caused by the lens model size with those from
            # the model itself. this won't be strictly correct because they're probably
            # correlated, but this is the worst-case scenario.
            modelfunc,lambdamodel,mjy_model,dmjy_model,reff_model,dreff_model = lensmodelpars
            Smodel = 1e29*modelfunc(1e-6*lambdamodel/(1+z),reff_model,*p,cosmo=cosmo)
            Slow   = 1e29*modelfunc(1e-6*lambdamodel/(1+z),reff_model-dreff_model,*p,cosmo=cosmo)
            Shigh  = 1e29*modelfunc(1e-6*lambdamodel/(1+z),reff_model+dreff_model,*p,cosmo=cosmo)
            dS_reff = np.average([Shigh-Smodel,Smodel-Slow])
            totdS = np.sqrt(dmjy_model**2. + dS_reff**2.)
            lenslik = (mjy_model-Smodel)**2. / (totdS**2.)
      else: lenslik = 0.
      
      freepars = np.where(np.asarray(fixed)==0)[0]
      dof = float(um_data.size-len(freepars))
      return -(1./dof)*(priorlik + detlik + undetlik + lenslik)
      
      
      
def sedfit_mcmc(fitfunc,um_data,mjy_data,mjy_errors,lensmodelpars=None,p0=None,fracbw=None,fixed=None,
      cosmo=Planck15,plotfile='triangle.png',mcmcsteps=[100,100,100],nthreads=2,pool=None):

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
      lensmodelpars: six-element list
            Optionally, constraints on the source size and brightness from
            independent lens models can be incorporated. To be used, should
            have six elements: the function used to evaluate the lens model
            portion of the model, the wavelength the model was fit at (in um),
            the source flux density and uncertainty at that wavelength (in mJy),
            and the source effective radius and uncertainty (in kpc).
      p0: array 
      fixed: array
            Array describing which parameter(s) should be held fixed
            during fitting. Should be same size as ::p::. If the i'th parameter
            is free, fixed[i]=False. If the i'th parameter is held fixed
            to the value x, fixed[i]=x. For example, if fitfunction is the
            greybody and we want to fix beta and z_spec, we should have
            fixed = [False,False,False,beta,z_spec].
      fracbw: array of length um_data
            To quasi-account for the instrumental response, the model can
            be averaged at the central wavelength, as well as +/- the fracbw
            of that point. Should be in the form, eg, ['10%','15%','10%',...]
      cosmo: astropy cosmology FlatLambdaCDM instance
            Default cosmology is Planck2015 (Paper XIII), but any other astropy
            cosmology thing can be used instead.
      
      """
#fitfunc.func_name     
      if p0 is None:
            if fitfunc.__name__ == 'greybody': p0 = np.array([1.,40.,150e-6,2,3])
            elif  fitfunc.__name__ == 'greybody_powerlaw': p0 = np.array([1.,40.,150e-6,2,2,3]) 
      else: p0 = np.asarray(p0)
      
      if fracbw is None:
            fracbw = list('10%' for d in um_data)
            
      if fixed is None:
            fixed = np.array([False for p in p0])
      
      # Some setup for emcee
      nwalkers,nburn,nstep = mcmcsteps
      ndim = p0.size
      initials = np.abs(emcee.utils.sample_ball(p0,0.1*p0,int(nwalkers)))
      
      sampler = emcee.EnsembleSampler(nwalkers,ndim,calc_likelihood,
            args=[fitfunc,um_data,mjy_data,mjy_errors,fixed,lensmodelpars,fracbw,cosmo],
            threads=nthreads,pool=pool)
      
      print("Running burn-in...")
      for i,res in enumerate(sampler.sample(initials,iterations=nburn,storechain=False)):
            if (i+1)%50==0: print("Burn-in step ",i+1,'/',nburn)
            pos,prob,rstate = res
            
      sampler.reset()
      
      print("Done. Running chains...")
      for i,res in enumerate(sampler.sample(pos,rstate0=rstate,iterations=nstep,storechain=True)):
            if (i+1)%50==0: print('Chain step ',i+1,'/',nstep)
            
      print("Mean acceptance fraction: {0:.3f}".format(100*sampler.acceptance_fraction.mean()))
      
      chains,lnlike = sampler.flatchain, sampler.flatlnprobability
      
      # Overwrite values which were fixed:
      for i,p in enumerate(fixed):
            if p: chains[:,i] = p
      z = fixed[-1]      
      DL = 10*u.pc
      um_rest_ir = np.linspace(8.,1000.,400)
      fir = ((um_rest_ir>42.5) & (um_rest_ir<122.5))
      LIR,LFIR,Tpeak = [],[],[]
      for i in range(chains.shape[0]):
            Sfit = fitfunc(1e-6*um_rest_ir,*chains[i,:],cosmo=cosmo)
            LIR.append(get_luminosity(Lambdas=um_rest_ir*(1+chains[i,-1]),Snu=Sfit,distance=DL))
            LFIR.append(get_luminosity(Lambdas=um_rest_ir[fir]*(1+chains[i,-1]),Snu=Sfit[fir],distance=DL))
            Tpeak.append(3e14/um_rest_ir[Sfit.argmax()]/58.79e9) # this version for Sfit in per-Hz mJy, courtesy dpm
      
      chains = np.c_[chains,LIR,LFIR,Tpeak]
      
      fits = np.median(chains,axis=0)
      Sfit = fitfunc(1e-6*um_rest_ir,*fits[:-3],cosmo=cosmo)
      a = np.percentile(chains,[15.87,84.13],axis=0)
      stds = np.abs(a[0,:] - a[1,:])/2.
      
      if plotfile is not None:
            freepars = np.where(np.asarray(fixed)==0)[0]
            freepars = np.append(freepars,fits.size-2) # always plot LFIR
            if fitfunc.__name__ == 'greybody':
                  labels = np.asarray(['Amplitude','$T_{dust}$','$\\lambda_0$ ($\\mu$m)','$\\beta$','z','L$_{IR}$ (10$^{12}$L$_\\odot$)','L$_{FIR}$ (10$^{12}$L$_\\odot$)','T$_{peak}$'])
            elif fitfunc.__name__ == 'greybody_powerlaw':
                  labels = np.asarray(['Amplitude','$T_{dust}$','$\\lambda_0$ ($\\mu$m)','$\\beta$','$\\alpha_{pl}$','z','L$_{IR}$ (10$^{12}$L$_\\odot$)','L$_{FIR}$ (10$^{12}$L$_\\odot$)','T$_{peak}$'])    
            make_outputplot(plotfile,chains[:,freepars],labels[freepars],plotsed=True,um_rest=um_rest_ir,
                            Sfit=Sfit,um_data=um_data,mjy_data=mjy_data,mjy_errors=mjy_errors, z=z)
      
      return chains,fits,stds,Sfit
