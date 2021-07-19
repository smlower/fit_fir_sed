import numpy as np
import pandas as pd
import emcee
import h5py
import sys
from sedfit_mcmc import *
from fitting_functions import greybody
from pd_utils import get_pd_sed

"""
Code to run sedfit_mcmc.py, originally written by Justin Spilker c. 2016
This takes a powderday SED (luminosity in erg/s) and fits a modified blackbody spectrum to it
We infer the amplitude of the blackbody fit (mostly to deal with weird units issues), dust temp, and beta
From these model fits, we get the distribution of inferred dust masses and total IR luminosities


This script takes 1 command line argument, which specifies which galaxy we are fitting. Run it by doing
$ python fit_seds.py 10
Which will fit galaxy 10

The actual fitting process doesn't take _too_ long, but feel free to adjust the steps parameter below. Increasing the # of workers or the # of iterations
can improve the fit if it's failing to converge


output.py has a simple plotting routine where you can plot the posteriors in a corner plot and the best fit SED
"""

galaxy = int(sys.argv[1])

#Model parameters. For greybody, these are [normalization,Td,L0(in m),beta, redshift]
fitfunc = greybody
print(f'using {fitfunc.__name__} model to fit FIR SED')
params = ['amp', 'Td', 'L0', 'beta']
fit_params = [False,False,100,False] #fixing peak FIR SED wavelength to 100 micron
#put our best fit SED plot here                          
plotfile = f'testfit_SED_galaxy{galaxy}.png'
#and put out fit results here
resultsfile = f'testfit_galaxy{galaxy}.hdf5'

#load powderday SED
print('loading pd SED')
wavelengths, flux = get_pd_sed(galaxy=galaxy, fir_only=True) #wavelengths in micron, flux in erg /s --> actually luminosity so bad label!
errors = flux * 0.3 #putting fake error bars on

#set up pool for multi processing. depending on how many cores you give your job, edit num_processes
num_processes = 15
print('running emcee')
pool = emcee.interruptible_pool.InterruptiblePool(processes=num_processes)
steps = [200,100,100]
chains,stds,sed,waves = sedfit_mcmc(fitfunc,wavelengths,flux,errors,params=params,fixed=fit_params,
              mcmcsteps=steps,pool=pool)
pool.terminate()

print('done. writing outputs')
outfile = h5py.File(resultsfile, 'w')
for parameter in params:
    outfile[parameter] = chains[:,params.index(parameter)]
outfile['LIR'] = chains[:,-2]
outfile['Mdust'] = chains[:,-1]
outfile['bestfit_sed'] = sed
outfile['bestfit_sed_wavelengths'] = waves 
outfile['observed_wavelengths'] = wavelengths
outfile['observed_sed'] = flux
outfile.close()
        
