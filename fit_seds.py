import numpy as np
import pandas as pd
import emcee
from astropy.cosmology import Planck15
import h5py
import sys
from sedfit_mcmc import *
from fitting_functions import greybody
from pd_utils import get_pd_sed

galaxy = int(sys.argv[1])

#Model parameters. For greybody, these are [normalization,Td,L0(in m),beta, redshift]
fitfunc = greybody
print(f'using {fitfunc.__name__} model to fit FIR SED')
fixpar = [False,False,100e-6,False, 0] #fixing peak FIR SED wavelength to 100 micron
thisfix = [True, True, False, True, False]
#put our best fit SED plot here                          
plotfile = f'testfit_SED_galaxy{galaxy}.png'
#and put out fit results here
resultsfile = f'testfit_galaxy{galaxy}.hdf5'

#load powderday SED
print('loading pd SED')
wavelengths, flux = get_pd_sed(galaxy=galaxy, fir_only=True) #wavelengths in micron, flux in mJy
errors = flux * 0.03 #putting fake error bars on

#set up pool for multi processing. depending on how many cores you give your job, edit num_processes
num_processes = 8
print('running emcee')
pool = emcee.interruptible_pool.InterruptiblePool(processes=num_processes)
chains,fits,stds,sed = sedfit_mcmc(fitfunc,wavelengths,flux,errors,fixed=thisfix,
              plotfile=plotfile,mcmcsteps=[300,150,150],pool=pool)
pool.terminate()

print('done. writing outputs')
outfile = h5py.File(resultsfile, 'w')
outfile['chains'] = chains
outfile['fits'] = fits
outfile['stds'] = stds
outfile['sed'] = sed
outfile.close()
        
