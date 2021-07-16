import numpy as np
import pandas as pd
import emcee
from astropy.cosmology import Planck15
import pickle; import gzip
import sys
sys.path.append('/Users/jspilker/Research')
from sedfit_mcmc import *

# Fix parameters. For greybody, these are [normalization,Td,L0(in m),beta]
#  For greybody_powerlaw, [norm,Td,L0(in m),beta,alpha_pl]
fitfunc = greybody
fixpar = [False,False,100e-6,2.0]
#fitfunc = greybody_powerlaw
#fixpar = [False,False,False,2.0,2.0]

resultsfile = 'SEDs_allspecz_'+fitfunc.func_name+'L100.txt'

# note, use cvs revision 1.39 of flux_masterlist for lens model paper, 
# to avoid pre-emptively publishing new redshifts
#flux = pd.read_csv('lensmodelseds/flux_masterlist.dat',
#                delim_whitespace=True,header=7,index_col=0)
flux = pd.read_csv('/Users/jspilker/Research/SPTCVS/smg_analysis/photometry/flux_masterlist.dat',
          delim_whitespace=True,header=7,index_col=0)
                
specz = ~np.isnan(flux['z'])

photcols=['S_100','S_160','S_250','S_350','S_500','S_870','S220DEB','S150DEB','S_3mm']
errcols =['err_100','err_160','err_250','err_350','err_500','err_870','err_220.1','err_150.1','err_3mm']
wavelengths = np.array([100.,160.,250.,350.,500.,870.,1355.,1945.,3000.])
abserrs =     np.array([0.1,  0.1, 0.1, 0.1, 0.1, 0.1,0.15,  0.15, 0.2])

outfile = open(resultsfile,'w')
if fitfunc is greybody:
        header = "#Source \tzspec \tNbb \tdNbb \tTd \tdTd \tL0 \tdL0 \tBeta \tdBeta \tL_IR \tdL_IR \tL_FIR \tdL_FIR \tTpeak \tdTpeak\n"
        hdunit = "#------ \t------ \t------ \t------ \tK    \tK    \tm    \tm     \t------ \t------ \tLsun   \tLsun \tLsun \tLsun  \tK    \tK\n"
if fitfunc is greybody_powerlaw:
        header = "#Source \tzspec \tNbb \tdNbb \tTd \tdTd \tL0 \tdL0 \tBeta \tdBeta \talpha \tdalpha \tL_IR \tdL_IR \tL_FIR \tdL_FIR \tTpeak \tdTpeak\n"
        hdunit = "#------ \t------ \t------ \t------ \tK    \tK    \tm    \tm     \t------ \t------ \t------ \t------ \tLsun   \tLsun \tLsun \tLsun  \tK    \tK\n"
outfile.write(header+hdunit)

for source in flux[specz].index:
#for source in ['SPT0103-45']:
        print source
        # Discard wavelengths without data *and* non-detections
        mask = np.isfinite(np.ma.masked_invalid(np.ma.array(flux.loc[source,photcols],dtype=float)))
        # Discard *only* wavelengths without data, keep non-detections
        #mask = np.isfinite(np.ma.masked_invalid(np.ma.array(flux.loc[source,errcols],dtype=float)))
        thislambda = wavelengths[mask]
        thisabserr = abserrs[mask]
        thisphot = np.asarray(flux.loc[source,photcols][mask],dtype=float)
        thisphot[~np.isfinite(thisphot)] = 0. # non-detections, if kept in mask; send as zeros to fitter
        thiserr  = np.asarray(flux.loc[source,errcols][mask],dtype=float)
        thiserr  = np.sqrt(thiserr**2. + (thisabserr*thisphot)**2.)
        z = flux.ix[source,'z']
        thisfix = np.append(fixpar,z)
        thisl = (thislambda/(1+z)>40.) # ignore points at <40um rest
        thislambda=thislambda[thisl]
        thisphot=thisphot[thisl]
        thiserr=thiserr[thisl]

        plotfile = 'tmp/'+source+'_'+fitfunc.func_name+'L100.png'
        chainfile= 'tmp/'+source+'_'+fitfunc.func_name+'L100.pzip'
        sedfile  = 'tmp/'+source+'_'+fitfunc.func_name+'L100.txt'
        
        # emcee doesn't appear to garbage collect multiprocessing pool's after it's done, so our
        # grand loop leads quickly to "too many open files" errors. here we create a pool manually
        # so we can kill it ourselves.
        pool = emcee.interruptible_pool.InterruptiblePool(processes=6)
        
        chains,fits,stds,sed = sedfit_mcmc(fitfunc,thislambda,thisphot,thiserr,fixed=thisfix,
              plotfile=plotfile,mcmcsteps=[300,150,150],pool=pool)
        
        pool.terminate()

        # We're only doing spec-z sources, so of course those will all be fixed to their proper values.
        # Just remove z from the chains, fits, and uncertainties.
        chains,fits,stds = np.delete(chains,-4,1), np.delete(fits,-4), np.delete(stds,-4)
        
        # Dump the chains to a gzipped pickle file; this could also be hdf5 or whatever.
        # Read in with chains = pickle.load(gzip.open(chainfile))
        if chainfile is not None:
              with gzip.open(chainfile,'wb') as f: pickle.dump(chains,f)

        if fitfunc is greybody:
                hdp1 = "Fit parameters - Normalization, Td(K), L0(m), Beta, LIR(Lsun), LFIR(Lsun), T(peak)"
                hdp2 = "\n{0:.3f}, {1:.3f}, {2:.1e}, {3:.1f}, {4:.3e}, {5:.3e}, {6:.3f}"\
                .format(*fits)
                hdp3 = "\nUncertainties - \n{0:.3f}, {1:.3f}, {2:.1e}, {3:.1f}, {4:.3e}, {5:.3e}, {6:.3f}"\
                .format(*stds)
        if fitfunc is greybody_powerlaw:
                hdp1 = "Fit parameters - Normalization, Td(K), L0(m), beta, alpha, LIR(Lsun), LFIR(Lsun), T(peak)"
                hdp2 = "\n{0:.3f}, {1:.3f}, {2:.1e}, {3:.1f}, {4:.1f}, {5:.3e}, {6:.3e}, {7:.3f}"\
                .format(*fits)
                hdp3 = "\nUncertainties - \n{0:.3f}, {1:.3f}, {2:.1e}, {3:.1f}, {4:.1f}, {5:.3e}, {6:.3e}, {7:.3f}"\
                .format(*stds)
        hdp4 = "\nSED, 8-1000um rest frame - \nLambda(um) \tSnu(mJy)"

        header = hdp1 + hdp2 + hdp3 + hdp4
        print hdp1+hdp2+hdp3
        np.savetxt(sedfile,np.array(zip((1+z)*np.linspace(8,1000,400),sed)),fmt='%.3f',delimiter='\t',header=header)

        res = [source,z]
        res.extend(q for q in [elem for sublist in zip(fits,stds) for elem in sublist])
        if fitfunc is greybody: tofilestr = "{0:7s} \t{1:.3f} \t{2:.2e} \t{3:.2e} \t{4:.3f} \t{5:.3f} \t{6:.2e} \t{7:.2e} \t{8:.2f} \t{9:.2f} \t{10:.3e} \t{11:.3e} \t{12:.3e} \t{13:.3e} \t{14:.3f} \t{15:.3f}\n".format(*res)
        if fitfunc is greybody_powerlaw: tofilestr = "{0:7s} \t{1:.3f} \t{2:.2e} \t{3:.2e} \t{4:.3f} \t{5:.3f} \t{6:.2e} \t{7:.2e} \t{8:.2f} \t{9:.2f} \t{10:.2f} \t{11:.2f} \t{12:.3e} \t{13:.3e} \t{14:.3e} \t{15:.3e} \t{16:.3f} \t{17:.3f}\n".format(*res)
        outfile.write(tofilestr)
        
outfile.close()
        