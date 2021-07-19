import numpy as np
from sedpy.observate import load_filters
import h5py
from hyperion.model import ModelOutput
import astropy.units as u
import astropy.constants as constants


def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

def get_pd_sed(galaxy, fir_only=False):
    pd_dir = '/orange/narayanan/s.lower/simba/pd_runs/snap305/'
    m = ModelOutput(pd_dir+f'/snap305.galaxy{galaxy}.rtout.sed')
    wave, flx = m.get_sed(inclination=0, aperture=-1)
    #get mock photometry                                                                                                                                     
    wave  = np.asarray(wave)*u.micron #wav is in micron                                                                                                      
    wav = wave[::-1].to(u.AA)
    flux = np.asarray(flx)[::-1]*u.erg/u.s
    #do we only want FIR SED?
    if fir_only:
        fir = find_nearest(wav.to(u.micron).value, 50) #we'll define FIR as the part of the SED spanning from 50 micron to 1000 micron
        fir_wave = wav[fir:].to(u.micron)
        fir_flux = flux[fir:]
    return fir_wave, fir_flux

