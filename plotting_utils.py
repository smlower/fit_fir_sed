import numpy as np
from get_luminosity import *
from astropy.cosmology import Planck15
import matplotlib.pyplot as pl
pl.ioff()
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import scipy.ndimage
from copy import deepcopy

# Contains a few routines for making output plots, including
# triangle plots of parameter covariances and the best-fit SED.

__all__ = ['marginalize_2d','marginalize_1d','plot_sed','text_summary','triangleplot',
      'make_outputplot']

def marginalize_2d(x,y,axobj,*args,**kwargs):
      """
      Routine to plot 2D confidence intervals between two parameters given arrays
      of MCMC samples.

      Inputs:
      x,y:
            Arrays of MCMC chain values.
      axobj:
            A matplotlib Axes object on which to plot.
      extent:
            List of [xmin,xmax,ymin,ymax] values to be used as plot axis limits.
            If not provided, something sensible is chosen.
      bins: 
            Number of bins to put the chains into.
      levs:
            Contour levels, in sigma. Default is 1,2,3sigma regions.
      
      Returns:
      axobj:
            The same axis object passed here, now with the regions plotted.
      """

      # Get values of various possible kwargs
      bins = kwargs.pop('bins',50)
      levs = kwargs.pop('levs',[1.,2.,3.])
      extent = kwargs.pop('extent',[x.min(),x.max(),y.min(),y.max()])
      cmap = kwargs.pop('cmap','Greys')

      cmap = cm.get_cmap(cmap.capitalize())
      cmap = cmap(np.linspace(0,1,np.asarray(levs).size))
      
      Xbins = np.linspace(extent[0],extent[1],bins+1)
      Ybins = np.linspace(extent[2],extent[3],bins+1)

      # Bin up the samples. Will fail if x or y has no dynamic range
      try:
            H,X,Y = np.histogram2d(x.flatten(),y.flatten(),bins=(Xbins,Ybins))
      except ValueError: return ValueError("One of your columns has no dynamic range... check it.")

      # Generate contour levels, sort probabilities from most to least likely
      V = 1.0 - np.exp(-0.5*np.asarray(levs)**2.)
      # Here we slightly smooth the contours to account for the finite number
      #  of MCMC samples. can adjust the 0.7 below, but too small is artificial
      #  and looks like shit.
      H = scipy.ndimage.filters.gaussian_filter(H,0.2*np.log10(x.size))
      Hflat = H.flatten()
      inds = np.argsort(Hflat)[::-1]
      Hflat = Hflat[inds]
      sm = np.cumsum(Hflat)
      sm /= sm[-1]

      # Find the probability levels that encompass each sigma's worth of likelihood
      for i,v0 in enumerate(V):
            try: V[i] = Hflat[sm <= v0][-1]
            except: V[i] = Hflat[0]

      V = V[::-1]
      clevs = np.append(V,Hflat.max())
      X1, Y1 = 0.5*(X[1:] + X[:-1]), 0.5*(Y[1:]+Y[:-1])

      if kwargs.get('plotfilled',True): axobj.contourf(X1,Y1,H.T,clevs,colors=cmap)
      axobj.contour(X1,Y1,H.T,clevs,colors=kwargs.get('colors','k'),linewidths=kwargs.get('linewidths',1.5),\
            linestyles=kwargs.get('linestyles','solid'))
      axobj.set_xlim(extent[0],extent[1])
      axobj.set_ylim(extent[2],extent[3])
      
      return axobj
      
def marginalize_1d(x,axobj,*args,**kwargs):
      """
      Plot a histogram of x, with a few tweaks for corner plot pleasantry.

      Inputs:
      x:
            Array of MCMC samples to plot up.
      axobj:
            Axes object on which to plot.
      """

      bins = kwargs.pop('bins',50)
      extent = kwargs.pop('extent',[x.min(),x.max()])
      fillcolor = kwargs.pop('color','gray')

      axobj.hist(x,bins=bins,range=extent,histtype='stepfilled',color=fillcolor)
      axobj.yaxis.tick_right()
      pl.setp(axobj.get_yticklabels(),visible=False)
      axobj.set_xlim(extent[0],extent[1])
      
      return axobj
      

def plot_sed(um_rest,Sfit,um_data,mjy_data,mjy_errors,axobj,*args,**kwargs):
      """
      Plot an SED
      """
      sedcolor = kwargs.pop('sedcolor','b')
      
      axobj.set_axis_on()
      axobj.plot(um_rest,Sfit,color=sedcolor,ls='-')
      axobj.set_xscale('log'); axobj.set_yscale('log')
      axobj.set_xlabel('$\\lambda_{obs}$, $\\mu m$')
      axobj.set_ylabel('S$_{\\nu}$, mJy')
      axobj.set_xlim(1.1**um_rest.max(),0.8*(1+z)*um_rest.min())
      axobj.set_ylim(0.5*mjy_data[mjy_data>0.].min(),2*np.max(Sfit))
      
      altax = axobj.twiny()
      altax.set_xlim(axobj.get_xlim()[0],axobj.get_xlim()[1])
      altax.set_ylim(axobj.get_ylim()[0],axobj.get_ylim()[1])
      altax.set_xscale('log'); altax.set_xlabel('$\\lambda_{rest}$, $\\mu m$')
      
      # Plot the data points, and plot upper limits if they exist
      for i,point in enumerate(um_data):
            if point<200: color = 'c' # PACS points
            elif point<= 500: color = 'g' # SPIRE points (also SABOCA...)
            else: color = 'r'
            
            if mjy_data[i] > 0:
                  axobj.errorbar(point,mjy_data[i],yerr=mjy_errors[i],color=color,\
                    marker='o')
            else: # Draw upper limits, but make errobar sizes consistent on log-log
                  ylims,xlims = axobj.get_ylim(), axobj.get_xlim()
                  yloc = -(np.log10(ylims[0]) - np.log10(3*mjy_errors[i]))/(np.log10(ylims[1])-np.log10(ylims[0]))
                  xloc = (np.log10(point)-np.log10(xlims[0]))/(np.log10(xlims[1])-np.log10(xlims[0]))
                  axobj.errorbar(xloc,yloc,yerr=[[0.07],[0.0]],uplims=True,color=color,marker='o',\
                    ecolor='grey',mec='grey',transform=axobj.transAxes)

      return axobj

def text_summary(samples,labels,axobj):
      """
      Write some parameter summary text to the axobj.
      """      
      
      #axobj.set_axis_on()
      axobj.text(-0.8,0.9,'Chain parameters:',fontsize='xx-large',transform=axobj.transAxes)
      tloc = 0.
      for par in range(0,samples.shape[1]):
            x = deepcopy(samples[:,par])
            xstd = np.ediff1d(np.percentile(x,[15.87,84.13]))[0]/2.
            xmed = np.median(x)
            if 'lambda' in labels[par]: xmed,xstd = 1e6*xmed,1e6*xstd
            if 'L$_\\odot$' in labels[par]: xmed,xstd = xmed/1e12, xstd/1e12
            axobj.text(-0.8,0.7-tloc,'{0:10s} = {1:.2f} $\\pm$ {2:.2f}'.
              format(labels[par],xmed,xstd),fontsize='xx-large',transform=axobj.transAxes)
            tloc += 0.18
      
      return axobj


def triangleplot(samples,labels):
      """
      Assemble the MCMC samples into the usual triangle plot format.
      
      Parameters:
      samples: 2-D array
            Array of MCMC samples, of shape (Nsamples,Nparameters)
      labels: list of strings
            What to label the axes. Latex math-mode is okay.
      
      Returns:
      f,axarr: matplotlib figure object and array of Axes objects of shape (Nparams,Nparams)
      """
      
      f,axarr = pl.subplots(samples.shape[1],samples.shape[1],figsize=(3*samples.shape[1],3*samples.shape[1]))
      
      for row in range(0,samples.shape[1]):
            for col in range(0,samples.shape[1]):
                  # been burned too many times by unintentionally altering arrays
                  x,y = deepcopy(samples[:,col]), deepcopy(samples[:,row])
                  
                  # Shield ourselves against nans or infinities.
                  x = x[np.isfinite(x) & np.isfinite(y)]
                  y = y[np.isfinite(x) & np.isfinite(y)]
                  
                  # do some unit conversions for the sake of our collective sanity
                  if 'lambda' in labels[col]: x*=1e6 # convert a wavelength to um from m
                  if 'lambda' in labels[row]: y*=1e6
                  if 'L$_\\odot$' in labels[col]: x /= 1e12 # divide down luminosity
                  if 'L$_\\odot$' in labels[row]: y /= 1e12
                  
                  # figure out some sensible axis limits
                  xstd = np.ediff1d(np.percentile(x,[15.87,84.13]))[0]/2
                  ystd = np.ediff1d(np.percentile(y,[15.87,84.13]))[0]/2
                  xmin,xmax = np.median(x)-6*xstd, np.median(x)+6*xstd
                  ymin,ymax = np.median(y)-6*ystd, np.median(y)+6*ystd
            
                  
                  if row>col:
                        try: marginalize_2d(x,y,axarr[row,col],\
                              extent=[xmin,xmax,ymin,ymax],bins=max(np.floor(x.size/1000),50))
                        except ValueError:
                              print(labels[row],labels[col])
                              raise ValueError("One of the columns has no dynamic range")
                        if col>0: pl.setp(axarr[row,col].get_yticklabels(),visible=False)
                        else: axarr[row,col].set_ylabel(labels[row],fontsize='x-large')
                        if row<axarr.shape[0]-1: pl.setp(axarr[row,col].get_xticklabels(),visible=False)
                        else: axarr[row,col].set_xlabel(labels[col],fontsize='x-large')
                        axarr[row,col].xaxis.set_major_locator(MaxNLocator(5))
                        axarr[row,col].yaxis.set_major_locator(MaxNLocator(5))
                  elif row==col:
                        marginalize_1d(x,axarr[row,col],extent=[xmin,xmax],bins=max(np.floor(x.size/1000),50))
                        if row==axarr.shape[0]-1: axarr[row,col].set_xlabel(labels[col],fontsize='x-large')
                        if col<axarr.shape[0]-1: pl.setp(axarr[row,col].get_xticklabels(),visible=False)
                        axarr[row,col].xaxis.set_major_locator(MaxNLocator(5))
                        
                  else:
                        axarr[row,col].set_axis_off()
                        
      return f, axarr


def make_outputplot(plotfile,samples,labels,plotsed=True,um_rest=None,
      Sfit=None,um_data=None, mjy_data=None,mjy_errors=None):
      """
      Assemble the whole output thing.
      """
      
      f,axarr = triangleplot(samples,labels)
      if plotsed: trcax = plot_sed(um_rest,Sfit,um_data,mjy_data,mjy_errors,axarr[0,-1])
      textax = text_summary(samples,labels,axarr[0,-2])
      
      f.subplots_adjust(left=0.1,right=0.97,bottom=0.1,top=0.95,hspace=0,wspace=0)
      f.savefig(plotfile)
      pl.close()
      
