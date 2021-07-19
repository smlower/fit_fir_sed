# fit_fir_sed
Fit far-IR powderday SED with emcee. Originally developed by Justin Spilker c. 2015-2016. Retro-fitted to use data from cosmological simulations

Install prereqs:
  1. emcee: https://github.com/dfm/emcee
  2. corner: https://github.com/dfm/corner.py
  3. h5py: conda install h5py
  4. tqdm: conda install -c conda-forge tqdm (just for timing for loops)


This code takes a powderday SED (luminosity in erg/s) and fits a modified blackbody spectrum to it                                                          
We infer the amplitude of the blackbody fit (mostly to deal with weird units issues), dust temp, and beta                                                  
From these model fits, we get the distribution of inferred dust masses and total IR luminosities                                                                                                                                                                                                                      
The fit_seds.py script takes 1 command line argument, which specifies which galaxy we are fitting. Run it by doing                                                       
   $ python fit_seds.py 10                                                                                                                                  
                                                                                                                                                              
The actual fitting process doesn't take _too_ long, but feel free to adjust the steps parameter below. Increasing the # of workers or the # of iterations canimprove the fit if it's failing to converge                                                                                                               
                                                                                                                                                             output.py has a simple plotting routine where you can plot the posteriors in a corner plot and the best fit SED 
  
 
