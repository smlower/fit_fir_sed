import numpy as np
import h5py
import pandas as pd
import corner
from corner import quantile
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": "STIXGeneral",
    "mathtext.fontset" : "cm",
    #"legend.title_fontsize": 20
})
import matplotlib.pyplot as plt


results = h5py.File('/home/s.lower/scripts/mcmc_sed_code/testfit_galaxy10.hdf5')

print('List of results:')
for key in results.keys():
    print(f'        {key}')


print('Making corner plot of posterior distributions')

data = {'T$_\mathrm{dust}$': results['Td'],'$\\beta$': results['beta'], '$\log(\mathrm{L}_{IR})$' : np.log10(results['LIR']), '$\log(\mathrm{M}_\mathrm{dust})$': np.log10(results['Mdust'])}
df = pd.DataFrame(data)


corner.corner(df,labels=list(data.keys()),
                       quantiles=[0.16, 0.5, 0.84], title_fmt=None,
                       show_titles=True, title_kwargs={"fontsize": 25}, label_kwargs={'fontsize': 25}, labelpad=0.15)

plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print('Making best fit SED plot')
plt.loglog(results['observed_wavelengths'], results['observed_sed'], color='black', lw=3, label='True SED')
plt.loglog(results['bestfit_sed_wavelengths'], results['bestfit_sed'], color='tomato', lw=2, label='Best Fit SED')
plt.ylabel('Luminosity [$L_{\odot}$]')
plt.xlabel('Wavelength [$\mu$m]')
plt.legend()
plt.savefig('sed_plot.png')
