import csv
import numpy as np
import matplotlib as mpl
#mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches
from astropy.io import fits
from astropy.stats import mad_std
from astropy import stats
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.integrate import simps
import h5py
from astrodendro import Dendrogram
import peakutils.peak
import datetime
import multiprocessing as mp
import sys
from scipy.optimize import fsolve
from sympy import Eq, Symbol, solve, symbols
import warnings 
import scipy.constants as C
import pandas as pd

import warnings
warnings.simplefilter('ignore')

vel_disp_list = []
def pix_to_pc(pix, dpc, arcsec_per_pix):
	return  pix*(arcsec_per_pix/206265)*dpc

def column_density_13(t12, t13, freq13_Hz):
        #constants
	tul_12 = 11.06/7 #K for 12CO(2-1)
	tul_13 = 10.58 #K for 13CO(2-1)
#	gu = 0.06968198444 #calculated using firecracker paper and known constants, could also be 1053859.15489
        #gu = 2J +1 
        # J = 2-1
	gu = 5
	#nu_naught = 220.399 #GHz
	nu_naught = freq13_Hz
	tcmb = 2.73
	B = 2.644 #K for 13CO
	Aul = 6.03*10**(-7) #http://www.ifa.hawaii.edu/users/jpw/classes/star_formation/reading/molecules.pdf
	
        #assume 12CO(2-1) is optically thick and use firecracker eqn 2 to find tex
	tex = tul_12/np.log((tul_12/t12)+1)
	print("tex max", np.nanmax(tex))

        #assume 13CO(2-1) is optically thin and use eqn 2 to calculate its optical depth
	tau_13 = -np.log(1-(t13/tul_13)*(((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1)))
	print("tau 13:", np.nanmax(tau_13))

	mom0_tau = np.nansum(tau_13, axis=0)
	mom8_tau = np.nanmax(tau_13, axis=0)
	tau_prof = np.nanmean(tau_13, axis=(1,2))
	tau_prof = np.where(np.isnan(tau_prof), 0.0, tau_prof)
	print("tau prof max:", np.nanmax(tau_prof))
	mom0_tex = np.nansum(tex, axis=0)
	mom8_tex = np.nanmax(tex, axis=0)
	tex_prof = np.nanmean(tex, axis=(1,2))
	tex_prof = np.where(np.isnan(tex_prof), 0.0, tex_prof)
	print("tex prof max", np.nanmax(tex_prof))
	print("tex prof shape", tex_prof.shape)

	ax=plt.subplot(111)
	plt.imshow(mom0_tau, origin='lower')
	plt.colorbar()
	plt.plot(1,2, color='red')
	plt.title('mom0 tau')
	plt.show()	
	
	ax=plt.subplot(111)
	plt.imshow(mom0_tex, origin='lower')
	plt.colorbar()
	plt.title('mom0	tex')
	plt.show()
	
	#integrate tau
#	fwhm13 = get_vels(propfile12, propfile13)[ncl]*np.sqrt(8*np.log(2.0))
#	T0 = C.h*freq13_GHz/C.k
#	Nu = 4*np.pi**1.5*freq13_Hz**3 / (C.c**3*Aul*np.sqrt(np.log(2.0)))
#	Nu /= np.exp(T0/tex) -1.0
#	Nu *= tau_13*fwhm13
#	area = simps(tau_prof)
#	print("area:", area, area.shape)
		
        #calculate column density for 13CO
#	Q = (tex/B)+(1/3)
#	print("q:", np.nanmax(Q))
#	tex = tex_prof
#	Q = (tex_prof/B)+(1/3)
#	N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*C.c**2)) * Q/(1-np.exp(-tul_13/tex)) *area
#	print("n max", np.nanmax(N_13))
#	print("n", N_13)
#	print("n shape", N_13.shape)

#	N_13 = Nu*Q*(1.0/gu)*np.exp(C.h*freq13_Hz/(C.k*tex))
#	print("ntot:", N_13)
#	print("nmax", np.nanmax(N_13))
#	return N_13


def calc_mass(N_13, pix_area):
        # use ratios of 13CO to estimate total density
	H_to_13CO = 2*10**6 # abundance ratio of H_2 / 13CO
	total_surface_density = 1.3*N_13*H_to_13CO
	print("total surface density:", total_surface_density)
	
	#convert from cm^-2 ro pc^-2
	density_pc = total_surface_density*(9.52140614*10**36)*1.3	

	total_mass = np.nansum(pix_area*density_pc)
	print("total mass", total_mass)
	print("m/m sun", total_mass/(1.989*10**30))
	return(total_mass)
        #calculate total mass per clump
        #sum all mass per pixel to get total mass
        #convert from pixel area to pc?
        #in firecracker paper pixel area = 2.23 pc^2

def get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13):
	clumpdata12 = np.where(clumps13==ncl, COtemp12, np.nan)
	clumpdata12 = np.where((clumps12 !=np.nan).any(), COtemp12, np.nan)
	clumpdata12 = np.where(clumpdata12 > nsig*rms_K12, COtemp12, np.nan)

	clumpdata13 = np.where(clumps13==ncl, COtemp13,	np.nan)
	clumpdata13 = np.where(clumpdata13 > nsig*rms_K13, COtemp13, np.nan)
	return(clumpdata12, clumpdata13)

def profiles():
	tau_13 = find_tau_tex(t12, t13, freq13_Hz)[0]
	tex = find_tau_tex(t12, t13, freq13_Hz)[1]

	mom0_tau = np.nansum(tau_13, axis=0)
	mom8_tau = np.nanmax(tau_13, axis=0)
	tau_prof = np.nanmean(tau_13, axis=(1,2))

	tex_prof = np.nanmean(tex, axis=(1,2))
	return (tau_prof, tex_prof)

def get_area(propfile12, propfile13):
	df12 = pd.read_csv(propfile12)
	df13 = pd.read_csv(propfile13)

	area_column12 = df12['area']
	area_column13 = df13['area']
	
	return(area_column13)


if __name__ == "__main__":
#N_13 = column_density_13(2,2.5)
#calc_mass(N_13, 20)
#print("done")
	cubefile12 = 'ab612co21.fits'
	cubefile13 = 'ab6low.fits'
	clumpfile12 = 'Ant_B6high_Combined_12CO2_1.cube.smoothed.clumps.4.5sig500npix.fits'
	clumpfile13 = 'Ant_B6low_FULL_13CO2_1_Robust2.0.cube.clumps.4sig500npix.fits'		
	nsig=5 #typically use 5sigma in maps but maybe use sigma used for qc
#        velocity_range = np.arange(1300, 1800, 5) #all velocity values in km/s
	dpc = 22*10**6 #distance in pc
	propfile12 = '12CO21_clump_props_final.csv' 
	propfile13 = '13CO21_props_final.csv'


        #HEADER DATA
	clumps12 = fits.getdata(clumpfile12)
	clumps13 = fits.getdata(clumpfile13)
	ncl_total12 = np.nanmax(clumps12)
	ncl_total13 = np.nanmax(clumps13)
	
	COdata12 = fits.getdata(cubefile12)
	COdata13 = fits.getdata(cubefile13)
	COdata_header12 = fits.getheader(cubefile12)
	COdata_header13 = fits.getheader(cubefile13)

	bmaj12 = COdata_header12['bmaj']*3600 #converted to arcsec
	bmin12 = COdata_header12['bmin']*3600 #converted to arcsec
	bmaj13 = COdata_header13['bmaj']*3600 #converted to arcsec
	bmin13 = COdata_header13['bmin']*3600 #converted to arcsec

#	print(bmaj13, bmaj12, bmin13, bmin12)
	freq12_GHz=COdata_header12['CRVAL3']*(10**(-9)) #converted to GHz
	freq13_GHz=COdata_header13['CRVAL3']*(10**(-9)) #converted to GHz
	freq12_Hz = COdata_header12['CRVAL3']
	freq13_Hz = COdata_header13['CRVAL3']
	rms12 = stats.mad_std(COdata12[~np.isnan(COdata12)])
	rms_K12 = 1.222 * 10**3 * (rms12 * 1000) / (freq12_GHz**2 * bmin12 * bmaj12)
	rms13 = stats.mad_std(COdata13[~np.isnan(COdata13)])
	rms_K13 = 1.222 * 10**3 * (rms13 * 1000) / (freq13_GHz**2 * bmin13 * bmaj13) 


#	pix_per_beam12 = 1.133*COdata_header12['bmaj']*COdata_header12['bmin']/(abs(COdata_header12['cdelt1']*COdata_header12['cdelt2']))
#	pix_per_beam13 = 1.133*COdata_header13['bmaj']*COdata_header13['bmin']/(abs(COdata_header13['cdelt1']*COdata_header13['cdelt2']))
#	arcsec_per_beam12 = COdata_header12['CDELT2']*3600*pix_per_beam12
#	arcsec_per_beam13 = COdata_header13['CDELT2']*3600*pix_per_beam13
	arcsec_per_pix12 = COdata_header12['CDELT2']*3600
	arcsec_per_pix13 = COdata_header13['CDELT2']*3600
#	brad12 = np.sqrt(bmaj12*bmin12)/206265*dpc
#	brad13 = np.sqrt(bmaj13*bmin13)/206265*dpc

	COtemp12 = 1.222 * 10**3 * (COdata12 * 1000) / (freq12_GHz**2 * bmin12 * bmaj12) #convert to K
	COtemp13 = 1.222 * 10**3 * (COdata13 * 1000) / (freq13_GHz**2 * bmin13 * bmaj13)


	
	
	for ncl in range(3,4):
		print(ncl)
		t12 = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[0]
		t13 = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[1]
		mom0 = np.nansum(t13, axis=0)
		ax=plt.subplot(111)
		plt.imshow(mom0)
		plt.colorbar()
		plt.show()
#		print("t12:", np.nanmax(t12))
#		print("t13:", np.nanmax(t13))
#		column_density_13(t12, t13, freq13_Hz)
#		clump_area = get_area(propfile12, propfile13)[ncl]
		clump_area = 2 #test
		calc_mass(column_density_13(t12, t13, freq13_Hz), clump_area)

#	print("COtemp12 max", np.nanmax(COtemp12))
#	print("cotemp13 max", np.nanmax(COtemp13))


	


	print("done")

