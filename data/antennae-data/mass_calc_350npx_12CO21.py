import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from astropy.io import fits
from astropy.stats import mad_std
from astropy import stats
from astropy import units as u
from scipy.optimize import curve_fit
import h5py
from astrodendro import Dendrogram
import sys
from scipy.optimize import fsolve
import warnings 
import pandas as pd

import warnings
warnings.simplefilter('ignore')

#choices
SAVE=True
equiv_radius=False
TITLES = True
check_12_optically_thin = False
use_filling_factor = True
f=1

#units
cms=u.cm/u.s
kms = u.km/u.s
ms = u.m/u.s
Hz = 1/u.s
Kkms = u.K/u.km/u.s
grav_units = (u.m**3)/(u.kg*u.s**2)

allsum_list = ['mass from summing all']
momsum_list = ['mass from summing mom0']

mom0_array = []
#lists
if TITLES:
	ncl_list = ["ncl"]
	number_density_sum_list = ["13CO number density sum"]
	number_density_mean_list = ["13CO number density mean"]
	number_density_max_list = ["13CO number density max"]
else:
	ncl_list = []
	number_density_sum_list = []
	number_density_mean_list = []
	number_density_max_list = []


def column_density_13(t12, t13, freq13, ncl):
        #constants
	tul_12 = 11.06*u.K #K for 12CO(2-1)
	tul_13 = 10.58*u.K #K for 13CO(2-1)
        #gu = 2J +1 
        # J = 2-1
	gu = 5
	#nu_naught = 220.399 #GHz
	nu_naught = freq13
	tcmb = 2.73*u.K
	B = 2.644*u.K #K for 13CO
	Aul = 6.03*10**(-7)*Hz #http://www.ifa.hawaii.edu/users/jpw/classes/star_formation/reading/molecules.pdf
	c = 2.998*10**8*ms

	#if check_12_optically_thin:
		#if ncl in [127, 137, 155, 157, 162]:
			#tex = t12*u.K
		#else:
			#tex = tul_12/np.log((tul_12.value/t12)+1)
	#else:
		#tex = tul_12/np.log((tul_12.value/t12)+1)	
	#assume 12CO(2-1) is optically thick and use firecracker eqn 2 to find tex
#	tex = tul_12/np.log((tul_12.value/t12)+1)
	tex = tul_12/np.log((11.255762*f+t12)/(t12+0.195762*f))
	print("tex max, tex mean", np.nanmax(tex), np.nanmean(tex))

        #assume 13CO(2-1) is optically thin and use eqn 2 to calculate its optical depth
	tau_13 = -np.log(1-((t13/f)/tul_13.value)*(((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1)))
#	tau_12 = -np.log(1-(t12/tul_12.value)*(((1/(np.exp(tul_12/tex)-1))-(1/(np.exp(tul_12/tcmb)-1)))**(-1)))
	print("tau 13 max, mean:", np.nanmax(tau_13), np.nanmean(tau_13))
#	print("tau 12 max, mean:", np.nanmax(tau_12), np.nanmean(tau_12))


        #calculate column density for 13CO
	Q = (tex.value/B.value)+(1/3)
	print("q:", np.nanmax(Q))
#	vel_disp = get_props(propfile12, propfile13, ncl)[0]
	vel_disp = 5*u.km/u.s
	vel_channel = 4999.99999998*u.m/u.s # channel size of velocity 5000 m/s
#no np sqrt 2pi
#	N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*vel_channel*np.sqrt(1) *freq13/c
	N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*vel_channel*np.sqrt(2*np.pi) *freq13/c  #vel	#best so far
#	print('v/c', freq13/c)
#	N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * (Q/(1-np.exp(-tul_13/tex))) * tau_13*np.sqrt(2*np.pi) * 3844959.96893  #freq

	mom0n = np.nansum(N_13, axis=0)
	mom0n=mom0n.value
	mom0n=mom0n/(u.m**2)
	mom0n=mom0n.to(1/u.cm**2)
	mom0_array.append(mom0n.value)
	mom0n=mom0n.value

#	ax = plt.subplot(111)
#	plt.imshow(mom0n, origin='lower')
#	plt.colorbar()
#	plt.title("density")
#	plt.show()
#	print("n units", N_13.unit)
	N_13 = (N_13.value)/((u.m)**2)
	N_13 = N_13.to(1/(u.cm**2))
	dens = N_13*(9.24394022969462*10**(-14))
	print('mean, max dens', np.nanmean(dens), np.nanmax(dens))
	t13area = np.where(t13!=np.nan, 1., np.nan)
	mass = t13area*dens
	print('unsummed mass (msol/pc) mean, max, sum', np.nanmean(mass), np.nanmax(mass), np.nansum(mass))
	mom0_mass = np.nansum(dens, axis=0)
	print('np.nansum(mass), np.nansum(mom0_mass)', np.nansum(mass),	np.nansum(mom0_mass))
	allsum = np.nansum(mass.value)
	momsum = np.nansum(mom0_mass.value)		
	allsum_list.append(allsum)
	momsum_list.append(momsum)
	return N_13


def get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13):
	#clumpdata12 = np.where((clumps12 !=np.nan).any(), COtemp12, np.nan)
	#clumpdata12 = np.where(COtemp12 > nsig*rms_K12, COtemp12, np.nan)
	clumpdata12 = np.where(clumps12==ncl, COtemp12, np.nan)
	#clumpdata12 = np.where(clumpdata12 > nsig*rms_K12, COtemp12, np.nan)

	#clumpdata13 = np.where(COtemp13 > nsig*rms_K13, COtemp13, np.nan)
	#clumpdata13 = np.where((clumps12==ncl).all() and (clumps13!=np.nan).any(), COtemp13, np.nan)
	clumpdata13 = np.where(clumps12==ncl, COtemp13, np.nan)
	#clumpdata13 = np.where(clumpdata13 > rms_K13, COtemp13, np.nan)

	#clumpdata12 = np.where(clumpdata13 > nsig*rms_K13, COtemp12, np.nan)
	err13CO = np.sqrt(np.nansum(clumpdata13))*rms_K13
	return(clumpdata12, clumpdata13, err13CO)


def get_props(propfile12, propfile13, ncl):
	df13 = pd.read_csv(propfile13)
	vels_column = df13['meansigv']*(u.km/u.s)
	print("vel", vels_column[ncl])
	vels_ncl=vels_column[ncl].to(u.m/u.s)
	print("vels disp", vels_ncl)
	if equiv_radius:
		area = df13['area']
		area = area[ncl]
		print('area ncl', area)
		R_column = np.sqrt(area/np.pi)
		print("r", R_column)
		R_column = R_column*u.parsec
		print('equiv rad', R_column)
	else:
		R_column = df13['ellipse R']*u.parsec
	R_column = R_column[ncl].to(u.m)
	print("r column:", R_column)
	return(vels_ncl, R_column)


if __name__ == "__main__":
#N_13 = column_density_13(2,2.5)
#calc_mass(N_13, 20)
#print("done")
	cubefile12 = 'ab612co21.fits'
	cubefile13 = 'ab6low.fits'
	clumpfile12 = 'ab612co21.clumps.5sig350npix.fits'
	clumpfile13 = 'ab6low.clumps.4sig350npix.fits'		
	nsig=1 #typically use 5sigma in maps but maybe use sigma used for qc
#        velocity_range = np.arange(1300, 1800, 5) #all velocity values in km/s
	dpc = 22*10**6 #distance in pc
	propfile12 = '12CO21_clump_props_final.csv' 
	propfile13 = '13CO21_props_final.csv'
	#pc = 3.086*10**16 #m/pc
	arcsec_per_pix = 0.0140000000000004
	#pixelarea = ((arcsec_per_pix/206265.)*dpc*pc*u.m)**2 #m^2

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
	freq12 = COdata_header12['CRVAL3']*Hz
	freq13 = COdata_header13['CRVAL3']*Hz
	print('freq12, 13:', freq12, freq13)
	rms12 = stats.mad_std(COdata12[~np.isnan(COdata12)])
	rms_K12 = 1.222 * 10**3 * (rms12 * 1000) / (freq12.to(u.GHz).value**2 * bmin12 * bmaj12)
	rms13 = stats.mad_std(COdata13[~np.isnan(COdata13)])
	rms_K13 = 1.222 * 10**3 * (rms13 * 1000) / (freq13.to(u.GHz).value**2 * bmin13 * bmaj13) 
	print("5rmsk12, 13", rms_K12*5, rms_K13*5)
	print('deltav', abs(COdata_header13['CDELT3'])) #deltav from dendro program
#	for i in np.arange(30):
#		print('nsig, rms12, rmsk12', i, rms12*i, rms_K12*i)
#	pix_per_beam12 = 1.133*COdata_header12['bmaj']*COdata_header12['bmin']/(abs(COdata_header12['cdelt1']*COdata_header12['cdelt2']))
#	pix_per_beam13 = 1.133*COdata_header13['bmaj']*COdata_header13['bmin']/(abs(COdata_header13['cdelt1']*COdata_header13['cdelt2']))
#	arcsec_per_beam12 = COdata_header12['CDELT2']*3600*pix_per_beam12
#	arcsec_per_beam13 = COdata_header13['CDELT2']*3600*pix_per_beam13
#	arcsec_per_pix12 = COdata_header12['CDELT2']*3600
#	arcsec_per_pix13 = COdata_header13['CDELT2']*3600
#	brad12 = np.sqrt(bmaj12*bmin12)/206265*dpc
#	brad13 = np.sqrt(bmaj13*bmin13)/206265*dpc

	COtemp12 = 1.222 * 10**3 * (COdata12 * 1000) / (freq12.to(u.GHz).value**2 * bmin12 * bmaj12) #convert to K
	COtemp13 = 1.222 * 10**3 * (COdata13 * 1000) / (freq13.to(u.GHz).value**2 * bmin13 * bmaj13)
		

#	for ncl in range(1,101):
	for ncl in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]:
		print(ncl)
		ncl_list.append(ncl)
		t12 = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[0]#*Kkms
		t13 = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[1]#*Kkms
		mom013 = np.nansum(t13, axis=0)
		mom8 = np.nanmax(t13, axis=0)
		#print('tmax:', np.nanmax(mom8))
		npix = np.where(mom8>nsig*rms_K13, 1., np.nan)
		npix=np.nansum(npix)
		err13CO = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[2]
#		print("t12:", np.nanmax(t12))
#		print("t13:", np.nanmax(t13))
#		column_density_13(t12, t13, freq13, ncl)
#		clump_area = get_area(propfile12, propfile13)[ncl]
#		clump_area = 2*(u.parsec**2) #test
	
#		get_props_list = get_props(propfile12, propfile13, ncl-1)	
#		sigv = get_props_list[0]
#		R = get_props_list[1]
#		sigv, R = get_props(propfile12, propfile13, ncl)[0,1]
#		R = get_props(propfile12, propfile13, ncl)[1]
		column_density_13(t12, t13, freq13, ncl)
#		calc_mass(column_density_13(t12, t13, freq13, ncl), pixelarea, err13CO, sigv, R, mom013, npix)

	print("lists")
	print(list(zip(ncl_list, number_density_sum_list, number_density_mean_list, number_density_max_list)))

	props_lists = zip(ncl_list, number_density_sum_list, number_density_mean_list, number_density_max_list)
	print(list(props_lists))
	if SAVE:
		with open('mass_12CO21_withsqrt2pi.csv', 'w') as f:
			writer=csv.writer(f, delimiter=',')
			writer.writerows(zip(ncl_list,allsum_list, momsum_list))
	print("done")


