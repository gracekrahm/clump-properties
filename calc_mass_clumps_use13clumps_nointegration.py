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

#units
cms=u.cm/u.s
kms = u.km/u.s
ms = u.m/u.s
Hz = 1/u.s
Kkms = u.K/u.km/u.s
grav_units = (u.m**3)/(u.kg*u.s**2)

#lists
if TITLES:
	ncl_list = ["ncl"]
	number_density_list = ["13CO number density"]
	total_density_list = ["total column density"]
	mass_list = ["total mass (kg)"]
	mass_msol_list = ["total mass (msol)"]
	errmass_list = ["mass error"]
	alphavir_list = ["alphavir"]
	alphavir_error_list = ["alphavir error"]
else:
	ncl_list = []
	number_density_list = []
	total_density_list = []
	mass_list = []
	mass_msol_list = []
	errmass_list = []
	alphavir_list = []
	alphavir_error_list = []
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
        #assume 12CO(2-1) is optically thick and use firecracker eqn 2 to find tex
	tex = tul_12/np.log((tul_12.value/t12)+1)
	print("tex max", np.nanmax(tex))

        #assume 13CO(2-1) is optically thin and use eqn 2 to calculate its optical depth
	tau_13 = -np.log(1-(t13/tul_13.value)*(((1/(np.exp(tul_13/tex)-1))-(1/(np.exp(tul_13/tcmb)-1)))**(-1)))
	print("tau 13 max:", np.nanmax(tau_13))


        #calculate column density for 13CO
	Q = (tex.value/B.value)+(1/3)
	print("q:", np.nanmax(Q))
	vel_disp = get_props(propfile12, propfile13, ncl)[0]
	N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * Q/(1-np.exp(-tul_13/tex)) * tau_13*vel_disp*np.sqrt(2*np.pi) #best so far
##	N_13 = ((8*np.pi*nu_naught**2)/(Aul*gu*c**2)) * Q/(1-np.exp(-tul_13/tex)) * tau_13*vel_disp*np.sqrt(np.pi)*2*nu_naught/c
##	N_13 = (2.6*10**14)*tex.value * (tau_13*vel_disp.value*np.sqrt(np.pi)*2*nu_naught.value/c.value) / (1-np.exp(-5.3/tex.value)) 
#	N_13 = (4*(np.pi**(3/2))*(freq13**3)/((c**3)*Aul*np.sqrt(np.log(2))*(np.exp(tul_13/tex)-1))) * tau_13*2.35482004503*vel_disp * Q/gu
#probably more accurate	N_13 = (8*np.pi*Q*2.35482004503*vel_disp*np.sqrt(np.pi/(4*np.log(2)))*np.exp(tul_13/tex)*tau_13*nu_naught**3) / (Aul*gu*(np.exp(15.9/tex.value)-1)*c**3) 
	print("n units", N_13.unit)
	N_13 = (N_13.value)/((u.m)**2)
	
	print("n max, n shape", np.nanmax(N_13), N_13.shape)
	number_density_list.append(np.nanmax(N_13.value))
	return N_13

def calc_mass(N_13, pix_area, err13CO, sigv, R):
	mH2 = (2*1.67*10**(-27))*u.kg #mass of H2 in kg	
	msol = (1.989*10**30)*u.kg
	G = (6.6743*10**(-11))*grav_units
        # use ratios of 13CO to estimate total density
#	H_to_13CO = 2.*10**6 # abundance ratio of H_2 / 13CO
	total_surface_density = (mH2*1.3*N_13*(2.0*10**6))
#	np.set_printoptions(threshold=sys.maxsize)
#	print(total_surface_density)
	print("total surface density(m^-2):", np.nanmax(total_surface_density), total_surface_density.shape)
	#convert from cm^-2 ro pc^-2

#	density_pc = total_surface_density.to(u.kg/(u.parsec**2))
#	print("density pc", np.nanmax(density_pc))	
	#density_pc = total_surface_density	

#	density_mom0 = np.nansum(total_surface_density, axis=0)
#	ax = plt.subplot(111)
#	plt.imshow(density_mom0, origin='lower', vmax = np.min([np.max([5.*np.nanmax(density_mom0), 8*4]), np.nanmax(density_mom0)]), vmin=0)
#	plt.colorbar()
#	plt.show()	
	total_mass = np.nansum(pix_area*total_surface_density)
	print("total mass", total_mass)
	print("m/m sun", total_mass/msol)
	
	#calculate mass error
	errmass = np.sqrt((0.1*total_mass)**2 + (np.nansum(total_surface_density*err13CO*pix_area))**2)
	print('errmass:', np.nanmax(errmass))
	
	#calculate the alpha virial parameter
	alphavir = 5*(sigv**2)*R/(G*total_mass)
	print('alphavir:', alphavir)
	alphavirtest = 5*((sigv/10)**2)*R/(G*total_mass)
	print('av test', alphavirtest)
	# i have no idea what molly's script means tbh alphavir_error = 
	
	total_density_list.append(np.nanmax(total_surface_density.value))
	mass_list.append(total_mass.value)
	print("mass list:", mass_list)
	mass_msol_list.append((total_mass/msol).value)
	print(mass_msol_list)
	errmass_list.append(errmass.value)
	alphavir_list.append(alphavir.value)
	print(alphavir_list)
#	alphavir_error_list.append(alphavir_error)

#	return(total_surface_density, total_mass, errmass, alphavir)

def get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13):
	clumpdata12 = np.where(clumps13==ncl, COtemp12, np.nan)
	clumpdata12 = np.where((clumps12 !=np.nan).any(), COtemp12, np.nan)
	clumpdata12 = np.where(clumpdata12 > nsig*rms_K12, COtemp12, np.nan)

	clumpdata13 = np.where(clumps13==ncl, COtemp13,	np.nan)
	clumpdata13 = np.where(clumpdata13 > nsig*rms_K13, COtemp13, np.nan)
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
		R_column = np.sqrt(area/np.pi)*u.parsec
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
	clumpfile12 = 'Ant_B6high_Combined_12CO2_1.cube.smoothed.clumps.4.5sig500npix.fits'
	clumpfile13 = 'Ant_B6low_FULL_13CO2_1_Robust2.0.cube.clumps.4sig500npix.fits'		
	nsig=5 #typically use 5sigma in maps but maybe use sigma used for qc
#        velocity_range = np.arange(1300, 1800, 5) #all velocity values in km/s
	dpc = 22*10**6 #distance in pc
	propfile12 = '12CO21_clump_props_final.csv' 
	propfile13 = '13CO21_props_final.csv'
	pc = 3.086*10**16 #m/pc
	arcsec_per_pix = 0.0140000000000004
	pixelarea = ((arcsec_per_pix/206265.)*dpc*pc*u.m)**2 #m^2

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
	rms12 = stats.mad_std(COdata12[~np.isnan(COdata12)])
	rms_K12 = 1.222 * 10**3 * (rms12 * 1000) / (freq12.to(u.GHz).value**2 * bmin12 * bmaj12)
	rms13 = stats.mad_std(COdata13[~np.isnan(COdata13)])
	rms_K13 = 1.222 * 10**3 * (rms13 * 1000) / (freq13.to(u.GHz).value**2 * bmin13 * bmaj13) 
	print("5rmsk12, 13", rms_K12*5, rms_K13*5)

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

		

	for ncl in range(1,3):
		print(ncl)
		ncl_list.append(ncl)
		t12 = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[0]#*Kkms
		t13 = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[1]#*Kkms
		err13CO = get_clumps(ncl, COtemp12, COtemp13, nsig, rms_K12, rms_K13)[2]
#		print("t12:", np.nanmax(t12))
#		print("t13:", np.nanmax(t13))
#		column_density_13(t12, t13, freq13, ncl)
#		clump_area = get_area(propfile12, propfile13)[ncl]
#		clump_area = 2*(u.parsec**2) #test
	
		get_props_list = get_props(propfile12, propfile13, ncl)	
		sigv = get_props_list[0]
		R = get_props_list[1]
#		sigv, R = get_props(propfile12, propfile13, ncl)[0,1]
#		R = get_props(propfile12, propfile13, ncl)[1]
		calc_mass(column_density_13(t12, t13, freq13, ncl), pixelarea, err13CO, sigv, R)

	print("lists")
	print(list(zip(ncl_list, number_density_list, total_density_list, mass_list)))
	props_lists = zip(ncl_list, number_density_list, total_density_list, mass_list, mass_msol_list, errmass_list, alphavir_list)#, alphavir_error_list)
	print(list(props_lists))
	if SAVE:
		with open('13CO21_clump_mass_props.csv', 'w') as f:
			writer=csv.writer(f, delimiter='\t')
			writer.writerows(zip(ncl_list, number_density_list, total_density_list, mass_list, mass_msol_list, errmass_list, alphavir_list))#, alphavir_error_list))
	print("done")


