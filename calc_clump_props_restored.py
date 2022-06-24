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
import h5py
from astrodendro import Dendrogram
import peakutils.peak
import datetime
import multiprocessing as mp
import sys


import warnings
warnings.simplefilter('ignore')

#lists
ncl_list=["ncl"]
npix_list=["Npix"]
nvox_list=["Nvox"]
maxpix_list=["maxpix"]
maxpix_location_list=[]
maxpix_x_list=["maxpix x"]
maxpix_y_list=["maxpix y"]
maxpix_v_list=["maxpix v"]
Tmax_list=["Tmax"]
vmax_list=["vmax"] 
sigv_list=["sigv"]
sigv_error_list=["sigv error"]
ellipse_a_list=["ellipse fit a"]
ellipse_b_list=["ellipse fit b"]
ellipse_theta_list=["ellipse theta"]
ellipse_R_list=["ellipse R"]
ellipse_R_error_list=["ellipse R error"]
area_list=["area"]        
perimeter_list=["perimeter"]





def gaussian_eqn(x, A, x0, sigma):
	return A*np.exp(-((x-x0)**2)/(2*sigma**2))

def fitEllipse(cont):
	# From online stackoverflow thread about fitting ellipses
	x=cont[:,0]
	y=cont[:,1]

	x=x[:,None]
	y=y[:,None]

	D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
	S=np.dot(D.T,D)
	C=np.zeros([6,6])
	C[0,2]=C[2,0]=2
	C[1,1]=-1
	E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))
	n=np.argmax(E)
	a=V[:,n]

	#-------------------Fit ellipse-------------------
	b,c,d,f,g,a=a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
	num=b*b-a*c
	cx=(c*d-b*f)/num
	cy=(a*f-b*d)/num

	angle=0.5*np.arctan(2*b/(a-c))*180/np.pi
	up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
	down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
	down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
	a=np.sqrt(abs(up/down1))
	b=np.sqrt(abs(up/down2))

	params=[cx,cy,a,b,angle]

	return params

def pix_to_pc(pix, dpc, arcsec_per_pix):
	return	pix*(arcsec_per_pix/206265)*dpc


def operations(clumps, rms_K, nsig, dpc, COtemp, ncl, velocity_range, arcsec_per_pix, brad):
	#isolate & mask clump
	current_clump = np.where(clumps==ncl, COtemp, np.nan)
	mask3d = np.where(current_clump > nsig*rms_K, 1., np.nan)
	#Calculate moments 0 and 8 and mask below nsig
	mom0 = np.nansum(current_clump, axis=0) #total intensity
	mom8 = np.nanmax(current_clump, axis=0) #peak intensity
	mask2d = np.where(mom8 > nsig*rms_K, 1., np.nan)

	#find maximum intensity
	maxpix = np.nanmax(mom8)
#	maxpix = mom8.argmax()
	maxpix_location = np.unravel_index(np.nanargmax(current_clump), (current_clump.shape))
#	maxpix_location = np.unravel_index(maxpix.argmax(), maxpix.shape())
	print("maxpix, location:", maxpix, maxpix_location)	

	#calculate pixels and voxels over nsig
	Npix = np.nansum(mask2d)
	Nvox = np.nansum(mask3d)
#	Nvox = np.nansum(np.where(current_clump > 0,1., np.nan))
	print("npix, nvox:", Npix, Nvox)
	

	#calculate intensity-weighted profile of the cluster
	profile = np.nanmean(current_clump, axis=(1,2))
#	profile = np.nansum(current_clump*mom8, axis=(1,2))/np.nansum(mom8)
	profile = np.where(np.isnan(profile), 0.0, profile)
	Tmax = np.nanmax(profile)
	vmax = velocity_range[np.nanargmax(profile)]
	print("vmax, tmax:", vmax, Tmax)
	
	#calculate average velocity profile
	sol, cov = curve_fit(gaussian_eqn, velocity_range, profile, p0=[Tmax, 2.5, vmax])
#	sol, cov = curve_fit(gaussian_eqn, velocity_range, profile, p0=[Tmax, 2.5, vmax])  #solution, covariance matrix  #check p0
	print("sol:", sol)
	print("cov:", cov)
	error_sigv = np.sqrt(np.diag(cov)[1])
	print("error sig v:", error_sigv)
	mean_sigv = sol[2]
	print("mean sigv", mean_sigv)


	#remask moments to 0 instead of nan
	mask2dzero = np.where(mom0>0, mom0, 0.0)
	full_contour_mask = np.where(mom0 >0, 1., 0.) #binary mask

	#contour lines
	half_contour = plt.contour(mask2dzero,[0.5*np.nanmax(mask2dzero)])
	full_contour = plt.contour(mask2dzero,[np.nanmax(mask2dzero)])
	
	#create contour array
	for i in np.arange(len(half_contour.allsegs[0])):
		if i==0:
			dat0 = np.array(half_contour.allsegs[0][i])
		else:
			dat0 = np.concatenate((dat0, np.array(half_contour.allsegs[0][i])))

	#Fit ellipse to half-power contour of CO
	ellipse_parameters = fitEllipse(dat0)
	xc,yc,a,b,theta = ellipse_parameters
	print("unchanged ellipse parameters", xc, yc, a, b, theta)
	
	#Calculate properties from ellipse fit
	t = np.arange(0,2.01, 0.01) * np.pi
	xt = xc + a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t)
	yt = yc + a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)
	x = dat0[:,0]
	y = dat0[:,1]
	xarr = np.reshape(np.repeat(x, len(t)), (len(x), len(t)))
	yarr = np.reshape(np.repeat(y, len(t)), (len(y), len(t)))
	d = np.sqrt((xarr - xt)**2 + (yarr - yt)**2)
	res = np.nanmin(d, axis=1)
#	print("res", res)
	print("a,b:", a, b)

	#convert a and b to pc
#	minor_axis = ((arcsec_per_pix/206265.) * dpc)* np.max([a, b])
#	major_axis = ((arcsec_per_pix/206265.) * dpc)* np.min([a, b])
	npmax = np.max([a,b])
	npmin = np.min([a,b])
	a = pix_to_pc(a, dpc, arcsec_per_pix)
	b = pix_to_pc(b, dpc, arcsec_per_pix) 	

	print("ellipse a, b, theta:", a, b, theta)
	# theta is the angle between the positive x axis and the axes of the ellipse (in deg)

	#calculate R and error
	R = np.sqrt(a*b)*(2./2.35) #pc, sigma (not HWHM)
#	R_error = np.nanmean(res) * ((arcsec_per_pix/206265.) * dpc) * (2./2.35)
	nanmean_res = np.nanmean(res)
	R_error = pix_to_pc(nanmean_res, dpc, arcsec_per_pix)*(2./2.35)
		

	#deconvolve #when?
	R = np.sqrt((R)**2 - (brad/2)**2)
	R_error = R*R_error/np.sqrt(np.abs((R)**2 - (brad/2)**2))

	#xradius (ask about later)
	R = 1.91*R
	R_error = 1.91*R_error
	
	print("R, error:", R, R_error)


	#calculate area and perimeter using full contour
#	area = np.nansum(full_contour_mask)
#	area = pix_to_pc(area, dpc, arcsec_per_pix) #convert area to pc
	area = np.nansum(full_contour_mask)*(arcsec_per_pix/206265.*dpc)**2
	perimeter = abs(np.diff(full_contour_mask, axis=0)).sum() + abs(np.diff(full_contour_mask, axis=1)).sum()
	perimeter = pix_to_pc(perimeter, dpc, arcsec_per_pix) #convert perimeter to pc

	print('area and perimeter: ', area, perimeter)

	

        #add to lists
#	output = np.array([ncl, maxpix, maxpix_location, Npix, Nvox])
	ncl_list.append(ncl)
	npix_list.append(Npix)
	nvox_list.append(Nvox)
	maxpix_list.append(maxpix)
#	maxpix_location_list.append(maxpix_location)
	maxpix_x_list.append(maxpix_location[0])
	maxpix_y_list.append(maxpix_location[1])
	maxpix_v_list.append(maxpix_location[2])
	Tmax_list.append(Tmax)
	vmax_list.append(vmax)
	sigv_list.append(mean_sigv)
	sigv_error_list.append(error_sigv)
	ellipse_a_list.append(a)
	ellipse_b_list.append(b)
	ellipse_theta_list.append(theta)
	ellipse_R_list.append(R)
	ellipse_R_error_list.append(R_error)
	area_list.append(area)
	perimeter_list.append(perimeter)
	keep_vars = (current_clump, mom0, mom8)
	return(keep_vars)



if __name__ == "__main__":

	#INPUTS
	cubefile= 'ab612co21.fits'   
	clumpfile= 'Ant_B6high_Combined_12CO2_1.cube.smoothed.clumps.4.5sig500npix.fits'
	nsig=5 #typically use 5sigma in maps but maybe use sigma used for qc	
	velocity_range = np.arange(1300, 1800, 5) #all velocity values in km/s
	dpc = 22*10**6 #distance in pc
	clumps = fits.getdata(clumpfile)

	#HEADER DATA
	ncl_total = len(clumps)
	COdata = fits.getdata(cubefile)
	COdata_header = fits.getheader(cubefile)
	rms = stats.mad_std(COdata[~np.isnan(COdata)])
	bmaj = COdata_header['bmaj']*3600 #converted to arcsec
	bmin = COdata_header['bmin']*3600 #converted to arcsec
	freq=COdata_header['CRVAL3']*(10**(-9)) #converted to GHz
	rms_K = 1.222 * 10**3 * (rms * 1000) / (freq**2 * bmin * bmaj)
	pix_per_beam = 1.133*COdata_header['bmaj']*COdata_header['bmin']/(abs(COdata_header['cdelt1']*COdata_header['cdelt2']))
	arcsec_per_beam = COdata_header['CDELT2']*3600*pix_per_beam
	arcsec_per_pix = COdata_header['CDELT2']*3600
	brad = np.sqrt(bmaj*bmin)/206265*dpc	
	print("rms, rmsk:", rms, rms_K)
	print("freq:", freq)
	COtemp = 1.222 * 10**3 * (COdata * 1000) / (freq**2 * bmin * bmaj)
	props_array = []
	for ncl in range(1,3):
#	for ncl in range(1, ncl_total):
#	for ncl in np.array([ncl_total]):
		print(ncl)
		props=operations(clumps, rms_K, nsig, dpc, COtemp, ncl, velocity_range, arcsec_per_pix, brad)
		props_array.append(props)
#	print(zip(ncl_list, npix_list, nvox_list, maxpix_list, maxpix_x_list, maxpix_y_list, maxpix_v_list))
#	print(ncl_list)

	with open('test4.csv', 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerows(zip(ncl_list, npix_list, nvox_list, maxpix_list, maxpix_x_list, maxpix_y_list, maxpix_v_list, vmax_list, Tmax_list, sigv_list, sigv_error_list, ellipse_a_list, ellipse_b_list, ellipse_theta_list, ellipse_R_list, ellipse_R_error_list, area_list, perimeter_list))


#	np.savetxt('clump_props_test.txt', props_array, fmt='%s', delimiter='\t')
	print("done")
