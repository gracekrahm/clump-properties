#!/usr/bin/env python2.7
import warnings
import os
import csv
import sys
import h5py
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from astrodendro.scatter import Scatter
from astrodendro import Dendrogram, ppv_catalog, analysis
from astropy import stats
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column
#from scimes import SpectralCloudstering

with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

def run_dendro(label='ab6high_5sig', cubefile='ab612co21.fits', flatfile='ab612co21mom0.fits', 
               redo='n', nsigma=5, min_delta=2.5, min_bms=2.,
               position_dependent_noise=False, # will use rms map in dendro
               criteria=['volume'], # for SCIMES
               doplots=True,
               dendro_in=None, # use this instead of re-loading
               **kwargs):

    global cubedata,rmsmap,threshold_sigma

    #%&%&%&%&%&%&%&%&%&%&%&%
    #    Make dendrogram
    #%&%&%&%&%&%&%&%&%&%&%&%
    hdu3 = fits.open(cubefile)[0]
    hd3 = hdu3.header
    cubedata = hdu3.data

    # Deal with oddities in 30 Dor cube
    if hd3['NAXIS'] == 3:
        for key in ['CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4', 'NAXIS4']:
            if key in hd3.keys():
                hd3.remove(key)

    # Get cube parameters
    sigma = stats.mad_std(hdu3.data[~np.isnan(hdu3.data)])
    print('Robustly estimated RMS: ',sigma)
    ppb = 1.133*hd3['bmaj']*hd3['bmin']/(abs(hd3['cdelt1']*hd3['cdelt2']))
    print('Pixels per beam: ',ppb)

    # Make the dendrogram if not present or redo=y
    if position_dependent_noise:
        dendrofile='dendro_dendrogram_rmsmap_ab6high_5sig.fits'
    else:
        dendrofile='dendro_dendrogram_ab6high_5sig.fits'

    if dendro_in!=None:
        d = dendro_in
    elif redo == 'n' and os.path.isfile(dendrofile):
        print('Loading pre-existing dendrogram')
        d = Dendrogram.load_from(dendrofile)
    else:
        print('Make dendrogram from the full cube')
        if position_dependent_noise:
            mask3d = cubedata<5.*sigma
            rmsmap = np.nanstd(cubedata*mask3d,axis=0) # assumes spectral 1st
            threshold_sigma = min_delta # for custom_independent function
            d = Dendrogram.compute(hdu3.data, min_value=nsigma*sigma,
                                   min_delta=min_delta*sigma,
                                   min_npix=min_bms*ppb, verbose = 1,
                                   is_independent=custom_independent)
        else:
            d = Dendrogram.compute(hdu3.data, min_value=nsigma*sigma,
                                   min_delta=min_delta*sigma,
                                   min_npix=min_bms*ppb, verbose = 1)
        d.save_to(dendrofile)

    if doplots:
        # checks/creates directory to place plots
        if os.path.isdir('ab6high_5sig_dendro_plots') == 0:
            os.makedirs('ab6high_5sig_dendro_plots')
        
        # Plot the tree
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)            
        #ax.set_yscale('log')
        ax.set_xlabel('Structure')
        ax.set_ylabel('Intensity ['+hd3['BUNIT']+']')
        p = d.plotter()
        branch = [s for s in d.all_structures if s not in d.leaves and s not in d.trunk]
        tronly = [s for s in d.trunk if s not in d.leaves]
        for st in tronly:
            p.plot_tree(ax, structure=[st], color='brown', subtree=False)
        for st in branch:
            p.plot_tree(ax, structure=[st], color='black', subtree=False)
        for st in d.leaves:
            p.plot_tree(ax, structure=[st], color='green')
        #p.plot_tree(ax, color='black')
        plt.savefig('ab6high_5sig_dendro_plots/'+label+'_dendrogram.pdf', bbox_inches='tight')

    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #   Generate the catalog
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    print("Generate a catalog of dendrogram structures")
    metadata = {}
    if hd3['BUNIT'].upper()=='JY/BEAM':
        metadata['data_unit'] = u.Jy / u.beam
    elif hd3['BUNIT'].upper()=='K':
        metadata['data_unit'] = u.K
    else:
        print("Warning: Unrecognized brightness unit")
    metadata['vaxis'] = 0
    if 'RESTFREQ' in hd3.keys():
        freq = hd3['RESTFREQ'] * u.Hz
    elif 'RESTFRQ' in hd3.keys():
        freq = hd3['RESTFRQ'] * u.Hz
    metadata['wavelength'] = freq.to(u.m,equivalencies=u.spectral())
    metadata['spatial_scale']  =  hd3['cdelt2'] * 3600. * u.arcsec
    if hd3['ctype3'][0:3]=='VEL' or hd3['ctype3'][0:4]=='VRAD':
        dv=hd3['cdelt3']/1000.*u.km/u.s
    else:
        assert hd3['ctype3'][0:4]=='FREQ'
        dv=2.99792458e5 *np.absolute(hd3['cdelt3'])/freq.value *u.km/u.s
    metadata['velocity_scale'] = dv
    bmaj = hd3['bmaj']*3600. * u.arcsec # FWHM
    bmin = hd3['bmin']*3600. * u.arcsec # FWHM
    metadata['beam_major'] = bmaj
    metadata['beam_minor'] = bmin

    if not( redo=="n" and os.path.exists(label+'_full_catalog.txt')):
        print("generating catalog")
        cat = ppv_catalog(d, metadata)
        print(cat.info())
        
        # Add additional properties: Average Peak Tb and Maximum Tb
        srclist = cat['_idx'].tolist()
        tmax  = np.zeros(len(srclist), dtype=np.float64)
        tpkav = np.zeros(len(srclist), dtype=np.float64)
#        for i, c in enumerate(srclist):
#            peakim = np.nanmax(hdu3.data*d[c].get_mask(), axis=0)
#            peakim[peakim==0] = np.nan
#            tmax[i]  = np.nanmax(peakim)
#            tpkav[i] = np.nanmean(peakim)
        if hd3['BUNIT'].upper()=='JY/BEAM':
            omega_B = np.pi/(4*np.log(2)) * bmaj * bmin
            convfac = (u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B,freq))
            tmax *= convfac
            tpkav *= convfac
        newcol = Column(tmax, name='tmax')
        newcol.unit = 'K'
        cat.add_column(newcol)
        newcol = Column(tpkav, name='tpkav')
        newcol.unit = 'K'
        cat.add_column(newcol)
        
        cat.write(label+'_full_catalog.txt', format='ascii.ecsv', overwrite=True)

    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #   Generate the catalog with clipping
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%

    if not( redo=="n" and os.path.exists(label+'_full_catalog_clipped.txt')):
        print("generating clipped catalog")
        ccat = ppv_catalog(d, metadata)
        print(ccat.info())
    
        # Add additional properties: Average Peak Tb and Maximum Tb
        srclist = ccat['_idx'].tolist()
        tmax  = np.zeros(len(srclist), dtype=np.float64)
        tpkav = np.zeros(len(srclist), dtype=np.float64)
#        for i, c in enumerate(srclist):
#            peakim = np.nanmax(hdu3.data*d[c].get_mask(), axis=0)
#            peakim[peakim==0] = np.nan
#            clmin = np.nanmin(hdu3.data*d[c].get_mask())
#            tmax[i]  = np.nanmax(peakim) - clmin
#            tpkav[i] = np.nanmean(peakim) - clmin
        if hd3['BUNIT'].upper()=='JY/BEAM':
            omega_B = np.pi/(4*np.log(2)) * bmaj * bmin
            convfac = (u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B,freq))
            tmax *= convfac
            tpkav *= convfac
        newcol = Column(tmax, name='tmax-tmin')
        newcol.unit = 'K'
        ccat.add_column(newcol)
        newcol = Column(tpkav, name='tpkav-tmin')
        newcol.unit = 'K'
        ccat.add_column(newcol)
    
        ccat.write(label+'_full_catalog_clipped.txt', format='ascii.ecsv', overwrite=True)
    

    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Running SCIMES
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
#     print("Running SCIMES")
#     dclust = SpectralCloudstering(d, cat, criteria = criteria, keepall=True)
#     print(dclust.clusters)
# 
#     print("Visualize the clustered dendrogram")
#     dclust.showdendro()
#     plt.savefig('ab6high_5sig_dendro_plots/'+label+'_clusters_tree.pdf')
# 
#     print("Produce the assignment cube")
#     dclust.asgncube(hd3)
#     try:
#         os.remove(label+'_asgncube.fits.gz')
#     except OSError:
#         pass
#     dclust.asgn.writeto(label+'_asgncube.fits.gz')

    #sys.exit("Stopping here")

    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Image the trunks
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    if doplots:
        print("Image the trunks")
        
        hdu2 = fits.open(flatfile)[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vmax = np.nanmax(hdu2.data)/2.
        im = ax.matshow(hdu2.data, origin='lower', cmap=plt.cm.Blues, vmax=vmax)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        if 'xlims' in kwargs:
            ax.set_xlim(kwargs['xlims'])
        if 'ylims' in kwargs:
            ax.set_ylim(kwargs['ylims'])
        
        # Make a trunk list
        tronly = [s for s in d.trunk if s not in d.leaves]
        f = open(label+'_trunks.txt', 'w')
        
        for c in tronly:
            f.write('{:<4d} | '.format(c.idx))
            # Plot the actual structure boundaries
            mask = d[c.idx].get_mask()
            mask_coll = np.amax(mask, axis = 0)
            plt.contour(mask_coll, colors='red', linewidths=1, levels = [0])
            # Plot the ellipse fits
            s = analysis.PPVStatistic(d[c.idx])
            ellipse = s.to_mpl_ellipse(edgecolor='black', facecolor='none')
            ax.add_patch(ellipse)
            # Make sub-lists of descendants
            print('Finding descendants of trunk ',c.idx)
            desclist = []
            if len(d[c.idx].descendants) > 0:
                for s in d[c.idx].descendants:
                    desclist.append(s.idx)
                desclist.sort()
                liststr=','.join(map(str, desclist))
                f.write(liststr)
            f.write("\n")
        f.close()
        
        fig.colorbar(im, ax=ax)
        plt.savefig('ab6high_5sig_dendro_plots/'+label+'_trunks_map.pdf', bbox_inches='tight')
        plt.close()
        
        # Make a branch list
        branch = [s for s in d.all_structures if s not in d.leaves and s not in d.trunk]
        slist = []
        for c in branch:
            slist.append(c.idx)
        slist.sort()
        with open(label+'_branches.txt', 'w') as output:
            writer = csv.writer(output)
            for val in slist:
                writer.writerow([val])    
        
        #%&%&%&%&%&%&%&%&%&%&%&%&%&%
        #     Image the leaves
        #%&%&%&%&%&%&%&%&%&%&%&%&%&%
        print("Image the leaves")
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vmax = np.nanmax(hdu2.data)/2.
        im = ax.matshow(hdu2.data, origin='lower', cmap=plt.cm.Blues, vmax=vmax)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        if 'xlims' in kwargs:
            ax.set_xlim(kwargs['xlims'])
        if 'ylims' in kwargs:
            ax.set_ylim(kwargs['ylims'])
        
        # Make a leaf list
        slist = []
        for c in d.leaves:
            slist.append(c.idx)
            # Plot the actual structure boundaries
            mask = d[c.idx].get_mask()
            mask_coll = np.amax(mask, axis = 0)
            plt.contour(mask_coll, colors='green', linewidths=1, levels = [0])
            # Plot the ellipse fits
            s = analysis.PPVStatistic(d[c.idx])
            ellipse = s.to_mpl_ellipse(edgecolor='black', facecolor='none')
            ax.add_patch(ellipse)
        slist.sort()
        with open(label+'_leaves.txt', "w") as output:
            writer = csv.writer(output)
            for val in slist:
                writer.writerow([val])    
        
        fig.colorbar(im, ax=ax)
        plt.savefig('ab6high_5sig_dendro_plots/'+label+'_leaves_map.pdf', bbox_inches='tight')
        plt.close()

    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
    #     Image the clusters
    #%&%&%&%&%&%&%&%&%&%&%&%&%&%
#     print("Image the resulting clusters")
# 
#     clusts = np.array(dclust.clusters)
#     colors = np.array(dclust.colors)
#     inds = np.argsort(clusts)
#     clusts = clusts[inds]
#     colors = colors[inds]
# 
#     print(clusts)
#     print(colors)
# 
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     vmax = np.nanmax(hdu2.data)/2.
#     im = ax.matshow(hdu2.data, origin='lower', cmap=plt.cm.Blues, vmax=vmax)
#     ax.axes.get_xaxis().set_ticks([])
#     ax.axes.get_yaxis().set_ticks([])
#     if 'xlims' in kwargs:
#         ax.set_xlim(kwargs['xlims'])
#     if 'ylims' in kwargs:
#         ax.set_ylim(kwargs['ylims'])
# 
#     # Make a cluster list
#     f = open(label+'_clusters.txt', 'w')
#     for i, c in enumerate(clusts):
#         f.write('{:<4d} {:7} | '.format(c,colors[i]))
#         # Plot the actual structure boundaries
#         mask = d[c].get_mask()
#         mask_coll = np.amax(mask, axis = 0)
#         plt.contour(mask_coll, colors=colors[i], linewidths=1, levels = [0])
#         # Plot the ellipse fits
#         s = analysis.PPVStatistic(d[c])
#         ellipse = s.to_mpl_ellipse(edgecolor='black', facecolor='none')
#         ax.add_patch(ellipse)
#         # Make sub-lists of descendants
#         print('Finding descendants of cluster ',c)
#         desclist = []
#         if len(d[c].descendants) > 0:
#             for s in d[c].descendants:
#                 desclist.append(s.idx)
#             desclist.sort()
#             liststr=','.join(map(str, desclist))
#             f.write(liststr)
#         f.write("\n")
#     f.close()
# 
#     fig.colorbar(im, ax=ax)
#     plt.savefig(label+'_plots/'+label+'_clusters_map.pdf', bbox_inches='tight')
#     plt.close()
# 
#     return

# -------------------------------------------------------------------------------

def custom_independent(structure,index=None, value=None):
    global cubedata,rmsmap,threshold_sigma
    mom0=0.
    momx=0.
    momy=0.
    ind=structure.indices()
    for i in range(len(ind[0])):
        v=cubedata[ind[0][i],ind[1][i],ind[2][i]]
        mom0=mom0+v
        momx=momx+v*ind[2][i]
        momy=momy+v*ind[1][i]
    momx=int(round(momx/mom0))
    momy=int(round(momy/mom0))
    return structure.values(subtree=True).max() > (threshold_sigma*rmsmap[momy,momx])


def explore_dendro(label='mycloud', xaxis='radius', yaxis='v_rms'):
    d = Dendrogram.load_from(label+'_dendrogram.fits')
    cat = Table.read(label+'_full_catalog.txt', format='ascii.ecsv')
    dv = d.viewer()
    ds = Scatter(d, dv.hub, cat, xaxis, yaxis)
    ds.set_loglog()
    dv.show()
    return

# -------------------------------------------------------------------------------

if __name__ == "__main__":
    run_dendro()
