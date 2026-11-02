#!/usr/bin/env python3 

from pysimdamicm.utils.config import Config
from pysimdamicm.scripts import anaSKImg
from pysimdamicm.utils.units import Units
u=Units()


from astropy.io import fits
import os
import time
import json
from collections.abc import Mapping

from pysimdamicm.dqm.models import *

DISPLAY = False
if DISPLAY:
    from matplotlib import pyplot as plt
DEBUG = False

def build_me_pcd(rawdata, mu, gain, qmin=-2.5, qmax=3.5, bin_width=0.10):

    # mask clusters
    mask = rawdata.mask_clusters.copy()
    # build masked image
    image = np.ma.array(rawdata.image_mean_compressed_pedestal_subtracted_e.astype(float).data, mask=mask)

    # convert to electrons and remove mu
    qe = image - mu/gain

    # do histogram in a fixed bin
    n_bins = int(round((qmax - qmin) / bin_width))
    bin_edges = np.linspace(qmin, qmax, n_bins + 1)
    frequencies, _ = np.histogram(qe.compressed(), bins=bin_edges)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return bin_centers, frequencies



def build_me_image(rawdata, sigma_e, Nccds=4, n_sigmas=3.75):
    nrows,ncols = rawdata.image_mean_compressed_pedestal_subtracted_e.data.shape 

    image = np.zeros((nrows*Nccds,ncols))
    rs = (rawdata.extensions-1)*nrows

    # set to 1 those pixels above charge threshold (1e-, which is defined as, 3.75 x sigma_e)
    image[rs:rs+nrows,:] = ( rawdata.image_mean_compressed_pedestal_subtracted_e > sigma_e * n_sigmas ).astype(float)
    
    # counter: set 1 only pixels for the corresponding extension
    counter = np.zeros((image.shape[0],1))
    counter[rs:rs+nrows,0] = 1

    return image, counter

def build_me_column_occupancy(rawdata, threshold=None, gain=None, sigma_noise=None, n_sigmas=3.75):
    """Number of pixels above a fixed charge threshold in each column (multiplicity column).
    """
    if sigma_noise is None:
        sigma_noise = rawdata.pedestal_sigma[f"EXT{rawdata.extension}"][0].mean() / gain
    if threshold is None:
        threshold = n_sigmas * sigma_noise

    # cluster mask (include overscan in the column multiplicity ME)
    mask = rawdata.mask_clusters.copy()
    pix_1st_ovs   = rawdata.ncols - rawdata.n_cols_overscan
    # overscan
    mask[:,pix_1st_ovs:] = False
    # prescan
    mask[:,:rawdata.n_cols_prescan] = False

    # build masked image
    image = np.ma.array(rawdata.image_mean_compressed_pedestal_subtracted_e.astype(float).data, mask=mask)

    # get the column multiplicity
    hits = image > threshold
    mcol = hits.sum(axis=0)
    
    # normalize and correct from masked
    n_valid = (~image.mask).sum(axis=0)
    fraction = mcol / n_valid
    scaled = fraction * image.shape[0]
    
    return scaled, np.linspace(1, rawdata.ncols, rawdata.ncols).astype(int)

def build_me_cti(rawdata,npix_active=100):
    q_mean_col = np.mean(rawdata.image_mean_compressed_pedestal_subtracted_e.astype(float), axis=0)

    pix_1st_ovs = rawdata.ncols-rawdata.n_cols_overscan
    q_1st_ovs   = q_mean_col[pix_1st_ovs]
    q_active    = np.mean( q_mean_col[pix_1st_ovs-npix_active:pix_1st_ovs] )
    q_ovs       = np.mean( q_mean_col[pix_1st_ovs+1:] )
    numerator   = q_1st_ovs - q_ovs
    denumerator = q_active  - q_ovs
    cti         = numerator / denumerator 

    sigma_active      = np.std( q_mean_col[pix_1st_ovs-npix_active:pix_1st_ovs] )
    sigma_ovs         = np.std( q_mean_col[pix_1st_ovs+1:] )
    sigma_numerator   = np.sqrt(2) * sigma_ovs
    sigma_denumerator = np.sqrt(sigma_active**2 + sigma_ovs**2)
    sigma_cti         = np.abs(cti * np.sqrt((sigma_denumerator)**2 + (sigma_numerator)**2))

    return (cti,sigma_cti)


def fill_metadata(inimage, tag, runid, extid):
    """
    """
    inputimage = inimage.split("/")[-1]
    try:
        moduleid = int(fits.getval(inimage,"ACMID"))
    except KeyError:
        moduleid = int(inputimage.split(".")[0].split("_")[-4])    
    try:
        level = int(inputimage.split(".")[0].split("_")[-5].split("L")[-1])
    except ValueError:
        level = "2"
    imageid = int(inputimage.split(".")[0].split("_")[-1])
    metadata = MetaData(
                    tag    = tag,
                    run    = runid, 
                    module = moduleid, 
                    ccd    = extid, 
                    image  = imageid, 
                    file   = fits.getval(inimage,"BINNAME"),
                    rows   = fits.getval(inimage,"NROW"),
                    cols   = fits.getval(inimage,"NCOL"),
                    skips  = fits.getval(inimage,"NDCM"),
                    npbin  = fits.getval(inimage,"NPBIN"),
                    nsbin  = fits.getval(inimage,"NSBIN"),
                    rolevel= level)
    print(f"INFO. Module(ACM)-{metadata.module}, Image-{metadata.image}, with readout level L{metadata.rolevel}")
    return metadata


def _load_quality_config(config_or_path):
    """Return a quality configuration from a Mapping or a JSON path."""
    if isinstance(config_or_path, Mapping):
        return config_or_path
    with open(config_or_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_me_collection(rawdata, metadata, jsonme):
    collection = MECollection(metadata=metadata)
    
    ####  in days
    dt = (metadata.end - metadata.start).total_seconds() / 86400. / 2.
    t  = (metadata.start - metadata.tzero).total_seconds() / 86400. + dt
    dcfit_results = getattr(rawdata,f"me_dcfit_results_EXT{metadata.ccd}")
    dcfit_goodness = dcfit_results['chi2'] # <4 ?? should we use this as test??
    
    ############################################################################### EVOLUTION ME
    ############################################ PER ROW PEDESTAL MEAN
    meprpm        = MERowPedestalMean()
    meprpm.x      = [t]
    meprpm.xerror = [dt]
    meprpm.y      = [rawdata.pedestal_mu[f"EXT{metadata.ccd}"][0].mean()]
    meprpm.yerror = [rawdata.pedestal_mu_error[f"EXT{metadata.ccd}"][0].mean()]
    collection.me_list.append(meprpm)

    ############################################ DARK CURRENT
    medc        = MEDarkCurrent()
    medc.x      = [t]
    medc.xerror = [dt]
    medc.y      = [dcfit_results['dc'][0]]
    medc.yerror = [dcfit_results['dc'][1]]
    collection.me_list.append(medc)

    ############################################ CALIBRATION
    megain        = MEGain()
    megain.x      = [t]
    megain.xerror = [dt]
    megain.y      = [dcfit_results['calibration'][0]]
    megain.yerror = [dcfit_results['calibration'][1]]
    collection.me_list.append(megain)

    ############################################ READOUT NOISE
    meser        = MEReadoutNoise()
    meser.x      = [t]
    meser.xerror = [dt]
    meser.y      = [dcfit_results['sigma'][0]]
    meser.yerror = [dcfit_results['sigma'][1]]
    collection.me_list.append(meser)
    
    ############################################ ZERO ELECTRON PEAK CENTER
    mezep        = MEZeroPeakCenter()
    mezep.x      = [t]
    mezep.xerror = [dt]
    mezep.y      = [dcfit_results['mu0'][0]]
    mezep.yerror = [dcfit_results['mu0'][1]]
    collection.me_list.append(mezep)

    ############################################ CTI using overscan
    cti, cti_error = build_me_cti(rawdata)
    mecti        = MECTI()
    mecti.x      = [t]
    mecti.xerror = [dt]
    mecti.y      = [cti]
    mecti.yerror = [cti_error]
    collection.me_list.append(mecti)

    ############################################################################### STATIC ME
    ############################################ PIXEL CHARGE DISTRIBUTION (LOW-E)
    x,y          = build_me_pcd(rawdata, dcfit_results['mu0'][0], dcfit_results['calibration'][0])
    mepcd        = MEPCD()
    mepcd.x      = np.array(x).astype(float)
    mepcd.xerror = np.zeros_like(mepcd.x)
    mepcd.y      = np.array(y).astype(int)
    mepcd.yerror = np.sqrt(mepcd.y)
    collection.me_list.append(mepcd)
    
    ############################################ SPECTRUM : Emin, Emax, Nbins  FIXME: hardcodded params for the high-E spectrum!
    Nbins, Emin, Emax = 499, 1, 500
    hist, bin_edges = np.histogram(rawdata.me_reconstructed_spectrum, bins=Nbins, range=(Emin,Emax))
    mespec        = MESpectrum()
    mespec.x      = (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist()
    mespec.xerror = np.diff(bin_edges)
    mespec.y      = hist
    mespec.yerror = np.sqrt(hist)
    collection.me_list.append(mespec)

    ############################################ PER-ROW PEDESTAL
    meprp        = MEPerRowPedestal()
    meprp.x      = np.linspace(1, rawdata.nrows, rawdata.nrows).astype(int)
    meprp.xerror = np.zeros_like(meprp.x)
    meprp.y      = rawdata.pedestal_mu[f"EXT{metadata.ccd}"][0].flatten()
    meprp.yerror = rawdata.pedestal_mu_error[f"EXT{metadata.ccd}"][0].flatten()
    collection.me_list.append(meprp)

    ############################################ PER-ROW PEDESTAL
    meprr        = MEPerRowResolution()
    meprr.x      = np.linspace(1, rawdata.nrows, rawdata.nrows).astype(int)
    meprr.xerror = np.zeros_like(meprp.x)
    meprr.y      = rawdata.pedestal_sigma[f"EXT{metadata.ccd}"][0].flatten() / dcfit_results['calibration'][0]
    meprr.yerror = rawdata.pedestal_sigma_error[f"EXT{metadata.ccd}"][0].flatten() / dcfit_results['calibration'][0]
    collection.me_list.append(meprr)

    ############################################ COLUMN MULTIPLICITY 
    multiplicity, cols = build_me_column_occupancy(rawdata, gain=dcfit_results['calibration'][0], sigma_noise=dcfit_results['sigma'][0])
    memcol        = MEColumnOccupancy()
    memcol.x      = cols
    memcol.xerror = np.zeros_like(memcol.x)
    memcol.y      = multiplicity
    memcol.yerror = np.sqrt( multiplicity )
    collection.me_list.append(memcol)

    ############################################ COLUMN MULTIPLICITY 
    image,counter  = build_me_image(rawdata, dcfit_results['sigma'][0])
    meimage        = MEMeanImage()
    meimage.x      = None
    meimage.xerror = None
    meimage.y      = image
    meimage.yerror = None
    meimage.counter=counter
    collection.me_list.append(meimage)


    if DISPLAY or DEBUG:
        i = 0
        for me in collection.me_list:
            if DISPLAY:
                #print(me)
                print(me.name)
                plt.figure(i)
                if me.name == 'MEPCD':
                    plt.yscale('log')
                if me.name == 'MEMeanImage': 
                    plt.imshow(me.y, origin='lower',aspect='auto')
                    plt.colorbar()
                else:
                    plt.errorbar(me.x,me.y, xerr=me.xerror, yerr=me.yerror, linestyle='None', ms=4, marker='s', color='k')
                plt.xlabel(me.xlabel)
                plt.ylabel(me.ylabel)
                plt.title(me.title +f" [kind={me.kind}]")
                i +=1
            if DEBUG:
                print(me)
            if DISPLAY:
                plt.show(block=True)
   
    # QUALITY IMAGE FLAG
    metadata.set_detector_observables(gain=dcfit_results['calibration'][0], noise=dcfit_results['sigma'][0], dc=dcfit_results['dc'][0])
    # --------------------------------------------------------------------- QUALITY TESTS
    # ``jsonme`` may be either a path or a configuration loaded once by the
    # caller. Passing the Mapping avoids reopening the same JSON for every image.
    quality_config = _load_quality_config(jsonme)
    metadata.evaluate_image_quality_from_config(quality_config, npix= ((rawdata.nrows*rawdata.ncols-np.sum(rawdata.mask_clusters))*rawdata.image_header['NPBIN']*rawdata.image_header['NSBIN']) )
    print(f"Quality Image: {metadata.gain_quality} {metadata.noise_quality} {metadata.dc_quality} | {metadata.image_quality}")
    return collection


