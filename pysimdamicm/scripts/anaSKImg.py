""":mod:`pysimdamicm.anaSKImg`

  .. moduleauthor:: Nuria Castello-Mor
"""

from pysimdamicm.io import rawdata
from pysimdamicm.utils import libplot4ana as libplot
from pysimdamicm.process_manager import ProcessManager
from pysimdamicm.io.data_formats import OutputDataManager, Cluster, InfoFromReconstruction, ProcessConfigTree
from pysimdamicm.utils.get_info_from_file_names import get_image_and_run_number, get_ccd_id_number
from pysimdamicm.utils.units import Units
u=Units()
##### modules

from matplotlib import use,get_backend
use(get_backend())

from matplotlib import pyplot as plt

import numpy as np
from astropy.io import fits
import warnings
warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)


def main(infiles,config,outdir=None,display=True,run_id=None,
        image_id=None,ccd_id=None, image_mask=None,
        out_root_file=None, MCText=None, run_tag=None):
    """
    """

    # sanity checks
    if not type(infiles) is list:
        raise RuntimeError("<anaSKImg> only accepts a list of input files!")

    ### Process Manager
    ############################################################################
    print(*["\n","main INFO. Process Manager initialization"])
    pman = ProcessManager()
    pman.set_configuration(config)
    print("main INFO. The following process will be applied:")
    print(*pman.__sequence__,sep="\n")
    
    ### Output Manager
    ############################################################################
    if pman.active_ClusterFinder:
        print("main INFO. Set output manager for clusterization")
        if out_root_file is None:
            if type(config.configuration['input']["image"]["extensions"])==list and config.configuration["input"]["image"]["ACM_multi_extension"]:
                _ext_naming = "".join(map(str,config.configuration['input']["image"]["extensions"]))
            else:
                _ext_naming = config.configuration['input']["image"]["extensions"]
            out_root_file = "panaSKImg_clustersRec_{}_ext{}.{}".format(infiles[0].split("/")[-1].split(".")[0],_ext_naming,"root")

        out_root_file = "{}/{}".format(outdir['avg'],out_root_file)
        outman = OutputDataManager(out_root_file)
    
    ### Image to mask pixels during clusterization
    if image_mask is not None:
        print("main INFO. Adding file to mask pixels during cluster reconstruction ", image_mask)
        mask_data = fits.getdata(image_mask,ext=1).astype(bool)
    else:
        mask_data = None

    ### Process each file within the input list
    ############################################################################
    Ncls_total=0
    print("main INFO. CCD Image processing starts ......  \n")
    for n_image, _file in enumerate(infiles):
        print("main INFO. Processing file ", _file)
        ### create RawData object
        rdata = rawdata.BuilderRawData(_file, config.configuration['input'])
        rdata.prepare_data()
        ### set display 
        rdata.__display__ = display

        ### add output directories, and set run number
        rdata.output = outdir
        
        ### UPDATE IMAGE, RUN, CCD NUMBER 
        _image_id,_run_id,isSource = get_image_and_run_number(_file)
        #### ADD IMAGE NUMBER FROM FILE
        rdata.n_image_from_file = _image_id
        # ... run number is send by the user (by command line value), if not use the one from the
        #           input file name
        rdata.n_run = int(run_id) if run_id is not None else int(_run_id)
        run_id = int(run_id) if run_id is not None else int(_run_id)
        # ... same for the image ID, if len(infiles)>1
        if len(infiles)>1:
            rdata.n_image = n_image if _image_id==0 else int(_image_id)
        else:
            rdata.n_image = image_id if image_id is not None else int(_image_id)
        # ... update the ccd id, if not use the one by default
        if ccd_id is not None:
            rdata.ccd = int(ccd_id)
        else:
            rdata.ccd = rdata.ccd if get_ccd_id_number(_file) is None else get_ccd_id_number(_file)
        
        # add run tag for the mongoDB data
        rdata.run_tag = run_tag
        
        print("......... run {}, image {} (from file {}), ccd {}, with tag {}".format(rdata.n_run,
            rdata.n_image, rdata.n_image_from_file, rdata.ccd, rdata.run_tag))

        ### add information to tohdr that later on will be added to the fits file header (only during DQM ) -- DEPRECATED
        # u._tohdr.update({'isSource':(int(isSource),'DQM: If string <Source> is found in the fits file name')})

        ### add mask as a data member (used during clusterization)
        if image_mask is not None:
            rdata.mask = mask_data.copy()
        
        ### add Monte Carlo Truth Image
        if MCText is not None:
            ## assuming image in units of eV
            rdata.mct_image = fits.getdata(_file,ext=MCText)
            rdata.mct_cls_id = fits.getdata(_file,ext=MCText+1)

        ### Run all active process over rdata
        _ = pman.execute_process(rdata)
        
        if pman.active_ClusterFinder:
            ### get some needed attributes
            if hasattr(rdata,'image_header'):
                try:
                    isBAD = np.logical_or(bool(rdata.image_header['MEBS']), bool(rdata.image_header['MESC']))
                except KeyError:
                    isBAD = False
            else:
                isBAD = False
            
            Nclusters    = rdata.Nclusters
            evt_clusters = rdata.evt_clusters

            ### create an entry for each file
            print("")
            if not isBAD:
                print("main INFO. Run {}, Image {}: {} clusters >>>> clustersRec".format(run_id,rdata.n_image,Nclusters))
                outman.fill_tree("clustersRec",run_id,evt_clusters,isdata=True)

                print("main INFO. Add ROOT TTree with reconstrcted information >> info")
                outman.fill_tree("info",run_id,[InfoFromReconstruction(rdata)],isdata=True)
            else:
                print("main INFO. Run {}, Image {}: {} clusters >>>> clustersRec_tagged".format(run_id,rdata.n_image,Nclusters))
                outman.fill_tree("clustersRec_tagged",run_id,evt_clusters,isdata=True)

                print("main INFO. Add ROOT TTree with reconstrcted information >> info_tagged")
                outman.fill_tree("info_tagged",run_id,[InfoFromReconstruction(rdata)],isdata=True)

        if display:
            print("main INFO. Displaying plots ... ")
            plt.show(block=True)

        if config.configuration['input']['image']['save_images']:
            image_list = []
            names      = []
            possible_calibrated_images = [
                    ('image_mean_compressed_pedestal_subtracted_e',  'calibrated'),
                    ('image_median_compressed_pedestal_subtracted_e','calibrated'),
                    ('image_median_compressed_pedestal_subtracted_e','calibrated'),
                    ('image_mean_compressed_e',                      'calibrated'),
                    ('image_median_compressed_e',                    'calibrated'),
                    ]
            possible_pedsubtracted_image = [
                    ('image_mean_compressed_pedestal_subtracted',    'pedsubtracted'),
                    ('image_median_compressed_pedestal_subtracted',  'pedsubtracted')
                    ]
            basic_images = [
                    ('image_mean_compressed',                        'mean'),
                    ('image_median_compressed',                      'median')
                    ]
            possible_int_images = [
                    ('mask_clusters',                                'fullmask'),
                    ('mask_clusters_only',                           'clsmask')
                    ]
            possible_other_images = [
                    ('image_std_compressed',                         'std'),
                    ('image_saturated_pixels',                       'pixsaturation'),
                    ('image_xtalk',                                  'crosstalk'),
                    ('mask_clusters_ids',                            'qmax')
                    ]
            
            print("main INFO. output fits file:")
            for image_name,naming in possible_calibrated_images:
                if hasattr(rdata,image_name):
                    print(f"    - Image {image_name} added with EXTNAME {naming}")
                    image_list.append(image_name)
                    names.append(naming)
            if not 'calibrated' in names:
                for image_name,naming in  possible_pedsubtracted_image:
                    if hasattr(rdata,image_name):
                        print(f"    - Image {image_name} added with EXTNAME {naming}")
                        image_list.append(image_name)
                        names.append(naming)
            if not  'calibrated' in names and not 'pedsubtracted' in names:
                for image_name,naming in basic_images:
                    if hasattr(rdata,image_name):
                        print(f"    - Image {image_name} added with EXTNAME {naming}")
                        image_list.append(image_name)
                        names.append(naming)

            for image_name,naming in possible_int_images:
                if hasattr(rdata,image_name):
                    print(f"    - Image {image_name} added with EXTNAME {naming}")
                    image_list.append(image_name)
                    names.append(naming)
                    break

            for image_name,naming in possible_other_images:
                if hasattr(rdata,image_name):
                    print(f"    - Image {image_name} added with EXTNAME {naming}")
                    image_list.append(image_name)
                    names.append(naming)
            if rdata.ACM and len(rdata.amplifier.keys())>1:
                for amp_name in rdata.amplifier.keys():
                    naming = f"MCCD{amp_name}"
                    setattr(rdata,f"mask_active_region_{amp_name}",rdata.amplifier[amp_name]['mask_image_active_region'])
                    print(f"    - MASK for extension CCD_{amp_name} added with EXTNAME {naming}")
                    image_list.append(f"mask_active_region_{amp_name}")
                    names.append(f"MCCD{amp_name}")

            if hasattr(rdata,"extname"):
                rdata._multiext_waders_image = rdata.output+"{}_waders.fits".format(rdata.extname)
            else:
                rdata._multiext_waders_image = rdata.output+"EXT{}_waders.fits".format(rdata.extension)
            
            try:
                bz2_format = config.configuration['input']['image']['bz2']
            except KeyError:
                bz2_format = False
            
            rdata.SaveAsFits(rdata._multiext_waders_image,image_list,naming=names, bz2=bz2_format)
            outfilename = rdata._multiext_waders_image if not bz2_format else rdata._multiext_waders_image.replace("fits","bz2")
            print(f"  Images save as multi extension fits file: {outfilename}\n")

    if pman.active_ClusterFinder:
        outman.fill_tree("process_config",0,[ProcessConfigTree(pman.__sequence__)])
        print("main INFO. Reconstructed Cluster at: ",out_root_file)
        outman.close()

    return rdata


def draw_PCD_for_single_skip_images(rdata):
    """
    """
    cmap = plt.cm.get_cmap(plt.cm.viridis,500)
    for skip in range(0,500,1):
        libplot.plot_pixel_charge_distribution(in_rdata.image_skips[:,skip::500],title='individual skip measurements',
            alpha=0.4, color=cmap(skip-1), histtype='step')
        _ = plt.hist(in_rdata.image.compressed(), bins=938, color='r', histtype='step')

    return
def draw_PCD_and_2Dimage(rdata):
    """
    """
    ### PIXEL CHARGE DISTRIBUTION
    images = {'averaged skips': rdata.image}
    if hasattr(rdata,'image_skips'):
        images['single skips'] = rdata.image_skips
        
    for kimg in images.keys():
        libplot.plot_pixel_charge_distribution( images[kimg], norm=True, histtype='step',
                label=kimg)

    ### IMAGE BEFORE AND AFTER COMPRESSING SINGLE SKIP MEASUREMENTS
    vmin=rdata.image.data.min()
    vmax=rdata.image.data.max()
    for kimg in images.keys():
        libplot.plot_image( images[kimg], figtitle=kimg, vmin=vmin, vmax=vmax )





