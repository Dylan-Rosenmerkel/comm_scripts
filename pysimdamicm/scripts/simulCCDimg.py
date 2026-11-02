""":mod:`pysimdamicm.simulCCDimg`

  Main script to process Geant4-based simulations of the DAMIC-M detector.

  Any process from :obj:`pysimdamicm.detector_response` and :obj:`pysimdamicm.reconstruction` can be
  included into a configuration JSON file.


  Using the class :obj:`pysimdamicm.utils.config.Config`, all process (as well as its
  configuration) will be loaded from the JSON file, and set the ProcessManager
  :obj:`pysimdamicm.process_manager.ProcessManager`.



  Note
  ----
  Use the script psimulCCDimg to rum from the terminal (see documentation)

  .. moduleauthor:: Nuria Castello-Mor
"""


##### module DEBUG VARIABLES
__verbose__=False
__debugs__=False

__debug_level__ = 1
__debug_param__="Edep"

##### BRANCHES NAMES from Geant4 sims
__geant4_branches__ = {
    'ccdout': 'CCDOut',
    'evtout': 'EventOut'
    }

from memory_profiler import profile

import time
import numpy as np
import pandas as pd
import array
import random
import ROOT
ROOT.PyConfig.StartGUIThread = False
#ROOT.gROOT.SetBatch(True)

ROOT_VERSION = int(ROOT.gROOT.GetVersion().split('/')[0].replace('.',''))

from matplotlib import use,get_backend
from matplotlib import rc
rc('xtick', labelsize=12) 
rc('ytick', labelsize=12) 
rc('font',**{'family':'DejaVu Sans','serif':['Times'],'size':13})
rc('text', usetex=True)
use(get_backend())


from pysimdamicm.process_manager import ProcessManager
from pysimdamicm.io.data_formats import OutputDataManager,G4HitCollection,Cluster,PixelizedEvent,G4InfoTree,ProcessConfigTree
from pysimdamicm.utils.units import Units

u = Units()

import sys
def progressbar(it, prefix="", size=60, file=sys.stderr):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def convert_into_pandas_oldROOT(tree):

    _df_evt = {}
    for b in tree.GetListOfLeaves():  
        bname = b.GetName()
        _df_evt[bname]=np.array(getattr(tree,bname))
    return pd.DataFrame(_df_evt)


def prepare_tree_reading(tree):

    if ROOT_VERSION<622:
        return tree

    _df = {}
    for b in tree.GetListOfBranches():
        bname = b.GetName()
        try:
            dtype = 'ROOT.std.{}()'.format(b.GetTypeName().replace('<',"('").replace('>',"')"))
            _df[bname] = eval(dtype)
        except AttributeError:
            _df[bname] = array.array( b.GetLeaf(bname).GetTypeName().lower()[0],[0])

        tree.SetBranchAddress(bname,_df[bname])

    return _df

def convert_into_pandas(bdict):
     
    if ROOT_VERSION<622:
        _pd = convert_into_pandas_oldROOT(bdict)
        return _pd
    
    _pd = {}
    for k,v in bdict.items():
        try:
            _pd[k] = np.ndarray((len(v),), buffer=np.array(v.data()).astype(float))
        except AttributeError:
            _pd[k] = [v[0]]
        
    _pd['EventID'] = _pd['EventID']*len(_pd['pdg'])
    
    return pd.DataFrame(_pd)

def main(config,NPIX_MIN=3,event_to_debug=None,Nmax=-999):
    
    # Read the random seed from the configuration file.
    configured_seed = config.configuration.get("CF", {}).get("seed")
    seed = (
            int(configured_seed)
            if configured_seed is not None
            else random.SystemRandom().randint(1, 2**32 - 1)
            )

    # Store the effective seed for reproducibility.
    config.configuration.setdefault("CF", {})["seed"] = seed
    
    # Seed the legacy NumPy random-number generator used by Diffusion.
    np.random.seed(seed)

    # Seed Python's standard random module for other detector processes.
    random.seed(seed)

    # Seed ROOT as well, in case ROOT random generators are used later.
    if hasattr(ROOT, "gRandom"):
        ROOT.gRandom.SetSeed(seed)
    print(f" -- INFO. Random seed: {seed}")

    # Continue with the existing main function.
    if __debugs__:
        ProcessManager.__debugs__ = __debugs__
        if __debug_level__>1:
            ProcessManager.__debug_level__=__debug_level__
        debug_plots=__debug_plots
    else:
        debug_plots=lambda *args,**kwargs: None

    if __verbose__:
        ProcessManager.__verbose__ = __verbose__
        print( "\n  --- Setting Process Manager")


    ### Set to Singleton Units some attributes needed along the process
    for attr in ["ccd_shape","ccd_pixel_size_x","ccd_pixel_size_y","ccd_thickness",
            "n_cols_overscan","n_cols_prescan","n_rows_overscan","n_rows_prescan",
            "e2eV", "ADC2eV", "detector_name","img_exp_time","invert_ccd_z_values"]:
        try:
            print( " Setting value of ", attr, " to ", config.configuration["CF"][attr])
            setattr(u,attr,config.configuration["CF"][attr])
        except KeyError:
            continue
    
    ### z_offset for Clusterization output?
    try:
        print(" Setting value of z_offset to ",config.configuration["CF"]["z_offset"])
        z_offset = config.configuration["CF"]["z_offset"]*u.mm
    except KeyError:
        z_offset = None

    ### Process Manager
    ###   set the Process Chain according to JSON configurations
    pman = ProcessManager()
    pman.set_configuration(config)
    print("main INFO. Process sequence (objects):\n ", pman.__sequence__, "\n")

    if __verbose__:
        print( "\n  --- Setting Output Manager")

    ### Output Manager
    ###     output root file name
    outman = OutputDataManager(config.configuration["rootfiles"]["out"])
    print(" -- INFO. Output root file: ", config.configuration["rootfiles"]["out"])

    ### If total number of input files is larger than one, only one output ROOT file will be created
    #       when total number of files is 1, older behavior 
    NbeamOn_total = 0
    t3 = time.time()
    g4info = []
    for n_file,infile in enumerate(config.configuration["rootfiles"]["in"]):
        print(" -- INFO. Input root file: ", infile)
        
        print( "\n --- Loading geant4 data")
        ### EVENTS REGISTERED ON THE SENSITIVE DETECTOR
        t0 = time.time()
        froot = ROOT.TFile.Open(infile)
        uccdout = froot.Get("CCDOut")
        ccdobjs = prepare_tree_reading(uccdout)
        ### Events to process?
        Nhits = uccdout.GetEntries("Edep>0")

        ### SIMULATED EVENTS
        t1 = time.time()
        uevtout = froot.Get("EventOut")
        NbeamOn = uevtout.GetEntries()
        uevtout.BuildIndex("EventID")
        eventobjs = prepare_tree_reading(uevtout)
        
        print("\n --- Reconstruction process starts:")
        time_get_img,time_cf = [],[]
        
        # initialize hits to zero, when no clusters is found this will not be created 
        #   but is still needed to call close_process
        hits = None
        Nclusters = 0
        #for k_ind,abs_evt in enumerate(sorted(set(df_uccdout.EventID.values))):
        for k_ind in progressbar(range(uccdout.GetEntries()), "Processing events: ", 40):
        #for k_ind in range(uccdout.GetEntries()):
            
            # break the for as the file has no hits to be processed
            if Nhits==0:
                break

            _ = uccdout.GetEntry(k_ind)
            reg_evt = convert_into_pandas(ccdobjs)
            try:
                abs_evt = reg_evt['EventID'][0]
            except KeyError:
                # the event has no hits, ignore it
                continue

            if not reg_evt['Edep'].sum()>0:
                print("skipping event, no hits!!!!")
                continue
            
            _ = uevtout.GetEntryWithIndex(int(abs_evt))
            reg_sim = convert_into_pandas(eventobjs)

            ### group event hits by timing: in equal exposure time
            img_time = (reg_evt['time']/u.img_exp_time).astype(int)

            evt_clusters  = []
            evt_pixelized = []
            all_killed = np.zeros_like(list(set(reg_evt.CCDid.values)), dtype=bool)
            for _id,ccd in enumerate(set(reg_evt.CCDid.values)):
                if __debugs__:
                    print( " --- clusters for event,ccd={},{}".format(abs_evt,ccd))

                ### process each image (group events happining within time window)
                #       as separate events
                for img in set(img_time):

                    event_tuple = (abs_evt,ccd)
                    ### events for event in a given CCD
                    mask_ccd = reg_evt['CCDid']==ccd

                    ### join image mask with the ccd mask
                    #       look for hits in a CCD within the same spectral window
                    mask_img = np.logical_and(img_time==img, mask_ccd)
                    if not np.any(mask_img==True):
                        continue

                    hits = G4HitCollection(event_tuple,reg_evt,mask=mask_img,z_offset=z_offset)
                    hits.AddCoordSimulatedEvent(reg_sim)
                    selected_event = (
                            event_to_debug is None
                            or int(abs_evt) in np.atleast_1d(event_to_debug).astype(int)
                            )
                    hits.debug_capture = bool(__debugs__ and selected_event)

                    setattr(hits,'outfile',config.configuration["rootfiles"]["out"])

                    ### detector response for this hit collection
                    # XXX FutureWarning scipy/stats/_binned_statistic.py XXX
                    _ = pman.execute_process(hits)
                    
                    if pman.active_ClusterFinder and not hits.killed:
                        ### define Cluster object for each cluster in (abs_evt,ccd) hits collection
                        for i,mask in hits.cluster_mask.items():
                            evt_clusters.append(Cluster(i,hits,mask))
                            evt_clusters[-1].get_cluster_properties(is_simulation=True)
                            if __verbose__:
                                print(" Cluster total energy: {} keV".format(evt_clusters[-1].Energy))
                                input("Press Enter to continue...")
                       
                            Nclusters +=1
                        try:
                            time_get_img.append(hits.time_gti)
                            time_cf.append(hits.time_cf)
                        except AttributeError:
                            # time attributes do not exists: DC and Noise habe not been activated
                            pass
                    
                    if not hits.killed:
                        ### if debug mode plot image, and individual process signals
                        debug_plots(hits,Npix_min=NPIX_MIN,only_this_event=event_to_debug)
                        ## DEBUG: Plot only events with >= 2 clusters and z below threshold
                        #z_max_um = 660.0   # Change this threshold (in micrometers)
                        #n_clusters = len(getattr(hits, "cluster_mask", {}))
                        #z_um = np.asarray(hits.g4_posz, dtype=float) / u.um
                        #E_total_keV = np.sum(hits.g4_Edep) / 1000.0  # eV -> keV

                        #if (
                        #        __debugs__
                        #        and n_clusters > 2
                        #        and z_um.size > 0
                        #        and np.all(z_um > z_max_um)
                        #        ):
                        #    print(
                        #            f"[DEBUG] Event {int(abs_evt)}, CCD {int(ccd)}: "
                        #            f"{n_clusters} clusters, "
                        #            f"z = [{z_um.min():.1f}, {z_um.max():.1f}] um"
                        #            )
                        #    
                        #    debug_plots(
                        #            hits,
                        #            Npix_min=NPIX_MIN,
                        #            only_this_event=event_to_debug
                        #            )

                        ##### Add pixelized (event,ccd)
                        evt_pixelized.append(PixelizedEvent(hits))
                    else:
                        all_killed[_id] = True

            ### fill clusters Tree only once per event
            if all_killed.sum() != all_killed.size:
                if pman.active_ClusterFinder:
                    outman.fill_tree("clustersRec",abs_evt+NbeamOn_total,evt_clusters)
                outman.fill_tree("pixelizedEvent",abs_evt+NbeamOn_total,evt_pixelized)
       
        ## closing EventOut tree
        froot.Close()

        # Add the number of simulations to the total value
        NbeamOn_total +=NbeamOn
        if pman.active_ClusterFinder:
            print(" -- file {}: Ncls = {}/{} sims \n".format(n_file+1,Nclusters,NbeamOn))
        ## Include one Entry for each file
        outman.fill_tree("geant4_config",0,[G4InfoTree(infile,n_file+1,Nclusters)])

   
    ### Print some statistics 
    print("\n --- Time statistics: ")
    if len(time_cf)>0:
        print("  - Internal process (related to intrinsic noise): ")
        print("     - Mean time for clustering algorithm: ", np.mean(time_cf),      "sec std=",np.std(time_cf), " T=",np.sum(time_cf)/60.," min")
        print("     - Mean time for Filtering CCD image:  ", np.mean(time_get_img), "sec std=",np.std(time_get_img), " T=",np.sum(time_get_img)/60., " min")

    tend = round((time.time()-t3)/60.,2)
    print("  - Reconstruction processes done in {} min (for processing {} events)\n".format(tend,k_ind))
    
    # Write a Tree with the configuration parameters for each process
    outman.fill_tree("process_config",0,[ProcessConfigTree(pman.__sequence__)])
    ### 

    #### Close all process
    if hits is not None:
        print("")
        pman.close_process(hits)

    #### close output root file
    outman.close()

    print("\n Output Root file: {}\n".format(outman.file_name))
    
    return


def __debug_plots(hits,Npix_min,only_this_event):
    """
    """
    hits.Draw(Npix_min,only_this_event)

