import ROOT
import numpy as np
from array import array

import time
from scipy.spatial.distance import cdist 
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
rc('xtick', labelsize=12) 
rc('ytick', labelsize=12) 
rc('font',**{'family':'DejaVu Sans','serif':['Times'],'size':13})
rc('text', usetex=True)

from pysimdamicm.utils.plotLib import particle_colors,particle_markers
from pysimdamicm.io.G4utils import G4Volume
from pysimdamicm.utils.units import Units
from pysimdamicm import __version__,__commit__

from collections import deque


#from pysimdamicm.utils.root_plot_styles import damicmStyle
#style = damicmStyle()
#style.cd()
#ROOT.gROOT.ForceStyle()

u = Units()

__python2C_type__ = {
        int:int, 
        np.int8:int, np.int16:int, np.int32:int, np.int64:int,
        float:float, 
        np.float16:float, np.float32:float, np.float64:float, np.float128:float,
        np.double:float, ROOT.std.string:ROOT.std.string
        }


class ValidationError(Exception):
    def __init__(self,message):
        super().__init__(message)


def as_native_int(value, name):
    """Convert a scalar endpoint coordinate to a native Python int."""

    array = np.asarray(value)

    if array.ndim != 0:
        raise TypeError(
            f"{name} must be scalar, received shape {array.shape}"
        )

    scalar = array.item()

    if not np.isfinite(scalar):
        raise ValueError(
            f"{name} must be finite, received {scalar!r}"
        )

    return int(scalar)


class OutputDataManager(object):
    """Class to manage the output ROOT files: create and fill different TTrees
    """
    
    __response_detector_tree_name__ = 'detector_config'
    __clusters_tree_name__ = 'clusters'
    
    def __init__(self,file_name, fmode="recreate", store_pixels=False,
            has_clusters_true=True,has_clusters_rec=False):
        
        self.store_pixels = False
        self.has_clusters_true=True
        self.has_clusters_rec=False
        self.file_name = file_name

        self.tfile = ROOT.TFile(self.file_name,fmode)       

    def create_cluster_collection_tree(self,tree_name,cluster_obj,isdata=False):
        """
        """

        ### Add tree for the cluster properties (one entry per event)
        self.tfile.cd()

        setattr(self,"{}_int_br_keywords".format(tree_name),[])
        int_br_keywords = getattr(self,"{}_int_br_keywords".format(tree_name))

        setattr(self,"{}_vec_br_keywords".format(tree_name),[])
        vec_br_keywords = getattr(self,"{}_vec_br_keywords".format(tree_name))

        setattr(self,"{}_vec_vec_br_keywords".format(tree_name),[])
        vec_vec_br_keywords = getattr(self,"{}_vec_vec_br_keywords".format(tree_name))

        setattr(self,"{}_tree_cluster_collection".format(tree_name), ROOT.TTree(tree_name,tree_name))
        tree_cluster_collection = getattr(self,"{}_tree_cluster_collection".format(tree_name))
        
        if tree_name in ["clustersRec","pixelizedEvent","info","clustersRec_tagged","info_tagged"]:
            ### Branches in the cluster collection tree
            if isdata:
                name_id = "RUNID"
            else:
                name_id = "event"

            setattr(self,"{}_{}".format(tree_name,name_id), np.array([-1],int))
            evt = getattr(self,"{}_{}".format(tree_name,name_id))
            
            setattr(self,"{}_branches".format(tree_name),{"event":tree_cluster_collection.Branch(name_id,evt,"{}/I".format(name_id))})           
            branches_dict = getattr(self,"{}_branches".format(tree_name))
            #self.branches = {
            #        "event":    self.tree_cluster_collection.Branch("event",self.event,"event/I")
            #        }
        else:
            ### Branches in the cluster collection tree
            setattr(self,"{}_id".format(tree_name), np.array([-1],int))
            idevt = getattr(self,"{}_id".format(tree_name))
            
            setattr(self,"{}_branches".format(tree_name),{"id":tree_cluster_collection.Branch("id",idevt,"id/I")})
            branches_dict = getattr(self,"{}_branches".format(tree_name))
            #self.branches = {
            #        "event":    self.tree_cluster_collection.Branch("event",self.event,"event/I")
            #        }

        #### Not include Nclusters in the tree data just before clusterization (has no sense)
        if not tree_name in ['pixelizedEvent','info','process_config','info_tagged']:
            setattr(self,"{}_Nclusters".format(tree_name), np.array([0],int))
            ncls = getattr(self,"{}_Nclusters".format(tree_name))
            branches_dict.update({'Nclusters':tree_cluster_collection.Branch("Nclusters",ncls,"Nclusters/I")})
       
        ### Add vectorilized branches
        setattr(self,"{}_vect_branches".format(tree_name),{})
        vect_branches=getattr(self,"{}_vect_branches".format(tree_name))
        setattr(self,"{}_vec_vec_br_type".format(tree_name),{})
        vec_vec_br_type=getattr(self,"{}_vec_vec_br_type".format(tree_name))
        ### Add integer branches
        setattr(self,"{}_int_branches".format(tree_name),{})
        int_branches=getattr(self,"{}_int_branches".format(tree_name))
        
        for keyword in sorted(cluster_obj.__dict__.keys()):
            val = cluster_obj.__dict__[keyword]

            if keyword in ["event","is_simulation"]:
                continue

            dtype = type(val)
            if tree_name not in ["clustersRec","clustersRec_tagged"]:
                if dtype in [int,np.int8,np.int16,np.int32,np.int64]:
                    setattr(self,"{}_{}".format(tree_name,keyword), np.array([-1],int))
                    bval = getattr(self,"{}_{}".format(tree_name,keyword))
                    branches_dict.update({keyword:tree_cluster_collection.Branch(keyword,bval,"{}/I".format(keyword))})
                    int_branches[keyword] = dtype
                    int_br_keywords.append(keyword)
                    continue
                elif dtype in [float]:
                    setattr(self,"{}_{}".format(tree_name,keyword), np.array([-1],float))
                    bval = getattr(self,"{}_{}".format(tree_name,keyword))
                    branches_dict.update({keyword:tree_cluster_collection.Branch(keyword,bval,"{}/D".format(keyword))})
                    int_branches[keyword] = dtype
                    int_br_keywords.append(keyword)
                    continue
                elif dtype in [ROOT.string,ROOT.std.string]:
                    setattr(self,"{}_{}".format(tree_name,keyword),ROOT.std.string(""))
                    bval = getattr(self,"{}_{}".format(tree_name,keyword))
                    branches_dict.update({keyword:tree_cluster_collection.Branch(keyword,bval)})
                    int_branches[keyword] = dtype
                    int_br_keywords.append(keyword)
                    continue 

            if dtype is not dict:
                if dtype is not np.ndarray:
                    vect_branches[keyword] = ROOT.std.vector(__python2C_type__[dtype])()
                    vec_br_keywords.append(keyword)
                else:
                    dtype_inner = type(val[0])
                    vect_branches[keyword] = ROOT.std.vector(ROOT.std.vector(__python2C_type__[dtype_inner]))()
                    vec_vec_br_type[keyword] = ROOT.std.vector(__python2C_type__[dtype_inner])
                    vec_vec_br_keywords.append(keyword)

        branches_dict.update(dict(map(lambda par:(par[0],tree_cluster_collection.Branch(par[0],par[1])),vect_branches.items())))
        

    def fill_tree(self,tree_name,event,cluster_collection,isdata=False):
        """Fill branches on the tree_name given by a list of objects having as attributes, all
            the required branches
        
        Parameters
        ----------
            tree_name   : str
                name of the ROOT TTree object
            event   : int
                unique id number for the simulated event
            cluster_collection : class
                object class where a branch for each attribute will be created
                Only numeric attributes, string are not implemented
                Classes that will be recorded using this functions are Cluster and PixelizedEvent

        """
        
        root_tree_name = "{}_tree_cluster_collection".format(tree_name)
        if not hasattr(self,root_tree_name):
            try:
                self.create_cluster_collection_tree(tree_name,cluster_collection[0],isdata=isdata)
            except IndexError:
                print("WARNING: IndexError: If 0 pixels has for clusterization Ignore, otherwise something went wrong!")
                return

        self.tfile.cd()
        
        if tree_name in ["clustersRec","pixelizedEvent","info","info_tagged","clustersRec_tagged"]:
            if isdata:
                getattr(self,"{}_RUNID".format(tree_name))[0] = event
            else:
                getattr(self,"{}_event".format(tree_name))[0] = event
        else:
            getattr(self,"{}_id".format(tree_name))[0] = event
       
        tree_cluster_collection = getattr(self,root_tree_name)
        vect_branches = getattr(self,"{}_vect_branches".format(tree_name))
        vec_br_keywords = getattr(self,"{}_vec_br_keywords".format(tree_name))
        vec_vec_br_keywords = getattr(self,"{}_vec_vec_br_keywords".format(tree_name))
        vec_vec_br_type = getattr(self,"{}_vec_vec_br_type".format(tree_name))
        # int branches
        int_branches  = getattr(self,"{}_int_branches".format(tree_name))
        int_br_keywords = getattr(self,"{}_int_br_keywords".format(tree_name))

        
        # Clean last event entries 
        for vec in vect_branches.values():
           # XXX how to clean vec((vec))?? XXX
           vec.clear()
           vec.reserve(len(cluster_collection))
      
        # Fill all cluster related branches
        n_clusters=0
        for i,cluster in enumerate(cluster_collection):
            n_clusters+=1
            ### for vec(type) params
            for keyword in vec_br_keywords:
                vect_branches[keyword].push_back(getattr(cluster,keyword))

            ### vec(vec(type)) values
            for keyword in vec_vec_br_keywords:
                array_val = getattr(cluster,keyword)
                 
                vect_branches[keyword].push_back(vec_vec_br_type[keyword]())
                vect_branches[keyword][-1].reserve(array_val.size)

                #### C++ do not understand int64, convert it into int
                p2r_type = lambda x: x
                if array_val.dtype in ['int64','int']:
                    p2r_type = lambda x: int(x)
                for val in array_val:
                    vect_branches[keyword][-1].push_back(p2r_type(val))

            ### int or string values
            for keyword in int_br_keywords:
                try:
                    getattr(self,"{}_{}".format(tree_name,keyword))[0] = getattr(cluster,keyword)
                except TypeError:
                    mystr = getattr(self,"{}_{}".format(tree_name,keyword))
                    mystr.replace(0,ROOT.std.string.npos,getattr(cluster,keyword))

        if not tree_name in ['pixelizedEvent','info','process_config', 'info_tagged']:
            getattr(self,"{}_Nclusters".format(tree_name))[0] = n_clusters
        
        tree_cluster_collection.Fill()
                        
    def close(self):
        self.tfile.Write()
        self.tfile.Close()

###########################################################################################################
#
#   CLASS FOR THE PIXELIZATION EVENT ROOT TTree 
#       Any attribute will be a branch on the Pixelized ROOT TTree
#
###########################################################################################################
class PixelizedEvent(object):
    """Pixelized Event Class

    A `PixelizedEvent` object is created for each event just before ClusterFinder process. All
    attributes of this object will be recorded in the *output root file* as a ROOT TTree object 
    called as **pixelizedEvent**.

    From the input hits (:obj:`G4HitCollection` ) object the following attributes (added to hits during
    its recronstruction) will be taking into account:
        
        * **Information related to the Primary Particle**. All parameters added with the 
        :method:`G4HitCollection.AddCoordSimulatedEvent()`) will be included (by default:
        coordenates, momentum and energy). This is information from the **EventOut** TTree (geant4
        simulation).

            * parameters starting with **pp_**
        
        * **Hit information** that corresponds to information from **CCDOut** after a pre-processing 
        (only Diffusion, if active, and PixelizeSignal). The following attributes will be included
        as branches on the output ROOT TTree object **pixelizedEvent**:
            
            1. pixel_x, pixel_y : pixel positon in the XY plane

            2. pixel_z    : depth in mm (z-axis)

            3. pixel_time : total energy loss per pixel in units of eV

            4. pixel_pdg  : the most frequent PDG value per pixel

            5. pixel_Edep : energy-weighted time average per pixel in units of seconds

            6. pixel_Neh  : number of carried charge created (i.e. electron-hole pairs) in each pixel

            7. pixel_sigma_xy : used sigma_xy to diffuse transversaly the e-h pairs position
    
    Example
    -------
    >>> tree.pixelizedEvent.Show(0)
    ======> EVENT:0
    event           = 29
    ccd             = (vector<int>*)0x55edf9279650
    Npix            = (vector<int>*)0x55edf927af60
    pp_posx         = (vector<vector<float> >*)0x55edf91ad4a0
    pp_posy         = (vector<vector<float> >*)0x55edf919bc20
    pp_posz         = (vector<vector<float> >*)0x55edf9199dc0
    pp_energy       = (vector<vector<float> >*)0x55edf918db30
    pp_momy         = (vector<vector<float> >*)0x55edf926bfc0
    pp_momz         = (vector<vector<float> >*)0x55edf7f58440
    pp_momx         = (vector<vector<float> >*)0x55edf927aa80
    pixels_x        = (vector<vector<int> >*)0x55edf9196dd0
    pixels_y        = (vector<vector<int> >*)0x55edf9194210
    pixels_z        = (vector<vector<float> >*)0x55edf91a3790
    pixels_Edep     = (vector<vector<float> >*)0x55edf92846a0
    pixels_pdg      = (vector<vector<float> >*)0x55edf927e230
    pixels_time     = (vector<vector<float> >*)0x55edf9277ab0
    >>> 
    >>> #Scan some variables
    >>> d.pixelizedEvent.Scan("event:ccd:pp_posx:pp_energy:pixels_x:pixels_Edep")
    ***********************************************************************************************
    *    Row   * Instance *     event *       ccd *   pp_posx * pp_energy *  pixels_x * pixels_Ed *
    ***********************************************************************************************
    *        0 *        0 *        29 *         1 * -21.22118 *         0 *      1608 * 2.6661031 *
    *        0 *        1 *        29 *           *           *           *      1609 * 2.4304614 *
    *        0 *        2 *        29 *           *           *           *      1609 * 2.1237189 *
    *        0 *        3 *        29 *           *           *           *      1610 * 1.7100080 *
    *        0 *        4 *        29 *           *           *           *      1610 * 4.7656173 *
    *        0 *        5 *        29 *           *           *           *      1610 * 0.1135936 *
    *        0 *        6 *        29 *           *           *           *      1611 * 4.2469434 *
    *        0 *        7 *        29 *           *           *           *      1611 * 2.8502533 *
    *        0 *        8 *        29 *           *           *           *      1612 * 2.2147212 *
    *        0 *        9 *        29 *           *           *           *      1612 * 8.2684907 *
    *        0 *       10 *        29 *           *           *           *      1612 * 1.6448229 *
    *        0 *       11 *        29 *           *           *           *      1612 * 5.7333755 *
    *        0 *       12 *        29 *           *           *           *      1612 * 2.9817473 *
    *        0 *       13 *        29 *           *           *           *      1613 * 4.0074143 *
    *        0 *       14 *        29 *           *           *           *      1613 * 2.7946200 *
    *        0 *       15 *        29 *           *           *           *      1613 * 3.0512294 *
    *        0 *       16 *        29 *           *           *           *      1613 * 3.5478715 *
    *        0 *       17 *        29 *           *           *           *      1613 * 7.6080102 *
    *        0 *       18 *        29 *           *           *           *      1613 * 7.3263616 *
    *        0 *       19 *        29 *           *           *           *      1613 * 0.1248200 *
    *        0 *       20 *        29 *           *           *           *      1614 * 2.8508651 *
    *        0 *       21 *        29 *           *           *           *      1614 * 4.9168648 *
    *        0 *       22 *        29 *           *           *           *      1614 * 3.4959092 *
    *        0 *       23 *        29 *           *           *           *      1614 * 0.1327403 *
    *        0 *       24 *        29 *           *           *           *      1615 * 0.0641222 *
    Type <CR> to continue or q to quit ==> q
    ***********************************************************************************************
 
    The information is recorded once per event, and within this once for each pair (event,ccd).
    So for each entry on the ROOT TTree **pixelizedEvent** there are three types of parameters:
        
        1. scalars: **event** and **Npix**, for the event ID number and the total number of pixels,
        respectively.
        
        2. *(vector<type>*)*: for those information that do not change from tuple to tuple

        3. *(vector<vector<type>>*)*: for those parameters that change from tuple to tuple
    
    
    Parameters
    ----------
        hits : :obj:`G4HitCollection`
            collection of hits from the geant4 simulations
    
    """    
    def __init__(self,hits):
        self.event=int(hits.event)
        self.ccd=int(hits.ccd)

        #### PRIMARY PARTICLE INFORMATION: position, momentum and energy
        for attr in filter(lambda a:a.count("pp_")>0,hits.__dict__.keys()):
            setattr(self,attr,getattr(hits,attr).astype(float))

        #### PIXELIZED INFORMATION: position, energy, pdg, time, Neh, sigma
        for attr in ['x_pixel','y_pixel','z_pixel','pdg_pixel','time_pixel','Edep_pixel',
                'N_carried_charges_pixel','sigma_xy_pixel','neutron_pixel','gamma_pixel',
                'silicon_pixel','alpha_pixel','ion_pixel','electron_pixel']:
            try:
                setattr(self,"pixels_{}".format(attr.replace("_pixel","")),getattr(hits,attr))
            except AttributeError:
                ### not all attributes exists (the last two, only exists if diffusion is active)
                pass
        #for attr in filter(lambda a:a.count("_pixel")>0,hits.__dict__.keys()):

        #### output units of the energy keV (the rest are already in the desired output units)
        self.pixels_Edep = self.pixels_Edep/u.keV

        self.Npix = int(self.pixels_x.size)

        #### Adding "displacement" (if exists): when PastClustersProcess is invoked
        for attr in ['rnd_rows','rnd_cols','rnd_Edep']:
            try:
                setattr(self,attr,np.array(getattr(hits,attr)))
            except AttributeError:
                ### not all attributes exists (the last two, only exists if diffusion is active)
                pass

###########################################################################################################
#
#   CLASS FOR THE CLUSTER RECONSTRUCTION ROOT TTree 
#       Any attribute will be a branch on the Cluster ROOT TTree
#
###########################################################################################################
class Cluster(object):
    """Cluster Class

    A `Cluster` object is created for processed hit collection after all process have been applied. This
    object will be instanciated only if :obj:`ClusterFinder` process is called (i.e. when set to active 
    at the configuration json file). All attributes of a Cluster instance will be recorded in the 
    *output root file* as a branch in the ROOT TTree object named as **clustersRec**.

    From the input hits (:obj:`G4HitCollection` ) object the following attributes will be taking into 
    account:
        
        * **Information related to the Primary Particle**. All parameters added with the 
        :method:`G4HitCollection.AddCoordSimulatedEvent()`) will be included (by default:
        coordenates, momentum and energy). This is information from the **EventOut** TTree (geant4
        simulation).
        
        * **Hit information** that corresponds to information from **CCDOut** after all process
        chain. See **How to run psimulCCDimg** for more detailed information on the output branches.
            
    
    Example
    -------
     >>> tree.clustersRec.Show(0)
     ======> EVENT:0
      event           = 11
      Nclusters       = 5
      DX              = (vector<float>*)0x39aa230
      DY              = (vector<float>*)0x4b2ac20
      DZ              = (vector<float>*)0x486f970
      Energy          = (vector<float>*)0x4878df0
      Npix            = (vector<int>*)0x4b1b160
      PosX            = (vector<float>*)0x4b29a80
      PosY            = (vector<float>*)0x4b2eff0
      PosZ            = (vector<float>*)0x4b34530
      Qmax            = (vector<float>*)0x4885730
      QmaxX           = (vector<float>*)0x487f750
      QmaxY           = (vector<float>*)0x4b349b0
      QmaxZ           = (vector<float>*)0x4b21940
      RMSX            = (vector<float>*)0x4b2a990
      RMSY            = (vector<float>*)0x2b12300
      RMSZ            = (vector<float>*)0x4b20490
      ccd             = (vector<int>*)0x487cf80
      cluster_id      = (vector<int>*)0x48905c0
      maxX            = (vector<float>*)0x4b1ce20
      maxY            = (vector<float>*)0x48702a0
      maxZ            = (vector<float>*)0x488e390
      meanTime        = (vector<float>*)0x487f150
      meanX           = (vector<float>*)0x4b1f9e0
      meanY           = (vector<float>*)0x487e830
      meanZ           = (vector<float>*)0x4b2c580
      minTime         = (vector<float>*)0x4b2d460
      minX            = (vector<float>*)0x486d780
      minY            = (vector<float>*)0x4878270
      minZ            = (vector<float>*)0x4877fd0
      pixels_E        = (vector<vector<float> >*)0x4da7c80
      pixels_time     = (vector<vector<float> >*)0x4dcb2d0
      pixels_x        = (vector<vector<float> >*)0x4dc7700
      pixels_y        = (vector<vector<float> >*)0x4cc0db0
      pixels_z        = (vector<vector<float> >*)0x4e26780
      pp_energy       = (vector<vector<float> >*)0x4c7a7e0
      pp_momx         = (vector<vector<float> >*)0x4de3430
      pp_momy         = (vector<vector<float> >*)0x4de3f20
      pp_momz         = (vector<vector<float> >*)0x4d06070
      pp_posx         = (vector<vector<float> >*)0x4c91f50
      pp_posy         = (vector<vector<float> >*)0x4e61530
      pp_posz         = (vector<vector<float> >*)0x4c7a4b0
      primary_part    = (vector<vector<float> >*)0x4e650e0
      timeQmax        = (vector<float>*)0x4e37880
      wTime           = (vector<float>*)0x4ecbd40
     >>> tree.clustersRec.Scan("event:ccd:Nclusters:DX:PosX:Qmax:Energy:pixels_x","event == 11 && ccd == 2")
      ***********************************************************************************************************************
      *    Row   * Instance *     event *       ccd * Nclusters *        DX *      PosX *      Qmax *    Energy *  pixels_x *
      ***********************************************************************************************************************
      *        0 *      390 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3254 *
      *        0 *      391 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3255 *
      *        0 *      392 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3255 *
      *        0 *      393 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3255 *
      *        0 *      394 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3255 *
      *        0 *      395 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3255 *
      *        0 *      396 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3255 *
      *        0 *      397 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      398 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      399 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      400 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      401 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      402 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      403 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3256 *
      *        0 *      404 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      405 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      406 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      407 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      408 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      409 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      410 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3257 *
      *        0 *      411 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3258 *
      *        0 *      412 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3258 *
      *        0 *      413 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3258 *
      *        0 *      414 *        11 *         2 *         5 *        38 * 3274.5185 * 19.981000 *   485.625 *      3258 *
      Type <CR> to continue or q to quit ==>q
      ***********************************************************************************************************************
      ==> 25 selected entries
      25
      
    There are an entry for each event, and within this once for each pair of values (event,ccd). 
    So for each entry on the ROOT TTree **pixelizedEvent** there are three types of parameters:
        
        1. scalars: **event** and **Nclusters**, for the event ID number and the total number of pixels,
        respectively.
        
        2. *(vector<type>*)*: attribute information related to characterize the cluster (for
        instance, the total energy of the cluster, or the mean position on the x-aixs).

        3. *(vector<vector<type>>*)*: attribute information to track the pixel information (energy
        loss per pixel for all pixels belonging to the cluster)
    
    
    Parameters
    ----------
        hits : :obj:`G4HitCollection`
            collection of hits from the geant4 simulations


    """

    __valid_E_units__ = ['e','eV','keV','ADC','ADCu']

    
    def __init__(self,cls_id,hits,mask,E_units='eV'):

        if not E_units in self.__valid_E_units__:
            raise TypeError("Energy units `{}` is not valid".format(E_units))

        self.event=int(hits.event)
        if hasattr(hits,'n_image'):
            self.nfile = int(hits.n_image)
        else:
            self.ccd = int(hits.ccd)

        if hasattr(hits,'ccd'):
            self.ccd = int(hits.ccd)

        #### PRIMARY PARTICLE INFORMATION: position, momentum and energy
        for attr in filter(lambda a:a.count("pp_")>0,hits.__dict__.keys()):
            setattr(self,attr,getattr(hits,attr).astype(float))

        self.cluster_id = int(cls_id)
        
        ### starting the pixel indexing at 1, instead of 0
        self.pixels_x = (getattr(hits,"x{}".format(hits.__attr_to_clusterize__))[mask]).astype(int)+1
        self.pixels_y = (getattr(hits,"y{}".format(hits.__attr_to_clusterize__))[mask]).astype(int)+1
        self.pixels_z = (getattr(hits,"z{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
        # std_xy
        self.STD_XY   = (((self.pixels_x-self.pixels_x.mean())**2 +(self.pixels_y-self.pixels_y.mean())**2).sum()/len(self.pixels_x))**0.5

        self.pixels_E = (getattr(hits,"Edep{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)/u.keV
        self.Energy   = self.pixels_E.sum()

        #### CLUSTER SIZE
        self.Npix = int(len(self.pixels_x))

        ###### ADD MONTE CARLO TRUTH IF EXISTS
        #######################################################################################
        if hasattr(hits,"Edep_pixels_MCT"):
            self.pixels_E_MCT = (hits.Edep_pixels_MCT[mask]).astype(float)/u.keV
            self.Energy_MCT = float(self.pixels_E_MCT.sum())
            self.Npix_MCT = int(sum(self.pixels_E_MCT > 0))

        if hasattr(hits,"Edep_MCT"):
            self.Edep_MCT = (hits.Edep_MCT).astype(float)/u.keV
            self.Energy_MCT = float(self.Edep_MCT.sum())
            self.Npix_MCT = int(sum(self.Edep_MCT > 0))
            self.posx_MCT = hits.posx_MCT.astype(float)
            self.posy_MCT = hits.posy_MCT.astype(float)
            self.posz_MCT = hits.posz_MCT.astype(float)

        if hasattr(hits,"Edep_pixels_MCT_ID"):
            ### id for each pixel of the cluster (clusters maybe are splitted, or joined, ...)
            self.pixels_MCT_cls_id = (hits.Edep_pixels_MCT_ID[mask]).astype(int)
            pileup = len(set(self.pixels_MCT_cls_id))
            self.MCT_pileup = int(pileup > 1)
            self.MCT_pileup_Ncls = pileup

        if hasattr(hits,"Edep_map_sat"):
            self.pixels_saturated = (hits.Edep_map_sat[mask]).astype(int)
            self.Npix_sat = float(self.pixels_saturated.sum())

        #### INFORMATION ONLY FOR SIMULATIONS
        if not hits.isdata:
            self.primary_part = (getattr(hits,"pdg{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            self.pixels_time = (getattr(hits,"time{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            ### Add also the flags for nuetron and gamma interactions
            self.neutron_in_pixel = (getattr(hits,"neutron{}".format(hits.__attr_to_clusterize__))[mask]).astype(float) 
            self.gamma_in_pixel = (getattr(hits,"gamma{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            self.isgamma = int(max(self.gamma_in_pixel))
            self.isneutron = int(max(self.neutron_in_pixel))

            ### Add also the flags for silicon, alpha, ion interactions
            self.silicon_in_pixel = (getattr(hits,"silicon{}".format(hits.__attr_to_clusterize__))[mask]).astype(float) 
            self.alpha_in_pixel = (getattr(hits,"alpha{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            self.ion_in_pixel = (getattr(hits,"ion{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            self.electron_in_pixel = (getattr(hits,"electron{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            self.issilicon = int(max(self.silicon_in_pixel))
            self.isalpha = int(max(self.alpha_in_pixel))
            self.ision = int(max(self.ion_in_pixel))
            self.iselectron = int(max(self.electron_in_pixel))

            ### Add some statistical parameters related with time (mostly for simulations full decay  mode)
            self.pixels_std_time = (getattr(hits,"std_time{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)
            self.pixels_img_time = (getattr(hits,"time_img{}".format(hits.__attr_to_clusterize__))[mask]).astype(float)

        #### INFORMATION ONLY FOR DATA IMAGES
        if hits.isdata:
            self.get_info_from_image(hits,cls_id,mask)

            #### IF IS DATA, AND HAS DQM HEADER INFORMATION
            # --- XXX add only this information in the info tree
            #if hasattr(hits,"image_header"):
            #    self.get_info_from_DQM(hits)
            
            #### TAG CLUSTER IF ANY OF THE PIXELS BELONGS TO A HOT COLUMN AND/OR ROW
            if hasattr(hits,"in_hot_rows") or hasattr(hits,"in_hot_columns"):
                hot_rows = (getattr(hits,"in_hot_rows")[sorted(set(self.pixels_y-1))]).any()
                hot_cols = (getattr(hits,"in_hot_columns")[sorted(set(self.pixels_x-1))]).any()
                self.hotreg = int(any([hot_rows,hot_cols]))

                if hasattr(self,"has_seed"):
                    self.excluded = int(np.logical_or(bool(self.hotreg),not self.has_seed))
                else:
                    self.excluded = self.hotreg

            # DATA FROM MOSKITA SETUP HAS THE LUMINOSITY OF THE PP COLISION AT THE FITS FILE HEADER
            if 'PPCOL' in hits.image_header:
                self.ppcol = float(hits.image_header['PPCOL'])*1e34
            # DATA FROM MOSKITA SETUP HAS THE LUMINOSITY OF THE HI COLISION AT THE FITS FILE HEADER
            if 'HICOL' in hits.image_header:
                self.hicol = float(hits.image_header['HICOL'])*1e27

        #### Add timestamp
        if hasattr(hits,'start_readout'):
            self.readout_start = hits.start.timestamp()
        if hasattr(hits,'end'):
            self.readout_end = hits.end.timestamp()

    def get_info_from_DQM(self,hits):
        for amp in hits.amplifier.keys():
            for fkey in ['MEGAIN','MEDC','MESIGMA','MEMU0']:
                fkey = fkey+amp.upper()
                skey = '{}_{}'.format(amp.upper(),fkey[:-1].replace('ME',''))
                try:
                    setattr(self, str.lower(skey),        float(hits.image_header[fkey]))
                    setattr(self, str.lower(skey+"_err"), float(hits.image_header['E'+fkey]))
                except KeyError:
                    setattr(self, str.lower(skey),        float(0.0))
                    setattr(self, str.lower(skey+"_err"), float(0.0))
                    continue
        
        # should be optimized for the LBC data ... do not include them yet
        # for fkey in ['MEBS','MEBSV','MEBSN','MEHC','MEHCSIZE','MESC','MESCV','MESCT']:
        #    try:
        #        setattr(self,str.lower(fkey), float(hits.image_header[fkey]))
        #    except KeyError:
        #        setattr(self,str.lower(fkey), float(0.0))
        #        continue
        return


    def get_info_from_image(self,hits,cls_id,mask):

        ###### REAL DATA RELATED PARAMETERS
        #######################################################################################
        ### sigma used to discriminate signal from background pixels
        self.sigma_seed_eV = hits.sigma0
        ### in some cases there is an error for sigma0 (see rawdata)
        if hasattr(hits,"sigma0_std"):
            self.sigma_seed_eV_std = hits.sigma0_std

        ### calculate minimum and maximum distance to any masked pixel
        if hasattr(hits,"mask"):
            points  = np.where(hits.mask == 1)
            mpoints = np.array(list(zip(points[0],points[1])))
            cls_points = np.array(list(zip(self.pixels_y,self.pixels_x)))
            distance_to_mask = cdist(mpoints,cls_points)
            setattr(self,"dist_mask_min",distance_to_mask.min())
            setattr(self,"dist_mask_max",distance_to_mask.max())
            
        ### add attribute "hass_seed" to detect those cluster that has at least one pixel with
        #       the minimum required charge
        if hasattr(hits,"q_min_cls_seed"):
            # q_min_cls_seed in units of eV (see rawdata)
            self.qmin_cluster_seed = hits.q_min_cls_seed/u.keV
            self.has_seed = int(any(self.pixels_E > self.qmin_cluster_seed))
            # minimum number of sigmas for each cluster
            sigma_seed_keV = hits.sigma0/u.keV
            ### extra parameters to define the level of noise 
            self.seed_nsig_pixels = self.pixels_E/sigma_seed_keV
            self.seed_nsig_min = min(self.seed_nsig_pixels)
            self.seed_nsig_max = max(self.seed_nsig_pixels)
            self.seed_nsig_delta = self.seed_nsig_max-self.seed_nsig_min
            
        #### only for data: get the energy from the previous images, to have the energy of the
        #       cluster at any intermediate stage
        adu_to_e = 1/u.ADC2e
        if hasattr(u,'calibration'):
            # from CalibrationProcess if this was used
            adu_to_e = 1/u.calibration
        # from ADU to keV
        adu_to_keV = adu_to_e * u.e2eV / u.keV
        
        attrnames = ['AVG','PS','ECTS','ROT']
        for i,iname in  enumerate(["image_mean_compressed","image_mean_compressed_pedestal_subtracted",
                "image_mean_compressed_pedestal_subtracted_correct_cols","image_mean_compressed_pedestal_subtracted_correlated"]):
            if hasattr(hits,iname):
                image = getattr(hits,iname)
                pixel_charge = image[self.pixels_y-1,self.pixels_x-1]
                setattr(self,'pixels_E_{}'.format(attrnames[i]), pixel_charge)
                setattr(self,'Energy_{}'.format(attrnames[i]), pixel_charge.sum())
                setattr(self,'Energy_{}'.format(attrnames[i]), pixel_charge.sum())
        return


    def get_cluster_properties(self,is_simulation=False,__DEBUG__=False,get_fitted_STD=False):
        """Run :method:`Cluster.get_properties_on_axes` for all axis (x,y and z), as well as,
        compute some statistics for time properties and charge (energy loss).
        """
        
        self.is_simulation = is_simulation

        self.get_properties_on_axes('x')
        self.get_properties_on_axes('y')
        setattr(self,"wSTD_XY",np.sqrt((getattr(self,"wSTD_X")**2 + getattr(self,"wSTD_Y")**2)/2.))
        self.compute_cluster_pca()

        if get_fitted_STD:
            self.get_fitted_STD(__DEBUG__)

        if is_simulation:
            self.get_properties_on_axes('z')
            self.get_properties_on_time()
        self.get_charge_properties()
        
        self.get_tracking_endpoints(__DEBUG__)


    def get_tracking_endpoints(self,__DEBUG__):
        """
        Find the two endpoints of the reconstructed track.
    
        The distance between pixels is computed along the cluster using
        8-neighbour connectivity. Therefore, paths are restricted to pixels
        belonging to the cluster.
    
        The algorithm is:
    
        1. Use the maximum-charge pixel as the seed.
        2. Find the cluster pixel with the largest graph distance from the seed.
           This pixel is the first endpoint.
        3. Starting from the first endpoint, find the cluster pixel with the
           largest graph distance. This pixel is the second endpoint.
    
        Returns
        -------
        endpoint1 : tuple[int, int]
            Coordinates ``(x, y)`` of the first track endpoint.
    
        endpoint2 : tuple[int, int]
            Coordinates ``(x, y)`` of the second track endpoint.
    
        Raises
        ------
        ValueError
            If the cluster contains no pixels.
        """
    
        pixels_x = np.asarray(self.pixels_x, dtype=int)
        pixels_y = np.asarray(self.pixels_y, dtype=int)
    
        if len(pixels_x) != len(pixels_y):
            raise ValueError(
                "pixels_x and pixels_y must contain the same number of pixels."
            )
    
        if len(pixels_x) == 0:
            raise ValueError("Cannot find endpoints of an empty cluster.")
    
        # Remove duplicated pixel coordinates, if any.
        pixels = list(dict.fromkeys(zip(pixels_x, pixels_y)))
        pixel_set = set(pixels)
    
        if len(pixel_set) == 1:
            endpoint = next(iter(pixel_set))
    
            self.track_endpoint1_X = float(endpoint[0])
            self.track_endpoint1_Y = float(endpoint[1])
            self.track_endpoint2_X = float(endpoint[0])
            self.track_endpoint2_Y = float(endpoint[1])
    
            return
    
        # QmaxX/QmaxY may be stored as floating-point values.
        seed_position = (float(self.QmaxX), float(self.QmaxY))
    
        # Make sure that the seed is an actual pixel belonging to the cluster.
        seed = min(
            pixel_set,
            key=lambda pixel: (
                (pixel[0] - seed_position[0]) ** 2
                + (pixel[1] - seed_position[1]) ** 2
            ),
        )
    
        def graph_distances(start):
            """
            Compute shortest-path distances from one pixel to all connected
            cluster pixels using 8-neighbour connectivity.
            """
            neighbours = (
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            )
    
            distances = {start: 0}
            queue = deque([start])
    
            while queue:
                x, y = queue.popleft()
    
                for dx, dy in neighbours:
                    neighbour = (x + dx, y + dy)
    
                    if neighbour not in pixel_set:
                        continue
    
                    if neighbour in distances:
                        continue
    
                    distances[neighbour] = distances[(x, y)] + 1
                    queue.append(neighbour)
    
            return distances
    
        def farthest_pixel(distances, reference):
            """
            Select the pixel with the largest graph distance.
    
            The Euclidean distance is used only as a deterministic tie-breaker.
            """
            ref_x, ref_y = reference
    
            return max(
                distances,
                key=lambda pixel: (
                    distances[pixel],
                    (pixel[0] - ref_x) ** 2
                    + (pixel[1] - ref_y) ** 2,
                ),
            )
    
        # First endpoint: farthest connected pixel from the maximum-charge seed.
        distances_from_seed = graph_distances(seed)
        endpoint1 = farthest_pixel(distances_from_seed, seed)
    
        # Second endpoint: farthest connected pixel from the first endpoint.
        distances_from_endpoint1 = graph_distances(endpoint1)
        endpoint2 = farthest_pixel(distances_from_endpoint1, endpoint1)
    
        # A reconstructed cluster should normally be one connected component.
        if len(distances_from_seed) != len(pixel_set):
            n_disconnected = len(pixel_set) - len(distances_from_seed)
    
            if __DEBUG__:
                print(
                    "Warning: the cluster contains "
                    f"{n_disconnected} pixel(s) disconnected from the Qmax component."
                )

        self.track_endpoint1_X = float(endpoint1[0])
        self.track_endpoint1_Y = float(endpoint1[1])
        self.track_endpoint2_X = float(endpoint2[0])
        self.track_endpoint2_Y = float(endpoint2[1]) 

        return 


    def get_fitted_STD(self,__DEBUG__):
        _isbatch = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(1)

        ### get the STD by fitting a gaussian to the cluster profile
        px = ROOT.TH1D("px","px", int(self.pixels_x.max()-self.pixels_x.min())+3, self.pixels_x.min()-1.5, self.pixels_x.max()+1.5)
        py = ROOT.TH1D("py","py", int(self.pixels_y.max()-self.pixels_y.min())+3, self.pixels_y.min()-1.5, self.pixels_y.max()+1.5)
                 
        for x,y,e in zip(self.pixels_x,self.pixels_y,self.pixels_E):
            _ = px.Fill(x,e/self.pixels_E.max())
            _ = py.Fill(y,e/self.pixels_E.max())
         
        setattr(self,"projectionX", np.zeros(px.GetNbinsX()).astype(float))
        setattr(self,"projectionX_cols", np.linspace(self.pixels_x.min()-1.5, self.pixels_x.max()+1.5,int(self.pixels_x.max()-self.pixels_x.min())+3).astype(float))
        for i in range(px.GetNbinsX()):
            self.projectionX[i] = px.GetBinContent(i+1)
        setattr(self,"projectionY", np.zeros(py.GetNbinsX()).astype(float))
        setattr(self,"projectionY_rows", np.linspace(self.pixels_y.min()-1.5, self.pixels_y.max()+1.5,int(self.pixels_y.max()-self.pixels_y.min())+3).astype(float))
        for i in range(py.GetNbinsX()):
            self.projectionY[i] = py.GetBinContent(i+1)

        fit_opt = "Q L"
        # fitting x projection to gauss
        gausx = ROOT.TF1("gaus_x","gaus")
        gausx.SetLineColor(2)
        gausx.SetParameter(1,self.PosX)
        gausx.SetParLimits(1,self.minX,self.maxX)
        px.SetTitle(";pixels_x;counts")
        px.Fit(gausx,fit_opt)
        setattr(self,"fSTD_X", float(gausx.GetParameter(2)))
        setattr(self,"fPosX",  float(gausx.GetParameter(1)))

        # fitting y projection to gauss
        gausy = ROOT.TF1("gaus_y","gaus")
        gausy.SetLineColor(2)
        gausy.SetParameter(1,self.PosY)
        gausy.SetParLimits(1,self.minY,self.maxY)
        py.SetTitle(";pixels_y;counts")
        py.Fit(gausy,fit_opt)
        setattr(self,"fSTD_Y",float(gausy.GetParameter(2)))
        setattr(self,"fPosY", float(gausy.GetParameter(1)))
        
        if __DEBUG__:
            ROOT.gROOT.SetBatch(0)
            c = ROOT.TCanvas("Projections_on_X","Projections on X")
            c.cd()
            px.Draw()
            px.Draw("HIST same")
            c.Update()
            c.Draw()

            c2 = ROOT.TCanvas("Projections_on_Y","Projections on Y")
            c2.cd()
            py.Draw()
            py.Draw("HIST same")
            c2.Update()
            c2.Draw()
            input("...")
        ROOT.gROOT.SetBatch(_isbatch)


    def get_properties_on_time(self):
        """Add several time attributes (in second units). One value per cluster.
            
            **wTime**: energy-weighted pixel time average
            
            **meanTime**: mean pixel time average
            
            **minTime**: minimum time among all pixel time values

        """
        
        values = self.pixels_time

        #### averaged weighted time
        setattr(self,"wTime",float(np.average(values,weights=self.pixels_E)))
        #### mean time
        setattr(self,'meanTime',float(np.mean(values)))
        #### minimum time
        setattr(self,'minTime',float(min(values)))
        #### add time images statistics
        setattr(self,'img_time',float(np.mean(self.pixels_img_time)))


    def get_properties_on_axes(self,axis,stats_list=['mean','min','max']):
        """Add several attributes related with the coordenates of the cluster.

            **RMS**: root-mean square of the pixel coordenates belonging to the cluster (one for each axis)

            **DX**: cluster elongation on the x-axis :math:`(max(x_{ij}) - min(x_{ij})` (same for DY and DZ)

            **PosX**: weighted-energy pixel positon on the x-axis (the same for y and z-axis, PosY and PosZ, respectively)

        """
        values = getattr(self,'pixels_{}'.format(axis))

        #### statistics from list
        for stat_name in stats_list:
            statistic_isnt = getattr(np,stat_name)
            setattr(self,stat_name+axis.upper(),float(statistic_isnt(values)))
        
        #### root mean square
        setattr(self,"RMS"+axis.upper(),float(np.mean(values**2.0)))

        #### cluster elongation
        setattr(self,"D"+axis.upper(),float(np.max(values)-np.min(values)+1.0))

        #### averaged weighted position
        setattr(self,"Pos"+axis.upper(),float(np.average(values,weights=self.pixels_E)))
        
        #### STD
        setattr(self,"STD_"+axis.upper(),float(values.std()))

        ##### weighted standard deviation  ------------------------------- AFTER VERSION 5.9
        wvalues = getattr(self,"Pos"+axis.upper())
        setattr(self,"wSTD_"+axis.upper(), np.sqrt((self.pixels_E * (values - wvalues)**2).sum()/self.pixels_E.sum()))

    def compute_cluster_pca(self, eps=1e-12):
        x = self.pixels_x
        y = self.pixels_y
        w = self.pixels_E
        sum_w = np.sum(w)

        wx = self.PosX
        wy = self.PosY
        dx = x - wx
        dy = y - wy
        
        var_x = self.wSTD_X**2
        var_y = self.wSTD_Y**2
        # Weighted covariance between x and y
        wSTD_XY = np.sum(w * dx * dy) / sum_w

        # Covariance matrix
        C = np.array([[var_x, wSTD_XY],[wSTD_XY, var_y]])
        # Diagonalize the covariance matrix
        # eigh is pappropiate because C is symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # np.linalg.eigh returns eigenvalues in ascending order.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        lambda1 = max(eigenvalues[0], 0.0)
        lambda2 = max(eigenvalues[1], 0.0)
        
        # Principal-axis RMS widths.
        sigma_long = np.sqrt(lambda1)
        sigma_trans = np.sqrt(lambda2)

        # Morphological variables.
        elongation = sigma_long / max(sigma_trans, eps)
        eccentricity = 1.0 - lambda2 / max(lambda1, eps)
        linearity = sigma_trans / max(sigma_long, eps)

        # Principal-axis direction.
        # eigenvectors[:, 0] is the eigenvector associated with lambda1.
        v_long = eigenvectors[:, 0]
        theta = np.arctan2(v_long[1], v_long[0])

        # Equivalent analytical expression for the angle:
        theta_analytic = 0.5 * np.arctan2(2.0 * wSTD_XY, var_x - var_y)

        # Project pixels onto the PCA axes.
        # u is the longitudinal coordinate, v is the transverse coordinate.
        u = dx * v_long[0] + dy * v_long[1]

        v_trans_axis = eigenvectors[:, 1]
        v = dx * v_trans_axis[0] + dy * v_trans_axis[1]

        # Robust geometrical length and width along PCA axes.
        # These are unweighted percentiles; useful for morphology.
        L_pca = np.percentile(u, 95) - np.percentile(u, 5)
        W_pca = np.percentile(v, 84) - np.percentile(v, 16)

        # Store PCA covariance matrix components.
        setattr(self, "pcaCovXX", var_x)
        setattr(self, "pcaCovXY", wSTD_XY)
        setattr(self, "pcaCovYY", var_y)
        
        # Store PCA eigenvalues.
        # lambda_long is the largest eigenvalue and corresponds to the main cluster axis.
        # lambda_trans is the smallest eigenvalue and corresponds to the transverse axis.
        setattr(self, "pcaLambdaLong", lambda1)
        setattr(self, "pcaLambdaTrans", lambda2)
        
        # Store RMS widths along the PCA axes.
        setattr(self, "pcaSigmaLong", sigma_long)
        setattr(self, "pcaSigmaTrans", sigma_trans)
        
        # Store morphology variables derived from the PCA eigenvalues.
        setattr(self, "pcaElongation", elongation)
        setattr(self, "pcaEccentricity", eccentricity)
        setattr(self, "pcaLinearity", linearity)
        
        # Store orientation of the main PCA axis.
        setattr(self, "pcaTheta", theta)
        setattr(self, "pcaThetaAnalytic", theta_analytic)
        
        # Store PCA axis direction vectors.
        setattr(self, "pcaAxisLongX", v_long[0])
        setattr(self, "pcaAxisLongY", v_long[1])
        
        setattr(self, "pcaAxisTransX", v_trans_axis[0])
        setattr(self, "pcaAxisTransY", v_trans_axis[1])
        
        # Store pixel coordinates in the PCA reference frame.
        # These are arrays with one entry per pixel in the cluster.
        setattr(self, "pcaU", u)
        setattr(self, "pcaV", v)
        
        # Store robust length/width estimators in the PCA frame.
        setattr(self, "pcaLength95", L_pca)
        setattr(self, "pcaWidth68", W_pca)
        setattr(self, "pcaWidthOverLength", W_pca / max(L_pca, eps))        


    def get_charge_properties(self):
        """Add some statistics related to the energy loss, to alternatively define the positon of
        the cluster. The position of the cluster is now defined as the pixel coordenates where the
        track lost his maximum amount of energy.

            **Qmax** : maximum 'pixelized' amount of energy lost 

            **QmaxX** : pixel in the x-axis where Qmax was deposited

            **QmaxY** : pixel in the y-axis where Qmax was deposited

            **timeQmax** : time when Qmax was deposited

        """
        ind = np.where(self.pixels_E == self.pixels_E.max())[0][0]

        setattr(self,'Qmax', float(self.pixels_E[ind]))
        setattr(self,'QmaxX',float(self.pixels_x[ind]))
        setattr(self,'QmaxY',float(self.pixels_y[ind]))
        
        if self.is_simulation:
            setattr(self,'QmaxZ',float(self.pixels_z[ind]))
            ### including time info
            setattr(self,'timeQmax',float(self.pixels_time[ind]))

    def get_properties_as_tuple(self):
        """

        Return
        ------

            A tuple with all attributes of the Cluster instance.
        """

        reg = []
        for attr in self.__properties_list__:
            reg.append( getattr(self,attr) )

        return tuple(reg)

###########################################################################################################
#
#
#   ROOT TTree for reconstruction processes (analysis)
#
#
###########################################################################################################
class InfoFromReconstruction(object):
    def __init__(self,hits):

        #self.RUNID=int(hits.event)
        if hasattr(hits,'n_image'):
            self.nfile=int(hits.n_image)
            self.idccd=int(hits.ccd)
        else:
            self.nfile=int(hits.ccd)
            self.idccd=int(hits.ccd)

        ### exposure time
        if hasattr(hits,"exposure_time") and hasattr(hits,"read_time"):
            self.total_exposure = hits.exposure_time + hits.read_time
       
        ### add mean(mu_axis) from pedestal subtraction
        if hasattr(hits,"pedestal_mu"):
            axis_name = {0:'row',1:'col'}
            for amp in hits.pedestal_mu.keys():
                pedestal_mu = hits.pedestal_mu[amp]
                pedestal_sigma = hits.pedestal_sigma[amp]
                for ax in pedestal_mu.keys():
                    setattr(self,"{}_pedestal_mu_{}_ADU".format(amp,axis_name[ax]), pedestal_mu[ax].flatten())
                    setattr(self,"{}_pedestal_sigma_{}_ADU".format(amp,axis_name[ax]),
                            pedestal_sigma[ax].flatten())
                    setattr(self,"{}_pedestal_mu_{}_mean_ADU".format(amp,axis_name[ax]),
                            float(pedestal_mu[ax].flatten().mean()))
                    setattr(self,"{}_pedestal_sigma_{}_mean_ADU".format(amp,axis_name[ax]),
                            float(pedestal_sigma[ax].flatten().mean()))

                    # add entry with time vector  to plot the pedestal information
                    n = len(pedestal_mu[ax].flatten())
                    dt = (hits.end.timestamp()-hits.start_readout.timestamp())/float(n)
                    tstamp_s = np.array([ dt*i for i in np.linspace(0,1,n) ])
                    setattr(self,f"readout_time_{axis_name[ax]}", tstamp_s/tstamp_s.max() )
      
        ### information about the electronic column transient
        if hasattr(hits,"ect_mu0"):
            self.ect_mu = hits.ect_mu0
            self.ect_amplitude = hits.ect_A0 
            self.ect_chi = hits.ect_chi2 

        ### add information that is found in the fits file header
        added_dc_fit_params = False
        if hasattr(hits,"image_header"):
            for amp in hits.amplifier.keys():
                for fkey in ['MEGAIN','MEDC','MESIG','MEMU0']:
                    fkey = fkey+amp.upper()
                    skey = '{}_{}'.format(amp.upper(),fkey[:-1].replace('ME',''))
                    try:
                        setattr(self, str.lower(skey),        float(hits.image_header[fkey]))
                        setattr(self, str.lower(skey+"_err"), float(hits.image_header['E'+fkey]))
                    except KeyError:
                        # if the FitDarkCurrentProcess is run at the same time than the others processes
                        # the fitted parameters of the DC are not found anymore on the fits file header,
                        # instead they are stored under the Unit instance, at the object class _tohdr
                        for pkey in ['MEGAIN','MEDC','MESIG','MEMU0']:
                            fkey = pkey+amp.upper()
                            if fkey in u._tohdr.keys():
                                val,txt = u._tohdr[fkey]
                                e_val,e_txt = u._tohdr['E'+fkey]

                                skey = '{}_{}'.format(amp.upper(),fkey[:-1].replace('ME',''))
                                setattr(self, str.lower(skey), float(val))
                                setattr(self, str.lower(skey+"_err"), float(e_val))
                            else:
                                setattr(self, str.lower(skey),        float(0.0))
                                setattr(self, str.lower(skey+"_err"), float(0.0))
                        continue

            # DATA FROM MOSKITA SETUP HAS THE LUMINOSITY OF THE PP COLISION AT THE FITS FILE HEADER
            if 'PPCOL' in hits.image_header:
                self.ppcol = float(hits.image_header['PPCOL'])*1e34
            # DATA FROM MOSKITA SETUP HAS THE LUMINOSITY OF THE HI COLISION AT THE FITS FILE HEADER
            if 'HICOL' in hits.image_header:
                self.hicol = float(hits.image_header['HICOL'])*1e27

        #### Add timestamp
        if hasattr(hits,'start_readout'):
            self.readout_start = hits.start.timestamp()
        if hasattr(hits,'end'):
            self.readout_end = hits.end.timestamp()

        return

###########################################################################################################
#
#   CLASS FOR GEANT4 SIMULATIONS
#
#
###########################################################################################################

class G4HitCollection(object):
    """Hit Collection Class
        
        Collection of hits from geant4 simulations. Each hit on the collection must have 
        the following attributes:
        
            * [mandatory] coordenates in units of mm. Coordenate on the z-axis represents the depth
            within the silicon bulk, the x-axis coordenate corresponds on the serial register
            dimension, and the y-axis the direction of the horitzontal register.
            
            * [mandatory] energy in units of eV
            
            * [optional] time in units of seconds
            
            * [optional] PDG (unique ID number to identify the nature of the particle)
            
    Parameters
    ----------
    event   :  (int,int)
        tuple of two integers, with the event and CCD ID numbers
    
    x   :  ndarray
        There are two possible **signatures**:
    
            1. Only x is informed. In this case, x is a pandas.DataFrame structure, and
                x,y,z, Edep, ... are given as attributs of the x object
    
            2. When a DataFrame is not used, all parameters must be passes as arrays of the
                same dimensions
    
    mask : bool
        boolean array to mask the input arrays (allowd in both signatures)

    """


    __attr_to_clusterize__ = ""

    def __init__(self,event,x,y=None,z=None,Edep=None,pdg=None,time=None,mask=None,z_offset=None):
        self.isdata = False

        self.event = event[0]        
        self.ccd = event[1]
        
        ### Add attribute killed
        ###     - energy deposition lower than 3.77 will not generate any e-h pair, 
        ###         if the hit collection has not a single pixel with e-h created Pixelization will
        ###         crash
        self.killed = False

        ### Define two signatures for the constructions
        if y is None:
            #### x is an entry of the CCDOut G4tree for a given Edep, and CCD out
            if mask is None:
                raise ValidationError("SignatureError: for this signature the 'mask' is mandatory")

            if z_offset is not None:
                # consider only energy depositions within the z range [z_offset,u.ccd_thickness+z_offset]
                indices = np.where(mask)[0]
                if bool(u.invert_ccd_z_values):
                    z = u.ccd_thickness/u.mm - x['posz'][mask].values
                else:
                    z = x['posz'][mask].values

                mz = np.logical_or(z<z_offset, z>(z_offset*u.mm+u.ccd_thickness))
                # actualize mask
                mask.values[indices[mz]]=False
                if mask.sum() == 0:
                    self.killed = True
                    return
                    
            for keyword,attr in zip(['posx','posy','posz','time','pdg','Edep'],['x','y','z','time','pdg','Edep']):
                values = x[keyword][mask]
                if keyword=='posz' and bool(u.invert_ccd_z_values):
                    values = u.ccd_thickness/u.mm - values
                setattr(self,attr,values)
                setattr(self,'g4_{}'.format(keyword),values)

            self.dim = len(self.x)

        else:
            if len(set([len(x),len(y),len(z)])) > 1:
                raise ValidationError("ArraySizeError: different dimensions for the coordenates")
            self.x = x 
            self.y = y 
            self.z = z 
            self.dim = len(x)
            
            if Edep is not None:
                self.AddEdep(Edep)
            if pdg is not None:
                self.AddPDG(pdg)
            if time is not None:
                self.AddTime(time)                
        
        ### Assuming the CCD will always be rotated 90 degrees (even: no rotation, odd: rotated)
        self.global_posx = abs((2*self.ccd-1)%2) * self.x + abs((2*self.ccd)%2) * self.y 
        self.global_posy = abs((2*self.ccd)%2) * self.x + abs((2*self.ccd-1)%2) * self.y

        ### Check if any of the coordantes has negative values 
        if any(self.x<0.0) or any(self.y<0.0):
            print(" Event {}, at ccd {} is killed due to negative local coordinates".format(self.event,self.ccd))
            self.killed = True

    def AddCoordSimulatedEvent(self,reg_evt,fields2include=['posx','posy','posz','momx','momy','momz','energy']):
        """Link primary particle parameters to the hit collection (events that hitted the CCDs at
        least once)

        Parameters
        ----------

        reg_evt : pandas.DataFrame
            data frame containing the parameters for the given event, i.e. EventOut tree as data
            frame

        fields2include : list
            list of parameters from the EventOut tree to append to the hits collections

        """
        for attr in fields2include:
            values = getattr(reg_evt,attr).values
            if values.size == 1:
                values = values[0]
            setattr(self,'pp_{}'.format(attr), getattr(reg_evt,attr).values)

    def AddEdep(self,E):
        """Add energy in untis of eV to the hits collection (**only signature 2**)
        """
        if len(E) != self.dim:
            raise ValidationError("ArraySizeError: Edep dimension is not {}".format(self.dim))
        setattr(self,'Edep',E)

    def AddTime(self,t):
        """Add time in units of seconds to the hits collection (**only signature 2**)
        """
        if len(t) != self.dim:
            raise ValidationError("ArraySizeError: time dimension is not {}".format(self.dim))
        setattr(self,'time',t)

    def AddPDG(self,pdg):
        """Add PDG number to the hits collection (**only signature 2**)
        """
        if len(pdg) != self.dim:
            raise ValidationError("ArraySizeError: PDG dimension is not {}".format(self.dim))
        setattr(self,'pdg',pdg)

  
    def get_roi_fixed_length(self,img_size,min_enlarge=30):
        """Add the ROI (region of interest) attribute to the G4HitCollection instance.
        
        When the intrinsic noise is not simulated for the full CCD image, a translation of 
        the hit collection from the full CCD image to one with smaller dimensions must be done.

        This method is only used when Dark Currend and/or Electronic noise are used under the mode
        **cropped-image**. This mode simulate the intrinsic detector noise in a cropped image
        surrounding the hit collection, with a maximum size in the x- and y-axis given by
        **img_size**, and for those cases where the cluster elongation is larger than the given image
        size, the szie of the simulated cropped image will be maximum cluster-elongation plus
        **min_enlarge** pixels.

        Parameters
        ----------
            img_size : int
                pixels size of the image for the intrinsic detector noise when running under the
                mode `cropped-image`
            
            min_enlarge : int
                number of pixels to add to the cropped image when the cluster has an elongation
                larger than the pixels given by *img_size*

        The Attribute added to the hit collection instance is a dictionary with the following
        parameters:

            self.roi  = {
                'shift': { 
                          'x':(translation from full image size to the cropped image, image halph size),
                          'y':(translation from full image size to the cropped image, image halph size)
                          },
                'size': (image size in the x-axis, image size in the y-axis)
                }
        """
        x_min=min(self.x_pixel)
        x_max=max(self.x_pixel)
        Dx=(x_max-x_min)+1

        y_min=min(self.y_pixel)
        y_max=max(self.y_pixel)
        Dy=(y_max-y_min)+1

        ### cluster is larger than image size, enlarge image
        if Dx>img_size or Dy>img_size:
            img_size=max([Dx,Dy])+min_enlarge

        ### cluster will be placed in the center of the image, except when it is close to the border
        ###     in the real image size
        Lx=int((img_size-Dx)/2.)
        ### check if cluster is in the lower and upper border on the x-axis
        if Lx>x_min:
            ### in the lower border
            x0=x_min
        elif (u.ccd_shape[1]-x_max)<Lx:
            ### in the upper border
            x0=img_size-(u.ccd_shape[1]-x_max)-Dx
        else:
            x0=int((img_size-Dx)/2.0)

        Ly=int((img_size-Dy)/2.0)
        if Ly>y_min:
            y0=y_min
        elif (u.ccd_shape[0]-y_max)<Ly:
            y0=img_size-(u.ccd_shape[0]-y_max)-Dy
        else:
            y0=int((img_size-Dy)/2.0)
        
        self.roi={
                'shift':{
                    'x':(x0,img_size-x0-Dx),
                    'y':(y0,img_size-y0-Dy)
                    },
                'size':(img_size,img_size)
                }
        
        #print("REGION of interest: ")
        #print(self.roi)

    def get_total_signal(self,threshold=0.0):
        """Add intrinsic detector noise to the simulated signal.

        This method also applies 
        
            * an energy cut (threshold, by default 0) and/or 

            * pixel saturation (if this process is set, active=1 in the configuration json file).

        Parameters
        ---------- 
            threshold : float
                Minimum energy (in eV) to consider a pixel charged as signal (energy loss plus
                noise).
                For instance, if SignalPatterRecognition is active, this process call
                get_total_signal with a threshold energy cut that the user have been set trhough the
                configuration json file

        Note
        ----
            PixelSaturation is included here. 
            PixelSaturation if is active, set the maximum charge that a pixel can account for to the
            attribute Units.pix_saturation, which is used in this function to get the total signal
            in units of eV.

        """
        if self.mode == 'full-image':
            self.get_total_signal_full_image(threshold)
        elif self.mode == 'cropped-image':
            self.get_total_signal_cropped_image(threshold)
        else:
            msm = "Running mode `{}` is not implemented. Try with `full-image` or `cropped-image`".format(self.mode)
            raise AttributeError(msm)

    def get_total_signal_cropped_image(self,threshold=0.0):
        """ :method:`Cluster.get_total_signal` for the running mode **cropped-image**

        A cropped image will be crated for each hit collection, with a size much smaller than the full detector one.
        This made the process of pattern recognition (filtering the signal >threshold) quite efficient.
        
        This process is far more faster than the mode **full-image**, where a single image is simulated with the 
        full detector size, making the pattern recognition signal quite slow.

        Parameters
        ---------- 
            threshold : float
                Minimum energy (in eV) to consider a pixel charged as signal (energy loss plus
                noise).
                For instance, if SignalPatterRecognition is active, this process call
                get_total_signal with a threshold energy cut that the user have been set trhough the
                configuration json file

        Note
        ----
            PixelSaturation is included here. 
            PixelSaturation if is active, set the maximum charge that a pixel can account for to the
            attribute Units.pix_saturation, which is used in this function to get the total signal
            in units of eV.

        """
 
        t1 = time.time()

        x_pixel_T = self.x_pixel + self.roi['shift']['x'][0] - min(self.x_pixel)
        y_pixel_T = self.y_pixel + self.roi['shift']['y'][0] - min(self.y_pixel) 


        zeros_image = np.zeros(self.roi['size'])
        ### Size of all parameters on self change from x_pixel.size to roi.size
        for attr,dv in zip(["Edep_pixel","time_pixel","pdg_pixel","z_pixel","std_time_pixel","time_img_pixel","neutron_pixel","gamma_pixel","silicon_pixel","alpha_pixel","ion_pixel","electron_pixel"],
                [0,-1,-1,-1,-1,-1,0,0,0,0,0,0]):
            values=getattr(self,attr)
            img_noise = zeros_image+dv
            img_noise[y_pixel_T,x_pixel_T]=values
            setattr(self,"{}_img_noise".format(attr),img_noise)
        
        ########################################################
        ### Store signal and noise, as an ndarray, separatelly
        img_pixel_noise = zeros_image
        if hasattr(self,'noise'):
            for n in self.noise:
                img_pixel_noise+=n

        self.CCD_image = {'noise':img_pixel_noise.copy() }
        ########################################################    

        ### -- ADD ALL NOISES TO THE SIMULATED LOST ENERGY  --
        self.Edep_pixel_img_noise+=img_pixel_noise

        #### APPLY SATURATION
        #### value of saturation is 0 by default, only process PixelSaturation can update this value
        if u.pix_saturation>0:
            self.Edep_pixel_img_noise[self.Edep_pixel_img_noise>u.pix_saturation]=u.pix_saturation

        ### PIXELS WITH NON-ZERO CHARGE
        ### get pixels with signal, and stored as arrays
        ### SignalPatterRecognition can pass another value to threshold different of zero
        pix_rows,pix_cols = np.where(self.Edep_pixel_img_noise>threshold)

        ### PIXELS,x e y 
        self.x_pixel_noise = pix_cols + (min(self.x_pixel)-self.roi['shift']['x'][0])
        self.y_pixel_noise = pix_rows + (min(self.y_pixel)-self.roi['shift']['y'][0])

        ### Get the values for the rest of the attributes Z, PDG, TIME and Edep
        for attr in ["Edep_pixel","time_pixel","pdg_pixel","z_pixel","std_time_pixel","time_img_pixel","neutron_pixel","gamma_pixel","silicon_pixel","alpha_pixel","ion_pixel","electron_pixel"]:
            img_noise=getattr(self,"{}_img_noise".format(attr))
            setattr(self,"{}_noise".format(attr),img_noise[pix_rows,pix_cols])

        self.time_gti = time.time()-t1

        #### SET THE VARIABLE TO BE CLUSTERIZED
        self.__attr_to_clusterize__+="_noise"

        #### Add an extra attr when threshold is different from 0
        ####    SPR have been applied
        #if threshold>0:
        #    self.__attr_to_clusterize__+="_SPR"

    def get_total_signal_full_image(self,threshold=0.0):
        """ :method:`Cluster.get_total_signal` for the running mode **full-image**

        Only one intrinsic detector noise image will be created at the beggining of the process chain,
        and used for the whole process.  The simulated image will have the same size as the detector 
        active region.

        This process is far more slowly than the mode **cropped-image**, which simulated only an small 
        region surrounding the cluster.

        Parameters
        ---------- 
            threshold : float
                Minimum energy (in eV) to consider a pixel charged as signal (energy loss plus
                noise).
                For instance, if SignalPatterRecognition is active, this process call
                get_total_signal with a threshold energy cut that the user have been set trhough the
                configuration json file

        Note
        ----
            PixelSaturation is included here. 
            PixelSaturation if is active, set the maximum charge that a pixel can account for to the
            attribute Units.pix_saturation, which is used in this function to get the total signal
            in units of eV.

        """
        t1 = time.time()
        
        zeros_image = np.zeros(self.roi['size'])
        col_offset = u.n_cols_prescan
        row_offset = u.n_rows_prescan

        #### Create an image for the following parameters of interest: Edep, PDG, time, and z
        ####    to differenciate from the signal, add un-logical values: 0, 0, -1, -1, respectively
        for attr,dv in zip(["Edep_pixel","time_pixel","pdg_pixel","z_pixel","std_time_pixel","time_img_pixel","neutron_pixel","gamma_pixel","silicon_pixel","alpha_pixel","ion_pixel","electron_pixel"],
                [0,-1,-1,-1,-1,-1,0,0,0,0,0,0]):
            values=getattr(self,attr)
            img_noise = zeros_image+dv
            img_noise[row_offset+self.y_pixel-1,col_offset+self.x_pixel-1]=values
            setattr(self,"{}_img_noise".format(attr),img_noise)
        
        #### INTRINSIC DETECTOR NOISE: SUM UP ALL IMAGES FROM NOISE ATTRIBUTE
        img_pixel_noise = zeros_image
        if hasattr(self,'noise'):
            for n in self.noise:
                img_pixel_noise+=n

        #### include attribute with noise, and pure signal
        self.CCD_image = {'noise':img_pixel_noise.copy() }
        self.CCD_image.update({'signal': self.Edep_pixel_img_noise.copy()})

        ### -- ADD ALL NOISES TO THE SIMULATED LOST ENERGY  --
        self.Edep_pixel_img_noise+=img_pixel_noise
        
        #### add the total signal to the CCD_image dictionary
        self.CCD_image.update({'total': self.Edep_pixel_img_noise.copy()})

        ##################################################################################
        ####    APPLY SATURATION
        ##################################################################################
        #### value of saturation is 0 by default, only process PixelSaturation can update 
        #       this value
        if u.pix_saturation>0:
            self.Edep_pixel_img_noise[self.Edep_pixel_img_noise>u.pix_saturation]=u.pix_saturation

        ##################################################################################
        ####    TRHESHOLD CUT: default 0
        ##################################################################################
        ### PIXELS WITH NON-ZERO CHARGE
        ### get pixels with signal, and stored as arrays
        ### SignalPatterRecognition can pass another value to threshold different of zero
        pix_rows,pix_cols = np.where(self.Edep_pixel_img_noise>threshold)

        ##################################################################################
        #   Add 1D arrays for all attributes
        #       x, y, z, Edep, time, PDG, ...
        #
        ### PIXELS,x e y 
        self.x_pixel_noise = pix_rows
        self.y_pixel_noise = pix_cols

        ### Get the values for the rest of the attributes Z, PDG, TIME and Edep
        for attr in ["Edep_pixel","time_pixel","pdg_pixel","z_pixel","std_time_pixel","time_img_pixel","neutron_pixel","gamma_pixel","silicon_pixel","alpha_pixel","ion_pixel","electron_pixel"]:
            img_noise=getattr(self,"{}_img_noise".format(attr))
            setattr(self,"{}_noise".format(attr),img_noise[pix_rows,pix_cols])

        self.time_gti = time.time()-t1

        #### SET THE VARIABLE TO BE CLUSTERIZED
        self.__attr_to_clusterize__+="_noise"

    def Draw(self,Npix_min,only_this_event=None,box_size=300):
        """
        Plot for degub
        
        The main plot has all the clusters on the hit collection. 
        One or more subplots will be plotted on a side, to zoom all clusters with number of pixels
        higher then Npix_min.

        Parameters
        ----------
            Npix_min : int
                minimum number of pixels that the cluster should have to be plotted as a subplot

        """
        if only_this_event is not None:
            if self.event not in only_this_event:
                return

        if not hasattr(self, 'roi'):
            self.mode = 'cropped-image'
            self.get_roi_fixed_length(box_size,30)
            self.get_total_signal()

        #if hasattr(self,"Edep_pixel_img_noise"):
        #    img=self.Edep_pixel_img_noise/u.keV
        #    mimg=np.ma.masked_array(img,mask=img<1E-9)
        #    vmax=img.max()
        #else:
        #    img=np.zeros(self.roi['size'])
        #    mimg=np.ma.masked_array(img,mask=img<1E-9)
        #    vmax=max(getattr(self,"Edep_pixel")/u.keV)
        #    print(self.Edep_pixel)
        #    print(vmax)
        
        #### INITIALIZE FIGURE WITH SUBPLOTS
        ###############################################################################################
        fig = plt.figure(figsize=(14,7))
        gs = GridSpec(nrows=2,ncols=1) 

        #### CLUSTERS PLOTS
        ###############################################################################################
        #if hasattr(self,"Edep_pixel_img_noise"):
        #    img=self.Edep_pixel_img_noise/u.keV
        #    mimg=np.ma.masked_array(img,mask=img<1E-9)
        #    vmax=img.max()

        x_cls=getattr(self,"x{}".format(self.__attr_to_clusterize__))
        y_cls=getattr(self,"y{}".format(self.__attr_to_clusterize__))
        # translate to image ROI
        x=x_cls-(min(self.x_pixel)-self.roi['shift']['x'][0])
        y=y_cls-(min(self.y_pixel)-self.roi['shift']['y'][0])
        
        #### create the grid to plot cluster by cluster individually
        Ncls = 0
        for cls_i,cls_mask in self.cluster_mask.items():
            if len(y[cls_mask])>=Npix_min:
                Ncls+=1
        #if Ncls >=2:
        #    gs_rows=[0,1]
        gs = GridSpec(nrows=2,ncols=Ncls+2)
        
        Nrow,Ncol=self.roi['size']
        # and highlight only clusters greather than 1 pixel (one by one)
        # from image
        img=self.Edep_pixel_img_noise/u.keV
        mimg=np.ma.masked_array(img,mask=img<1E-9)
        vmax=img.max()

        ax_cls=[]
        kcol=0
        for cls_i,cls_mask in self.cluster_mask.items():
            if len(y[cls_mask])>=Npix_min:
                # create an image with all pixels masked
                img_cls = np.ones(Ncol*Nrow)
                img_cls = img_cls.astype(bool)
                ### add a subplot for each cluster
                ax_cls.append(fig.add_subplot(gs[1,kcol+2]))
                kcol+=1

                ### masking the cluster
                xp=x[cls_mask]
                yp=y[cls_mask]
                ind=yp*Ncol+xp
                img_cls[ind]=False
                x1,x2,y1,y2=min(xp),max(xp),min(yp),max(yp)
                mimg_cls=np.ma.masked_array(img,mask=img_cls)
                cls_cmap=ax_cls[-1].imshow(mimg_cls,origin='lower',aspect=1,cmap='summer',vmax=vmax)
                Lx=x2-x1
                Ly=y2-y1
                ax_cls[-1].set_ylim(y1-5,y1+max([Lx,Ly])+5)
                ax_cls[-1].set_xlim(x1-5,x1+max([Lx,Ly])+5)
                #### most frequent pdg in these cluster
                pdg_np=getattr(self,"pdg{}".format(self.__attr_to_clusterize__))
                pdg = list(pdg_np[cls_mask])
                PDG_mfv = int(max(pdg,key=pdg.count))
                pdg=(np.array(pdg)).astype(int)
                if 10000200401 in pdg:
                    PDG=int(10000200401)
                elif 13 in pdg:
                    PDG=int(13)
                else:
                    PDG=PDG_mfv
                
                ax_cls[-1].set_title("cluster id {} \n E={} eV \n most frequent PDG {} \n assigned PDG {}".format(
                    cls_i,round(mimg_cls.sum()*1000.,2),PDG_mfv,PDG), fontsize=11)

        ax = fig.add_subplot(gs[:,:2])
        ##### PLOTTING electron-hole pairs (in case diffusion was set)
        ###############################################################################################
        #### e-h pairs (IF DIFFUSION IS APPLIED)
        if True:#self.g4_posx.shape != self.x.shape:
            yval=self.y/u.ccd_pixel_size_y-(min(self.y_pixel)-self.roi['shift']['y'][0])
            xval=self.x/u.ccd_pixel_size_x-(min(self.x_pixel)-self.roi['shift']['x'][0])
            ax.scatter(xval,yval,facecolors='#f70227',s=20,marker="x",edgecolors="#f70227",alpha=0.5,
                    label="w/Diffusion")
            
        #### GEANT4 SIMULATIONS: FOR PDG TYPE
        ###############################################################################################
        for pdg in set(self.g4_pdg):
            mask_pdg = self.g4_pdg==pdg
            x = self.g4_posx[mask_pdg]/u.ccd_pixel_size_x-(min(self.x_pixel)-self.roi['shift']['x'][0])
            y = self.g4_posy[mask_pdg]/u.ccd_pixel_size_y-(min(self.y_pixel)-self.roi['shift']['y'][0])
            #### size as a function of the z-values
            z = self.g4_posz[mask_pdg].values/u.um
            max_z=u.ccd_thickness/u.um
            if max(z) > u.ccd_thickness/u.um:
                max_z=max(z)
            z_norm = plt.Normalize(min(z),max_z)
            z_sizes = z_norm(z)*225+25
            #### sum up the total energy per pdg
            e = sum(self.g4_Edep[mask_pdg]/u.keV)
            #### get colors and markers
            #dp = ('red','o','red' )#
            dp = (particle_colors(pdg),particle_markers(pdg),'none')
            ax.scatter(x,y,s=40,marker=dp[1],edgecolor=dp[0],facecolor=dp[2],linewidths=3,
                    label="PDG={}, E={}eV".format(pdg,round(e*1000,1)), alpha=0.5)
             
        #### [NOISE]+SIGNAL IMAGE
        ###############################################################################################
        if hasattr(self,"Edep_pixel_img_noise"):
            img=self.Edep_pixel_img_noise/u.keV
            mimg=np.ma.masked_array(img,mask=img<1E-9)
            img_cmap=ax.imshow(mimg,origin='lower',extent=[0,self.roi['size'][1],0,self.roi['size'][0]],
                    aspect=1,cmap='summer')
            cb=fig.colorbar(img_cmap,ax=ax)
            ax.set_aspect('auto')
            cb.ax.set_ylabel("Edep [keV]",rotation=270, labelpad=10)


        #### plotting
        ###############################################################################################
        fig.suptitle(r"{\sc \bf Simulation of the detector response and Cluster finder (subplots)}",size=25)
        fig_title="ROI for event,ccd=({},{}) at  z $\in$[{},{}]um".format(self.event,
                self.ccd,round(self.z_pixel.min()/u.um,0),round(self.z_pixel.max()/u.um,0))
        plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=False,shadow=False,
                prop={'size':10}, title=fig_title)
        plt.xlabel("pixel (y0={})".format(min(self.y_pixel)-self.roi['shift']['y'][0]))
        plt.ylabel("pixel (x0={})".format(min(self.x_pixel)-self.roi['shift']['x'][0]))

        kwargs = {'color':'#bad1be'}
        plt.figtext(0.6,0.01, r'{\bf psimulCCDimg}, \emph{N\'uria Castell\'o Mor} (IFCA), castello@ifca.unican.es', 
                horizontalalignment='right', fontsize=13, color='#315094')
        
        plt.subplots_adjust(top=0.8, bottom=0.11, left=0.10, right=0.95, hspace=0.15, wspace=0.25)
#        plt.savefig("debug_plots_psimulCCDimg_reconstruction_evt{}_ccd{}.eps".format(self.event,self.ccd))
        plt.show()


    def SaveCCDimg(self,mask,outfile,max_size):
        # XXX DEPRECATED XXX

        ### most probable value for the cluster
        pdg_values = getattr(self,"pdg{}".format(self.__attr_to_clusterize__))
        (values,counts)=np.unique(pdg_values[mask], return_counts=True)
        PDG_mpv=int(values[np.argmax(counts)])

        ### excluding noise
        if PDG_mpv in [0]:
            return
        
        ### for alpha particle with high energy (some of the clusters are lost as e-cls
        if 10000200401 in values:
            PDG_mpv=int(10000200401)
        
        ### output file name
        fname="{}_image_event_{}_ccd_{}_ind_{}_pdg_{}.npz".format(outfile,self.event,self.ccd,self.npz_ind,PDG_mpv)

        # create the image for the signal only, based on the size of the noise image
        Lx,Ly=self.CCD_image['noise'].shape
        img_energy=np.zeros((Lx,Ly))
        # random position for the cluster
        x0=int(np.random.uniform(0,Lx,1)[0])
        y0=int(np.random.uniform(0,Ly,1)[0])
        # translate the positon of the cluster to the image at the random position
        x_pixel=getattr(self,"x{}".format(self.__attr_to_clusterize__))
        y_pixel=getattr(self,"y{}".format(self.__attr_to_clusterize__))
        x = x_pixel[mask] - min(x_pixel[mask]) + x0
        y = y_pixel[mask] - min(y_pixel[mask]) + y0

        # check if in the random position is fully encapsulated by the image Lx,Ly
        mask_xy=np.logical_and(x<Lx,y<Ly)
        x=x[mask_xy]
        y=y[mask_xy]
        if len(x)==0:
            #### completely out of the ccd
            x0=0
            y0=0
            x = x_pixel[mask] - min(x_pixel[mask]) + x0
            y = y_pixel[mask] - min(y_pixel[mask]) + y0
            mask_xy=np.logical_and(x<Lx,y<Ly)
            x=x[mask_xy]
            y=y[mask_xy]

        Ex=max(x)-min(x)+1
        Ey=max(y)-min(y)+1
        energy=getattr(self,"Edep{}".format(self.__attr_to_clusterize__))
        img_energy[x,y]=energy[mask][mask_xy]

        if Lx == max_size:
            np.savez(fname, energy=img_energy, noise=self.CCD_image['noise'])
            self.npz_ind+=1
        
class G4InfoTree(object):
    """G4Info Tree

    Parameters
    ----------
        hits : :obj:`G4HitCollection`
            collection of hits from the geant4 simulations
    
    """
    
    def __init__(self,infile,n_file,Ncls,tree='RunInfo'):
        
        g4file = ROOT.TFile.Open(infile)
        g4tree = g4file.Get('RunInfo')
        g4tree.GetEntry(0)
        g4tree.SetDirectory(0)
        g4file.Close() 

        self.Nfile= int(n_file)
        self.Ncls = int(Ncls)

        # number of events simulated per file
        self.NEvts = int(getattr(g4tree,'NEvts'))
        #  number of simulated CCDs
        try:
            self.NCCDs = int(getattr(g4tree,'NCCDs'))
        except (KeyError,AttributeError) as _error:
            self.NCCDs = int(getattr(g4tree,'NCCD'))
        # random Seed used for geant4 sims
        self.Seed = int(getattr(g4tree,'Seed'))
        
        # simulated PP
        self.pp_gps = ROOT.std.string(g4tree.primaryParticle)
        # simulated ion 
        self.pp_ion = ROOT.std.string(g4tree.primaryIon)

        # properties of the simulated sensitive detector
        detector = G4Volume(g4tree,u.detector_name)
        if len(detector.name)>1:
            print("""\x1b[34m Possible Error: Found more than one sensitive detector with name {}\x1b[m""".format(u.detector_name))
        self.ccd_mass    = float(detector.mass[0])
        self.ccd_vol     = float(detector.volume[0])
        self.ccd_density = float(detector.density[0])
        self.ccd_surface = int(detector.surface[0])
        
        # properties of the volume where PP was placed
        ppvol = G4Volume(g4tree,g4tree.simulatedVolume)
        self.pp_vol_n = int(len(ppvol.name))
        if self.pp_vol_n>1:
            self.pp_vol_pvnames = ROOT.std.string(';'.join(ppvol.name))
            self.pp_vol_density = ppvol.density
            self.pp_vol_mass    = ppvol.mass
            self.pp_vol_volume  = ppvol.volume
            self.pp_vol_surface = ppvol.surface
        else:
            self.pp_vol_pvnames = ROOT.std.string(ppvol.name[0])
            self.pp_vol_density = float(ppvol.density[0])
            self.pp_vol_mass    = float(ppvol.mass[0])
            self.pp_vol_volume  = float(ppvol.volume[0])
            self.pp_vol_surface = float(ppvol.surface[0])



class ProcessConfigTree(object):
    """G4Info Tree

    Parameters
    ----------
        hits : :obj:`G4HitCollection`
            collection of hits from the geant4 simulations
    
    """
    
    def __init__(self,process_list,tree='detector_config'):

        ### Read all process in the list, and add all  configuration parameters as a data member of
        #       this class
        
        self.waders_commit  = ROOT.std.string(__commit__)
        self.waders_version = ROOT.std.string(__version__)

        for p in process_list:
            add_sequecne = True
            p_attr = p.__dict__

            for k,v in sorted(p_attr.items()):
                if k in ['exit','show_fit','histequ',"__units__","__DEBUG__","__display__","__verbose__"]:
                    continue

                if not type(v) in [int,float,dict]:
                    continue

                if k == 'saturation':
                    v = v/u.ADC2eV
                if k == '__sequence_id__':
                    k = 'sequence_id'
                    add_sequecne = False
                
                if type(v) == dict:
                    for sk,skv in v.items():
                        # add attribute to the class
                        if type(skv) in [int,float]:
                            if not np.isnan(skv):
                                setattr(self,"{}_{}_{}".format(p.__name__,k,sk),skv)
                else:
                    if not np.isnan(v):
                        # add attribute to the class
                        setattr(self,"{}_{}".format(p.__name__,k),v)

            if add_sequecne:
                setattr(self,"{}_sequence_id".format(p.__name__), p.__sequence_id__)

        ### Attributes from the Units system
        #
        for attr in ['e2eV','ADC2e','ADC2eV','img_exp_time']:
            try:
                v = getattr(u,attr)
                setattr(self,attr,v)
            except AttributeError:
                pass


