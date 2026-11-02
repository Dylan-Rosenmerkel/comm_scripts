import json
import numpy as np
from pysimdamicm import __path__ as module_path
######################################################################################
##                                                                                  ##
##          FOR SIMULATIONS                                                         ##
##                                                                                  ##
######################################################################################
def get_default_configuration(is_sims=True,is_dqm=False):
    ### Add keys from default json dict
    if is_sims:
        if not is_dqm:
            __default_json__ = module_path[0]+"/json/psimulCCDimg_config_file.json"
        else:
            __default_json__ = module_path[0]+"/json/dqmSKImg_configuration.json"
    else:
        __default_json__ = module_path[0]+"/json/panaSKImg_configuration.json"
    ### read parameters from default json file
    with open(__default_json__) as json_data_file:
        default_config = json.load(json_data_file)

    return default_config

######################################################################################
##                                                                                  ##
##          DEFINITION OF ALL PARAMETERS                                            ##
##                                                                                  ##
######################################################################################
option_help = {
	############### FOR SIMULATIONS
        "detector_name"     :"Sensitive detector name in the geant4 geometry, as appears \n\tin the input root file.",
        "save_CCDimg"       :"Boolean option to store the image for each cluster in a npz"+\
                             "\n\tformat with two independent arrays: `energy` and `noise`"+\
                             "\n\tfor the pixel energy of the geant4 simulation and the pixel"+\
                             "\n\tnoise distribution due to dark current and/or electronic noise.",
        "in_dir_path"       :"Path for <in_root.root_file_name>",
        "root_file_name"    :"ROOT file name of the geant4-based DAMICG4 simulation",
        "prefix_root_name"  :"Output file name prefix, the response of the detector \n\t\t will be sotored as <dir_path>/<prefix_root_name>_<root_file_name>.root",
        "out_dir_path"      :"Path for the output root file",
        "store_pixel_info"  :"If set, the properties of each pixel in the cluster will be stored in the output ROOT file",
        "A"                 :"Coefficient for the xy dispersion measured by the \n\t\t detector as a function of the depth",
        "B"                 :"Coefficient for the xy dispersion measured by the \n\t\t detector as a function of the depth",
        "z_offset"          :"Starting z-position in the silicon bulk",
        "fanofactor"        :"Fano factor to model e-h pair creation \n\t\t within the silicon bulk",
        "pixel_read_time"   :"Time to read a pixel in units of seconds",
        "ampli"             :"Readout: number of amplifiers used for skipper\n\t\t continous readout: 1/2/4",
        "N_pixels_in_x"     :"Number of pixels along the x-axis",
        "N_pixels_in_y"     :"Number of pixels along the y-axis",
        "pixel_size_in_x"   :"Pixel size along the x-axis in units of mm",
        "pixel_size_in_y"   :"Pixel size along the y-axis in units of mm",
        "shift_pixel_in_x"  :"Skip the first `shift_pixel_in_x` pixels from the left (x-axis)",
        "shift_pixel_in_y"  :"skip the first `shift_pixel_in_y` pixels from the left (y-axis)",
        "pedestal"          :"Mean value in units of electrons/pixel",
        "sigma"             :"STD value in units of electrons/pixel",
        "img_size"          :"x and y-axis CCD region size. Both noises, dark current and electronic noise,\n\t\t are simulated in a reduced CCD region for computational reassons",
        "min_enlarge"       :"Minimum extra-pixel size when the elongation of the cluster is equal to img_size",
        "darkcurrent"       :"Dark current value in units of e-/pixel/day",
        "exp_time"          :"Time for the darkcurrent mesurement in units of days, being darkcurrent*exp_time\n\t\t the lambda parameter in the poisson distribution",
        "saturation"        :"Number of ADC units for a saturated pixel",
        "method"            :"Algorithm number for cluster finder",
        "max_nearest_neighbor" :"Maximum distance between two pixels with non-zero-charge that belong to the same cluster",
        "threshold"         :"Minimum pixel charge as signal",
        "ADC2eV"            :"Convertion factor from ADC units to keV",
        "e2eV"              :"Mean energy to create an e-h pair in the silicon \n\t\t bulk at 140K (in units of keV)",
        "active"            :"Include the detector response and/or reconstruction process to the data acquisition process chain",
        "alpha"             :"Parameter for the second term on the diffusion model, the proportionality with the energy in units of pixel/eVee",
        "alpha0"            :"Parameter for the energy-independent term of the diffusion model",
        "mode"              :"Full-image or cropped-image for intrinsic detector noise simulation",
        "model"             :"Use karthick to apply probability distributions to get the number of e-h pairs",
        "units"             :"informative keyword to highlight the units of each process parameter",
        "blank_outdir"      :"directory to record the blank+simulation image files",
        "blank_dir"         :"<directory>/patter_file_name to load blank images",
        "extension"         :"extension number for the blank images",
        "row_cls_dist_fx"   :"model to randomly generate a column position [linear/compton]",
        "clusters_per_blank":"event rate per image, i.e. number of events to be pasted in each blank  image",
        "in_ADU"            :"unset If blank is not in ADU units",
        "only_pair_creation":"Set only pair eh creation, and do not simulate diffusion, both are contained in the same process",
        "row_readtime"      :"Exposure time used in the dark current process, time to read a full row",
        ############### comm_scripts additions (ClusterFinder, GainNonlinearityProcess)
        "algorithm"         :"ClusterFinder: 'waders' (threshold + connect, default) or 'agmx' (A_GMX model-based clustering)",
        "agmx"              :"ClusterFinder with algorithm 'agmx': {crowded: auto|on|off, crowded_shapes: on|off,\n\t\t fe_candidates: true|false, temperature: K, first_row: input.image.id_row_start}",
        "spr_threshold_auto":"ClusterFinder: true or {alpha, seed_e}: pixel/seed thresholds in electrons from the\n\t\t dark charge of each CCD (smallest k >= 2 with P(Poisson >= k) <= alpha); unresolved CCDs keep spr_threshold",
        "model_file"        :"GainNonlinearityProcess: *_gain_vs_charge.json from 'gain_vs_charge measure' (g(n) per extension)",
        "anchor"            :"GainNonlinearityProcess: 'comb' (g(n) incl. its low-charge gain, default) or 'fit' (keep the\n\t\t calibration gain at low charge)",
        "temperature"       :"GainNonlinearityProcess: CCD temperature [K] for the energy per electron of the corrected energies"
        }


##########################################################################################
##########################################################################################
class Config(object):
    """Class to read Configuration parameters from a JSON file
    """
    def __list__(self):
        print("Valid options are: ")

        for key in self.__default_config__.keys():
            print('\x1b[31m"{}"\x1b[m:'.format(key))
            for sub_key in self.__default_config__[key].keys():
                try:
                    print('    \x1b[33m"{}"\x1b[m : {} (by default \x1b[32m {}\x1b[m)'.format(sub_key,option_help[sub_key],
                        self.__default_config__[key][sub_key]))
                except KeyError:
                    print('    \x1b[34m"{}"\x1b[m:'.format(sub_key))
                    try:
                        for sub_sub_key in self.__default_config__[key][sub_key].keys():
                            print('      \x1b[33m"{}"\x1b[m : {} (by default \x1b[32m {}\x1b[m)'.format(sub_sub_key,option_help[sub_sub_key],
                                self.__default_config__[key][sub_key][sub_sub_key]))
                    except AttributeError:
                        continue

    def __init__(self,cfile=None, simulations=True, is_dqm=False):
        self.__default_config__ = get_default_configuration(simulations,is_dqm)
        ### Add keys from default json dict
        self.__valid_options__ = self.get_keys(self.__default_config__)

        ### Load config params from json file
        if cfile is not None:
            self.read_file(cfile)
            ### check JSON options
            self.validate_json() 
        else:
            self.__list__()

    def validate_json(self):
        """Method to list all parameters from JSON file that will not be used (if any)
        """
        json_keys = self.get_keys(self.configuration)
        
        _ignore_params = set()
        if not json_keys.issubset( self.__valid_options__ ):
            _ignore_params = json_keys.difference( self.__valid_options__ )
        
        #if len(_ignore_params)>0:
        #    self.__raise_warning__(_ignore_params)

    def get_keys(self,d):
        """Get all keys from a dictionary of dictionary (only 3 levels are allowed)
        """
        list_of_keys = []
        for _key in d.keys():
            list_of_keys.append( _key )
            for _sub_key in d[_key].keys():
                list_of_keys.append( _sub_key )
                try:
                    for _sub_sub_key in d[_key][_sub_key].keys():
                        list_of_keys.append( "{}.{}".format(_sub_key,_sub_sub_key) )
                except AttributeError:
                    continue
        return set(list_of_keys)


    def __raise_warning__(self,param):
        for p in param:
            msm = "WARNING: Parameter <{}> will be ignored (Not implemented)".format(p)
            print("\x1b[32m {} \x1b[m".format(msm))

    def read_file(self, cfile):
        """Load JSON file as dictionary
        """
        self.file_name = cfile
        with open(cfile) as json_data_file:
            self.configuration = json.load(json_data_file)
    
    def __set_process_sequence__(self):

        if "process" in self.configuration and "sequence" in self.configuration["process"]:
            process_sequence = self.configuration["process"]["sequence"].split(";")
            self.configuration["process"].pop("sequence",None)
            if len(process_sequence) == 0:
                return
            
            ### Activate all process in the keyword process.sequence
            process_sequence_done = process_sequence.copy()
            for process in self.configuration["process"].keys():
                if process in process_sequence:
                    self.configuration["process"][process]["active"] = True
                    self.configuration["process"][process]["__sequence_id__"]=process_sequence.index(process)
                    process_sequence_done.remove(process)
                else:
                    self.configuration["process"][process]["active"] = False

            ### active also those process not defined for the user, and use the degault
            #       configuration
            for process in process_sequence_done:
                self.configuration["process"][process]["active"] = True
                self.configuration["process"][process]["__sequence_id__"]=process_sequence.index(process)

    def activate_configuration(self):
        """Set all configurarion parameters, either from the JSON file (if proceed) 
        or set to default values.
        
        Set the process attributes for any detector response and/or reconstruction processes 
        if they appear as active in the configuration JSON file; ignore the rest of processes.

        """ 
        self.__set_process_sequence__()

        __configuration = self.__default_config__
        self.process_to_sim = []
        
        ### Search for those process that should be simulated        
        for key in self.configuration.keys():
            for sub_key in self.configuration[key].keys():
                if type(self.configuration[key][sub_key]) is dict:
                    try:
                        if bool(self.configuration[key][sub_key]["active"]):
                            __configuration[sub_key] = self.configuration[key][sub_key]
                            del __configuration[sub_key]["active"]
                            self.process_to_sim.append( sub_key )
                            
                            print(" Config.activate_configuration --- Add {} process to the simulated process chain.".format(sub_key))
                        else:
                            __configuration[key][sub_key]["active"] = False
                    except KeyError:
                        __configuration[key][sub_key] = self.configuration[key][sub_key]
                else:
                    __configuration[key][sub_key] = self.configuration[key][sub_key]

        self.configuration = __configuration


