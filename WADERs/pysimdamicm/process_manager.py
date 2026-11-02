### PROCESS RELATED TO SIMULATED DATA
### process to simulate the response of the DAMIC-M/LBC detector
from pysimdamicm.processes.detector_response import Diffusion, ContinuousReadout, PixelizeSignal,DarkCurrent, ElectronicNoise, PixelSaturation, PasteClusterProcess
### PROCESS FOR RECONSTRUCTION: BOTH DATA AND SIMULATIONS
from pysimdamicm.processes.reconstruction import ClusterFinder, CreateFitsImage, ApplySelectionCuts, BuildClusterMask
### PROCESS RELATED WITH REAL DATA: LOW LEVEL
from pysimdamicm.processes.skipper_analysis import CompressSkipperProcess, PedestalSubtractionProcess, FitDarkCurrentProcess, CalibrationProcess, CorrectElectronicColumnTransient, AddBKGComponent, FitDarkCurrentPerRow, FitDarkCurrentPerCol, GaussianFitProcess, MeanPixelChargePerAxis, TwoDGaussProcess, CrossTalkProcess, GaussianNoiseProcess, EvalCTIProcess, EvalHaloProcess
from pysimdamicm.processes.skipper_comissioning import ChargeLossPlot, FFTNoisePlot, RNvsNskipsPlot, FitCalibrationConstant, ChargeLossSkewnessProcess
### SINGLETON: UNITS CLASS WITH GLOBAL PARAMETERS
from pysimdamicm.utils.units import Units
u=Units()

#### other python packages externals to pysimdamicm
import ROOT
from array import array

###############################################################################################
#####       RESPONSE OF THE DETECTOR (FULL RESPONSE,. i.e. all process are account for)
#####       AND (CLUSTER) RECONSTRUCTION
###############################################################################################
class ActivateProcessProperty(object):
    """Descriptor to create dynamically the activate process properties.

    For each :obj:`DigitizeProcess` added to the `ProcessManager` class, an attribute
    `activate_<DigitizeProcess.__name__>` must be included to active or
    unactivate such digitize/reconstruc process during the configuration
    of the data process manager.

    The creation of such attribute is done dynamically by using the method
    `__set__` of this class.

    Parameters
    ----------
    manager : :obj:`ProcessManager`

    process_name : str
        The name of the DigitizeProcess given by `DigitizeProcess,.__name__`
    """
    def __init__(self,manager,process_name):
        #### if the process do not exist, KeyError will raise
        self._process_class = manager.__valid_processes__[process_name]

    def __get__(self,inst_manager,class_type):
        return len(list(filter(lambda p: isinstance(p,self._process_class),inst_manager.__sequence__))) == 1

    def __set__(self,inst_manager,value):
        # Check if the process is in the sequence (needed in the two cases)
        # If not None is returned
        process = inst_manager.get_process_from_sequence(self._process_class)
        
        ### remove existing process, to be sure the new instanciation does not contain previous
        #       attributes
        if process is not None:
            inst_manager.__sequence__.remove(process)

        if value == True:
            # If process exists, create a new brand instance
            process = self._process_class()
            inst_manager.__sequence__.append(process)
        elif type(value)!=bool:
            raise Runtime("Not valid value (only bool accepted)")
        #### (re-)sort the remaining sequence
        inst_manager.__sequence__.sort(key=lambda p: p.__sequence_id__)

class ProcessManager(object):
    """
    Class to manage (activate/deactivate, configure, ...) the full detector response
    of the DAMIC-M Experiment and the (cluster) reconstruction
    process.


    The manager involves part of the creation of the RAW data from pure SIM data
    (digitization) and using the RAW data (here created or from real data), the
    pure RECO process such the cluster reconstruction.

    Attributes
    ----------

    """
    __sequence__ = []
    ### REMEMBER whenever you are implementing a new process:
    ### Any DigitizeProcess that can be used by this manager class
    ### must be enumerated here
    __valid_processes__ = {
            ### specific for simulations (only simulations)
            Diffusion.__name__:Diffusion,
            ContinuousReadout.__name__:ContinuousReadout,
            PixelizeSignal.__name__: PixelizeSignal,
            DarkCurrent.__name__:DarkCurrent,
            ElectronicNoise.__name__:ElectronicNoise,
            PixelSaturation.__name__:PixelSaturation,
            CreateFitsImage.__name__:CreateFitsImage,
            PasteClusterProcess.__name__:PasteClusterProcess,
            ### specific for skiper data
            CompressSkipperProcess.__name__:CompressSkipperProcess,
            PedestalSubtractionProcess.__name__:PedestalSubtractionProcess,
            TwoDGaussProcess.__name__:TwoDGaussProcess,
            CalibrationProcess.__name__:CalibrationProcess,
            FitDarkCurrentProcess.__name__:FitDarkCurrentProcess,
            FitDarkCurrentPerRow.__name__:FitDarkCurrentPerRow,
            FitDarkCurrentPerCol.__name__:FitDarkCurrentPerCol,
            GaussianFitProcess.__name__:GaussianFitProcess,
            CrossTalkProcess.__name__:CrossTalkProcess,
            GaussianNoiseProcess.__name__:GaussianNoiseProcess,
            EvalCTIProcess.__name__:EvalCTIProcess,
            EvalHaloProcess.__name__:EvalHaloProcess,
            ### sims and data: adding component on top of the input image
            AddBKGComponent.__name__:AddBKGComponent,
            ### reconstruction: both sims and data
            ClusterFinder.__name__:ClusterFinder,
            ApplySelectionCuts.__name__:ApplySelectionCuts,
            BuildClusterMask.__name__:BuildClusterMask,
            ### related to low level processing (data)
            ChargeLossSkewnessProcess.__name__:ChargeLossSkewnessProcess,
            ChargeLossPlot.__name__:ChargeLossPlot,
            FFTNoisePlot.__name__:FFTNoisePlot,
            RNvsNskipsPlot.__name__:RNvsNskipsPlot,
            FitCalibrationConstant.__name__:FitCalibrationConstant,
            CorrectElectronicColumnTransient.__name__:CorrectElectronicColumnTransient,
            MeanPixelChargePerAxis.__name__:MeanPixelChargePerAxis
            }

    ##### module DEBUG VARIABLES
    __verbose__ = False


    ##### To reduce the simulations time for some processes (such us dark current and noise)
    #####   The simplification is explained in each processs.execute_process method
    __fast_sims__ = False

    def __init__(self,level_of_debug=1):
        # create the process activation properties (XXX Must be created to the class)
        for pname,process_class in self.__valid_processes__.items():
            setattr(ProcessManager,'active_{}'.format(pname),ActivateProcessProperty(ProcessManager,pname))

        self.debug_level=level_of_debug

    @property
    def debug_level(self):
        return self.__debug_level__

    @debug_level.setter
    def debug_level(self,val):
        self.__debug_level__=val
        if self.__debug_level__>1:
            self.debug_plots=self.__debug_plots__
        else:
            self.debug_plots=lambda *args,**kwargs: None

    def __debug_plots__(self,process,hits):
        """
        """
        if process.__name__ in ["Diffusion","PixelizeSignal"]:
            CF=1
            if not hits.__attr_to_clusterize__.count("pixel")>0:
                CF=u.mm2pix

            print(" .... process ", process.__name__)
            xval = getattr(hits,"x{}".format(hits.__attr_to_clusterize__))*CF
            yval = getattr(hits,"y{}".format(hits.__attr_to_clusterize__))*CF
            
            _isbatch = ROOT.gROOT.IsBatch()
            ROOT.gROOT.SetBatch(0)

            c = ROOT.TCanvas()
            tg = ROOT.TGraph( len(xval), array('d',xval), array('d',yval) )
            tg.SetMarkerSize(0.8)
            tg.GetHistogram().GetXaxis().SetTitle("pixel")
            tg.GetHistogram().GetYaxis().SetTitle("pixel")
            tg.Draw("AP")
            input("press enter ... ")
            ROOT.gROOT.SetBatch(_isbatch)


    def get_process_from_sequence(self,process_class):
        """
        Parameters
        ----------
        process_class : :obj:`pysimdamicm.detector_response.DigitizeProcess`
            an instance class of such process class

        Return
        ------
        process_isnt : :obj:`pysimdamicm.detector_response.DigitizeProcess`
            the instance of the input class from pysimdamicm.process_manager.__sequence__
        """
        try:
            return list(filter(lambda p: isinstance(p,process_class),self.__sequence__))[0]
        except IndexError:
            return None
        
    def set_configuration(self,config):
        """Function to set the parameters for the full set of processes
        """        
        for pname in config.process_to_sim:
            if hasattr(self,"active_{}".format(pname)):
                setattr(self,"active_{}".format(pname),True)
            self.set_config_process( pname, config.configuration[pname])

    def set_config_process(self, process_name, model_parameter):
        """
        Function to set the parameters of an specific process

        Parameters
        ----------
        process_name: str
            Class name for the process to be update the parameters of the model

        Raises
        ------
        KeyError: If the process name is not present in __valid_processes__

        RuntimeError: Process not in the current sequence
        """
        process_class = self.__valid_processes__[process_name]
        process_inst  = self.get_process_from_sequence(process_class)
        if process_inst is not None:
            process_inst.set_parameters(**model_parameter)
        else:
            print('InstanceError: No instance of "{}" class is found at <ProcessManager>.__sequence__'.format(process_name))
        
        ## (re-)sort the process in the __sequence__ (in case the user changed its sequence_id value)
        self.__sequence__.sort(key=lambda p: p.__sequence_id__)

    def execute_process(self,hits):
        """ This method will execute all process that have been added to self.__sequence__
        through the configuration JSON file. Any active process in this file is included in the
        process list (private data memeber self.__sequence__).
        
        Each process has an identifier (<process>.__sequence_id__) that is used to sort the 
        processes in the processing chain (self.__sequence__). Before starting the execution the 
        processes are sorted using that identifier, and then executed sequentially.  All that is
        done internally, except for processes that can be executed at any time in the processing
        chain. For this process the user can modify the data member __sequence_id__ from the
        configuration JSON file.

        """
        for k,process in enumerate(self.__sequence__):
            if hits.killed:
                break

            if hasattr(hits,'amplifier'):
                for amp in hits.amplifier.keys():
                    hits.set_amplifier(amp)
                    process.execute_process(hits)
                    if not process._per_amp:
                        # only one time per amplifier (like Compress image, ...)
                        break
            else:
                ### MOSTLY FOR SIMULATIONS
                setattr(hits,'execute_process_in_amp',None)
                process.execute_process(hits)
                #### PLOT ONLY FOR SIMULATIONS // NOT PREPARE TO HAVE AMPLIFIER 
                self.debug_plots(process,hits)

    def close_process(self,hits=None):
        """Some of the process have to be close before ending simulation
        """
        for k,process in enumerate(self.__sequence__):
            if hits is not None:
                process.__del__(hits)
                print("Process {} closed.".format(process.__name__))






