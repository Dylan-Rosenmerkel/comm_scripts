""":mod:`pysimdamicm.processes.skipper_analysis`
    
    skipperimg is a collection of algorithms for CCD image processing
    Module of the main methods for the skiat of a CCD image      
    
.. moduleauthor:: Nuria Castello-Mor
"""

from pysimdamicm.processes.absp import SKImageProcess
from pysimdamicm.utils.libplot4ana import ImagePlot
from pysimdamicm.utils.units import Units
from pysimdamicm.utils.ratioplot import create_ratio_plot

u=Units()

import numpy as np
import pandas as pd
import ROOT
### For ROOT TGraph and TH1F
from array import array
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.stats import poisson, trim_mean
from scipy.stats.mstats import trimmed_std
from scipy import stats
import itertools

from pysimdamicm.utils.root_plot_styles import damicmStyle
style = damicmStyle()
style.cd()
ROOT.gROOT.ForceStyle()
ROOT.gErrorIgnoreLevel = ROOT.kWarning

from memory_profiler import profile

#################################################################################################
#                                                                                               #
#       Process Class to compress an skipper image by reducing the skips measurments            #
#           according the user function (or list of functions                                   #
#                                                                                               #
#################################################################################################

class CompressSkipperProcess(SKImageProcess):
    """
    """
    __name__ = 'CompressSkipsProcess'
    __sequence_id__ = 10

    def __init__(self):
        super().__init__()
        
        # this process is independent of the amplifier
        self._per_amp = False

        # parameters of the model
        self.func_to_compress = ['mean','std'] 
        # plot equalized image
        self.histequ = False

        # To include the evaluation of possible cross-talk during the execute_process
        self.xtalk = False
        self.saturation = True
        # number of sigmas to evaluate posible cross-talk between amplifiers
        self.n_sig_xtalk = 5
        
        ### TO SAVE IMAGES AS ROOT FILES
        self.create_ttree = False

        self.__units__.update({'func_to_compress':1,'histequ':1,'n_sig_xtalk':1,'create_ttree':1,'xtalk':1, 'saturation':1})
    

    def evaluate_xtalk(self,rawdata):
        """Identify pixels with xtalk between both amplifiers

        """
        if rawdata.n_amp==1:
            print(f" WARNING: Data was taking with only one amplifier {rawdata.n_amp}, cross-talk can not be evaluated \n")
            return

        hstd = ROOT.TH1
        if not hasattr(rawdata,"image_std_compressed"):
            self.func_to_compress.append( "std")
            self.execute_process()
            return
        
        # FIT NDCM's STD to a guassian profile (each amplifier may have a different mu and sigma values)
        hstd = []
        mask = []
        for amp in sorted(rawdata.amplifier.keys()):
            rawdata.set_amplifier(amp)
            image = rawdata.get_image_by_attribute_name("std_compressed")
            
            cimg = image.compressed()
            limits = np.percentile(cimg,(5,95))
            qmean = stats.tmean(cimg,limits)
            qstd  = stats.tstd(cimg,limits)
            sample = cimg[np.logical_and(cimg> qmean-3*qstd, cimg<qmean+3*qstd)]
            
            #mu,emu, sig,esig, chi2 
            mu,emu,sig,esig,chi2, hamp = self.fit_gaussian(sample, **{'Nbins':"sturge","title":f"skip_std_{amp}","get_histogram":True})
            hamp.SetName(f"h_skip_std_{amp}")
            hamp.SetTitle(f"h_skip_std_{amp}")
            hamp.GetXaxis().SetTitle("std(pixel skips)")
            hstd.append( hamp )
            
            # detect possible x-talk pixels (possitive or negative)
            print(f"    - XTALK {amp}: Search for CROSS-Talk using: abs(std - trim_mean(std)) > {self.n_sig_xtalk} * {np.round(qstd,2)}")
            mask.append( np.abs(image-qmean) > self.n_sig_xtalk * qstd )
        
        # SEARCH FOR XTALK: STD in both sides is larger than mean + N std 
        # This situation may also happen, when the image is super crouded (full of energetic clusters, as the STD is larger)
        mL = mask[0][:, rawdata.n_cols_prescan:image.shape[1]//2-rawdata.n_cols_overscan] 
        mU = np.flip(mask[1], axis=1)[:, rawdata.n_cols_prescan:image.shape[1]//2-rawdata.n_cols_overscan]
        mask = np.logical_and(mL,mU)
        print(f"    - number of possible xtalk pixels: {mask.sum()}, with coordenates (shifted by {rawdata.n_cols_prescan} in columns)")
        print( np.where( mask ))

        setattr(rawdata,"image_xtalk", np.zeros_like(image.data))
        rawdata.image_xtalk[:, rawdata.n_cols_prescan:image.shape[1]//2-rawdata.n_cols_overscan] = mask
        rawdata.image_xtalk = np.flip(rawdata.image_xtalk, axis=1)
        rawdata.image_xtalk[:, rawdata.n_cols_prescan:image.shape[1]//2-rawdata.n_cols_overscan] = mask

        return hstd
   

    def execute_process(self,rawdata):
        """
        """
        if not self.__silent__:
            print("Process <CompressSkipsProcess> INFO. Compressing raw image data using the following statistics: {}".format(self.func_to_compress))
        
        if rawdata.nskips == 1:
            print("  WARNING. Number of skips = 1, nothing to do!")
            return

        ### func_to_compress can be semicolon separated list of function names
        if not type(self.func_to_compress) is list:
            self.func_to_compress = self.func_to_compress.split(";")

        #### SET THE USER REGION FOR EACH AXIS: row, col and skips
        ################################################################################
        user_rows_range = self.get_user_axis_range(rawdata,'row')
        user_cols_range = self.get_user_axis_range(rawdata,'col')
        user_skip_range = self.get_user_axis_range(rawdata,'skip')

        ### USE ONLY THE SKIP RANGE DEFINED FOR THE USER
        ### Compress the skipper images using the user functions, 
        ###     and create a new image for each used function
        for func in self.func_to_compress:
            ## Function to comproess the skipper image is assumed to exists in the numpy package
            try:
                np_func = getattr(np,func)
            except AttributeError:
                msm = "<CompressSkipsProcess>  ERROR. module 'numpy' has no attribute '{}'\n".format(func)
                msm += "    try to use another function for 'func_to_compress'"
                raise AttributeError(msm)
            ## add this image as a data member in the rawdata (some other process can look for)
            setattr(rawdata, "image_{}_compressed".format(func), np_func(rawdata.image[user_rows_range,user_cols_range,user_skip_range] ,axis=2))
       
        ### set the data member compressed to easier check if the data was already compressed (useful from other processes)
        rawdata.compressed = True
        
        # UPDATE HEADER ACCORDINGLY
        rawdata.image_header[rawdata.nskips_keyword] = 1
        rawdata.image_header['Nskips'] = (rawdata.nskips, 'DQM: NDCMS from raw image')
        rawdata.image_header['NSTART'] = (user_skip_range.start+1,'DQM: Fist NDCMS used for image averaging')
        rawdata.image_header['NEND']   = (user_skip_range.stop, 'DQM: Last NDCMS used for image averaging')
        rawdata.image_header['NUSED']  = (user_skip_range.stop-user_skip_range.start, "DQM: Total NDCMS used ")
        
        if self.xtalk:
            hstd = self.evaluate_xtalk(rawdata)
        
        if self.saturation:
            # DEFINITION OF SATURATED PIXELS: skip measurement will be at his maximum value and STD will be 0
            if "std" in self.func_to_compress and "mean" in self.func_to_compress:
                setattr(rawdata,"image_saturated_pixels", np.logical_and(
                                            rawdata.image_mean_compressed == rawdata.image_mean_compressed.max(),  
                                            rawdata.image_std_compressed == 0).astype(int)) 

        setattr(rawdata,"func_to_compress",self.func_to_compress)
        if not rawdata.__process_chain__.split("/")[-1] == self.__name__:
            rawdata.__process_chain__+="/"+self.__name__

        if not self.__silent__:
            print("     - used rows range {}:{}".format(user_rows_range.start,user_rows_range.stop))
            print("     - used cols range {}:{}".format(user_cols_range.start,user_cols_range.stop))
            print("     - used skip range {}:{}\n".format(user_skip_range.start,user_skip_range.stop))
        
        if self.create_ttree:
            self.save_images_as_root(rawdata)

        return


#################################################################################################
#                                                                                               #
#       Process Class to Evaluate the CROSS-TALK BETWEEN AMPLIFIERS                             #
#           process that should be used if image is not an skip image,                          #
#           and STD comes in another fits/fz file (ACM or MOSKITA too)                          #
#################################################################################################

class CrossTalkProcess(SKImageProcess):
    """
    """
    __name__ = 'CrossTalkProcess'
    __sequence_id__ = 11

    def __init__(self):
        super().__init__()
        
        self.__DEBUG__ = False
        self.__silent__ = True
        
        self.percentile = [5,95]

        # this process is independent of the amplifier as evaluates the XT between them
        self._per_amp = False

        # number of sigmas to evaluate posible cross-talk between amplifiers
        self.n_sig = 5
        self.std_pattern = ["avg_skips_contrun","var_skips_contrun"]
        self.std_file = None

        self.__units__.update({'n_sig': 1,'std_pattern':1,'percentile':1})

    def execute_process(self,rawdata):
        """Identify pixels with xtalk between both amplifiers
            
            --- IDENTIFY CROSS-TALK AND TRANSIENTS FROM VIDEO SIGNAL ---
            El cross-talk es una señal que se da en otro canal por tener señal en
            un canal, pero muchas veces no es que "pasa" señal de un canal al otro
            (como seria un cross-talk optico por ejemplo en un telescopio). Aqui
            son capacitancias parasitas creo o cosas por el estilo (no se nada de
            electronica) que producen señal por induccion de alguna manera en las
            pistas vecinas... O un pulso grande hace que el cero baja un poco para
            compensar y entonces uno ve señales negativas pero solo porque en
            realidad el cero bajo (solo cuando se mide la carga, con lo cual no se            
            soluciona con el skip).
            Y si, esos ruidos que tenemos no son realmente crosstalk, o por lo
            menos no en el sentido usual de la palabra ya que usualmente se
            entiende por crosstalk un ruido inducido por una señal en otro canal
            del detector. Si es un ruido externo, como que se prende el
            cryocooler, o otro CCD, o el aire condicionado, no se lo suele llamar
            crosstalk. Si pudiesemos estimar a la perfeccion el timing de cada CCD
            respecto uno al otro y ver que ese ruido se produce por ejemplo a cada
            TG de otro CCD, entonces si podriamos llamarlo mas facilmente un
            crosstalk porque tendriamos una "señal" (aun si no es la señal video
            usual) que se traduce en un transient en la señal video... En fin...
        """

        print(f"Process <{self.__name__}> INFO. Evaluate the 'positive cross-talk' between amplifiers: q > qmean + n*sigma (true in all amp)")

        hstd = ROOT.TH1
        if not hasattr(rawdata,"image_std_compressed"):
            std_file = rawdata.file_name.replace(self.std_pattern[0],self.std_pattern[1])
            if rawdata.ACM:
                std_image = fits.getdata(std_file,rawdata.__extensions__[0]).astype(float)
                for ext in rawdata.__extensions__[1:]:
                    std_image = np.concatenate((std_image,fits.getdata(std_file,ext).astype(float)),axis=0) 
            setattr(rawdata,"image_std_compressed",std_image.copy())

        
        # FIT NDCM's STD to a guassian profile (each amplifier may have a different mu and sigma values)
        hstd = []
        mask = []
        for amp in sorted(rawdata.amplifier.keys()):
            rawdata.set_amplifier(amp)
            image = rawdata.get_image_by_attribute_name("std_compressed")

            cimg = image.compressed()
            print(f"    - XTALK {amp}: range of std values: {np.round(cimg.min(),2)}-{np.round(cimg.max(),2)}")
            limits = np.percentile(cimg,self.percentile)
            qmean = stats.tmean(cimg,limits)
            qstd  = stats.tstd(cimg,limits)
            if not self.__silent__:
                print(f"                   mean={qmean}, std={np.round(qstd,2)}, threshold={np.round(qmean+3*qstd)}")
            sample = cimg[np.logical_and(cimg> qmean-3*qstd, cimg<qmean+3*qstd)]
            
            #mu,emu, sig,esig, chi2 
            mu,emu,sig,esig,chi2, hamp = self.fit_gaussian(sample, **{'Nbins':"sturge","title":f"skip_std_{amp}","get_histogram":True})
            hamp.SetName(f"h_skip_std_{amp}")
            hamp.SetTitle(f"h_skip_std_{amp}")
            hamp.GetXaxis().SetTitle("std(pixel skips)")
            hstd.append( hamp )
            
            # detect possible x-talk pixels (possitive or negative)
            if not self.__silent__:
                print(f"                   Search for CROSS-Talk using: abs(std - trim_mean(std)) > {self.n_sig} * {np.round(qstd,2)}")
            m_amp = np.abs(image-qmean) > self.n_sig * qstd 
            m_amp.mask = False
            if not self.__silent__:
                print(f"                   {np.sum(m_amp)/(m_amp.shape[0]*m_amp.shape[1])*100} pixels above STD threshold")
            # select region non-masked, i.e. amp region
            r_start,r_end = rawdata.amplifier[amp]['amp_image_shape']['rows']
            c_start,c_end = rawdata.amplifier[amp]['amp_image_shape']['cols']
            mask.append(m_amp[r_start:r_end,c_start:c_end])

        # SEARCH FOR XTALK: STD in both sides is larger than mean + N std 
        # This situation may also happen, when the image is super crouded (full of energetic clusters, as the STD is larger)        
        mask = np.logical_and(mask[0],mask[1])
        for m in mask[2:]:
            mask = np.logical_and(mask, m)

        print(f"    - number of possible xtalk pixels: {mask.sum()}")
        if not self.__silent__:
            print(f"        with coordenates")
            print( np.where( mask ))
        #from matplotlib import pyplot as plt
        #plt.imshow(mask, origin='lower', aspect='auto')
        #plt.show(block=True)
        #input("enter")

        setattr(rawdata,"image_xtalk", np.repeat(mask,len(rawdata.amplifier.keys()),axis=0))

        return
   


#################################################################################################
#                                                                                               #
#       Process Class to compress an skipper image by reducing the skips measurments            #
#           according the user function (or list of functions                                   #
#                                                                                               #
#################################################################################################
class PedestalSubtractionProcess(SKImageProcess):
    """Image Equalization is the process of subtracting the baseline or pedestal from an image

    The baseline or pedestal is the ADU values for a pixel with zero exposure time.

    """
    __sequence_id__ = 20
    __name__ = 'PedestalSubtractionProcess'
    
    ### The baseline (pedestal) can be extimated in rows, columns, in rows and then columns, or
    #       using the full image (independenly of rows and columns)
    def __init__(self):
        """
        """
        super().__init__()

        self.image = "mean_compressed"

        ### methods: gaussian_fit, mean
        ###     - in both cases, the user can choose if the pedestal estimation is done by rows,
        ##       columns, or independently
        #       - the user can also use only an specific region: overscan, or sensitive region      
        #   possible methods are: gauss_fit/median/mean
        self.method = "gauss_fit"
        # to select which pixels to use for the fitting: q in [median(q)+3 MAD, median(q)-3 MAD]
        #   use MAD or MAD as un estimator of the standard deviation
        self.use_mad    = True

        self.image = "mean_compressed"
        self.in_overscan = True
        self.axis = "row"
        
        ### pedestal subtraction skip image by skip image
        self.in_rawskipimage = False
        
        ### for the gauss_fit method
        self.n_sigma_win_fit = 10.0
        
        ### list of cols and rwos to skip
        self.skip_cols = []
        self.skip_rows = []

        ### when image mask be masked
        self.n_sigma_to_mask = -1
        ### for images with a lot of clusters
        self.cut_left_tail = True
        self.cut_right_tail = True

        ### method to calculate the number of bins for the PCD histogram: sturge | freedman | default | manual
        self.hist_nbins_method = "default" 
        self.hist_nbins = 0
        
        self.histequ = False
        self.th1_guas_fits = []
        self.save_pcd     = True
        
        ### TO SAVE IMAGES AS ROOT FILES
        self._create_ttree_counter = 0
        self.create_ttree = False

        self.is_highDC = False 
        self.highT     = False
        self.is_highDC_Xrange = None

        self.DEBUG_ROWS = None
        self.rebin_rows = 1
        ### 
        self.__units__.update({'method':1,'in_overscan':1,'in_rawskipimage':1, 'axis':1,'n_sigma_win_fit':1,'n_sigma_to_mask':1,'histequ':1,'show_fit':1,
            'use_mad':1, 'save_pcd':1 , 'skip_cols':1, 'skip_rows':1 , "hist_nbins_method":1, "hist_nbins":1,
            "create_ttree":1 , 'is_highDC':1, 'is_highDC_Xrange':1,
            'cut_left_tail':1,'cut_right_tail':1, "DEBUG_ROWS":1, "rebin_rows":1})
    
    def execute_process(self,rawdata):
        """
        """
        if not self.method.lower() in ['gauss_fit','median','mean','gaussian_fit','gaus_fit']:
            raise AttributeError('<PedestalSubtractionProcess> ERROR: Available methods for pedestal subtractio are: gauss_fit, median or mean')
        if self.method in ['gauss_fit','gaussian_fit','gaus_fit']:
            self.method = 'gaus_fit'
        if not self.hist_nbins_method.lower() in ["default","freedman","sturge","manual"]:
            raise AttributeError(f"<PedestalSubtractionProcess> ERROR: Available methods to calculate the number of bins for the histogram are: sturge, freedman, default or manual. If manual, hist_nbins should be given.")
        
        if self.hist_nbins_method.lower()=="manual":
            if self.hist_nbins<=0:
                raise AttributeError(f"<PedestalSubtractionProcess> ERROR: For 'manual' option, the number of bins should be informed via 'hist_bin' option.")

        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)

        print("Process <PedestalSubtractionProcess> INFO. Equalize the image by substracting the pedestal {}.".format(amplifier))
        ### set properly the use of is_highDC algorithm to find the center of the gaussian in case DC is quite large
        if type(self.is_highDC)==dict:
            self.highT = self.is_highDC[rawdata.execute_process_in_amp]
        elif self.is_highDC:
            self.highT = self.is_highDC

        ## check if pedestal should be estimated in the overscan, and ovescan has at least 3 columns
        if self.in_overscan  and rawdata.n_cols_overscan<3:
            self.in_overscan = False
            # mask clusters
            if self.n_sigma_to_mask < 0:
                self.n_sigma_to_mask = 100

        #### GET IMAGE TO ESTIMATE THE PEDESTAL
        ################################################################################
        if self.in_rawskipimage:
            image = rawdata.get_skip_image_and_broadcasted_mask()
            print(f"   - skip image by skip image: {image.shape}")
       
        else:
            itype = rawdata.get_image_attribute_name(self.image)
            image_amp = rawdata.get_image_by_attribute_name(self.image)
        
            #### SET THE REGION TO BE USED TO ESTIMATE THE PEDESTAL: overscan rows? or full?
            ################################################################################

            if self.in_overscan:
                print("   - using overscan region")
                image = np.ma.array(image_amp,mask=rawdata.mask_image_overscan_cols)
            else:
                print("   - using the sensitive region")
                if self.axis == "col":
                    image = np.ma.array(image_amp,mask=np.zeros_like(image_amp))
                else:
                    image = np.ma.array(image_amp,mask=rawdata.mask_image_active_region)


                #### skip cols and rows only if possible
                for col in self.skip_cols:
                    print(f"   - skipping columns {col}")
                    if type(col)==list:
                        image.mask[:,col[0]:col[1]+1] = True
                    else:
                        image.mask[:,col] = True
                for row in self.skip_rows:
                    print(f"   - skipping rows {row}")
                    if type(row)==list:
                        image.mask[row[0]:row[1]+1,:] = True
                    else:
                        image.mask[row,:] = True

        if self.highT:
            if type(self.is_highDC_Xrange) == list:
                low, high = self.is_highDC_Xrange
                new_mask = np.logical_or((image.data < low) , (image.data > high))
                image = np.ma.array(
                            image.data, 
                            mask=np.logical_or(image.mask,  new_mask)
                            )


        ### MASK DATA
        # when the image to be used for pedestal subtraction if the full image, as well as
        # when data is only overscan. In the later, if the data is taken in the surface (not underground), 
        # the probability of a particle interacting with the serial register while reading
        # the overscan region is not negligigle. To prevent the algorithm to include pixels with 
        # signal from ionizing particles is to mask those pixels above a minimum value
        if self.n_sigma_to_mask>0:
            mad   = np.ma.median(np.ma.abs(image-np.ma.median(image)))
            if self.cut_left_tail and self.cut_right_tail:
                image = np.ma.array(image, mask = np.logical_or( (image-np.ma.median(image)) >  1.0*self.n_sigma_to_mask*mad,
                    (image-np.ma.median(image)) < -1.0*self.n_sigma_to_mask*mad))
            elif self.cut_left_tail and not self.cut_right_tail:
                image = np.ma.array(image, mask = (image-np.ma.median(image)) < -1.0*self.n_sigma_to_mask*mad )
            elif self.cut_right_tail and not self.cut_left_tail:
                image = np.ma.array(image, mask =  (image-np.ma.median(image)) >  1.0*self.n_sigma_to_mask*mad )

            print("   - masking image: mask all pixels with qij > {} + {}x{}".format(np.ma.median(image),self.n_sigma_to_mask,mad))

            if self.__DEBUG__ and self.__verbose__:
                from matplotlib import pyplot as plt
                plt.figure()
                plt.imshow(image.mask, aspect='auto', cmap='jet')
                plt.colorbar()
                plt.show()
                input("Does the mask meet your expectations? If not, play with <n_sigma_to_mask> [press enter]")

        #### THE USER CAN ALSO DECIDE IN WHICH AXIS ESTIMATE THE PEDESTAL: 
        ####        row/col/both/any other
        ################################################################################
        #### Set the axis[or axes] to estimate the pedestal
        try:
            axis = self.__axis_id__[self.axis]
            if type(axis) is not list:
                axis = [axis]
        except KeyError:
            ### any other will use the full selected region
            axis = None
        
        #### SEVERAL METHODS are allowed to estimate the pedestal: mean/median or gaussian fit
        ####        gaussian fit to the zero-charged peak or a simple median 
        ################################################################################
        ###     the gaussian fit method is a method of the parent class (find it at absp) 
        if self.method in ["gaus_fit"]:
            print("   - pedestal as the mu of a the fitted gaussian to the zero-charged peak (axis {})".format(self.axis))
            #func_to_estimate_pedestal = self.fit_gaussian_peak
            # which returns # mu, error mu, sigma, error sigma, chi2
            func_to_estimate_pedestal = self.fit_gaussian

        elif self.method == "median":
            print("   - pedestal as the median of a [masked] image (axis {})".format(self.axis))
            def func_to_estimate_pedestal(x,nsig=None,axis=None,**kwargs):
                mu,sigma = self.get_med_and_mad(x,mad=self.use_mad)
                # mu, error mu, sigma, error sigma, chi2
                return mu,0.0,sigma,0.0,1.0

        elif self.method == "mean":
            print("   - pedestal as the mean of a [masked] image (axis {})".format(self.axis))
            def func_to_estimate_pedestal(x,nsig=None,axis=None,**kwargs):
                mu,sigma = self.get_med_and_mad(x,mad=self.use_mad,median=False)
                # mu, error mu, sigma, error sigma, chi2
                return mu,0.0,sigma,0.0,1.0
        else:
            raise KeyError("<PedestalSubtractionProcess>: method is not implemented: available 'gaussian_fit', 'median' or 'mean'")
        
        ##########################################################################################################
        ##### PROCESS EXECUTION STARTS HERE
        ##########################################################################################################
        if self.in_rawskipimage:
            self.pedestal_subtraction_on_skipper_image(rawdata, image, axis, func_to_estimate_pedestal, amplifier)
        else:
            #### algorithm to subtract the pedestal subtraction of the compressed image
            self.pedestal_subtraction_on_compressed_image(rawdata, image, image_amp, axis, func_to_estimate_pedestal, amplifier, itype)
        
        return    

    def pedestal_subtraction_on_skipper_image(self, rawdata, image_skip, axis, func_to_estimate_pedestal, amplifier):        
        mu, sigma, emu, esigma, chi2 = {},{},{},{},{}
        x_mask_values = {}

        __verbose__ = self.__verbose__
        self.__verbose__ = False
        #### full size of the image
        full_image_slice = [slice(0,image_skip.shape[0]), slice(0,image_skip.shape[1])]

        ### the function can be apply in one axis, and then in the other axis
        #       for instance estimate the pedestal for each row, subtract the pedestal,
        #           and then estimate the pedestal for each column to be also subtracted
        #           some images has patterns depending on both axis
        self._row_debug = True if (self.__DEBUG__ and self.DEBUG_ROWS is not None) else False
        
        if len(axis)>1 or self.__axis_name__[axis[0]] != 'row':
            raise AttributeError(f"Pedestal subtraction at the skip image level is supported for axis=row only!")
        
        #### image to subtract the baseline
        image_skip_sub = image_skip.data.copy()
        mu_limits = [np.inf,-np.inf]
        for ax in range(image_skip.shape[2]):
            print(ax)

            image = image_skip[full_image_slice[0],full_image_slice[1],ax]
            ### Estimate the pedestal for each axis
            mu[ax]     = np.zeros(image.shape[0])
            emu[ax]    = np.zeros(image.shape[0])
            sigma[ax]  = np.zeros(image.shape[0])
            esigma[ax] = np.zeros(image.shape[0])
            chi2[ax]   = np.zeros(image.shape[0])
            x_mask_values[ax] = []
            bad_fit    = []
            
            ### On a single skip image, estimate the baseline and subtract it roow by row
            for k in range(image.shape[0]):
                if self._row_debug:
                    self.__DEBUG__ = True if k in list(range(self.DEBUG_ROWS[0],self.DEBUG_ROWS[1]+1)) else False
                
                #### FIXME? row by row, not allowed in column axis!!
                img_ax = image[ k, full_image_slice[1]]
                
                if img_ax.compressed().size <2:
                    #### note that some of the cols or rows may be fully masked
                    #       in this case add nan
                    mu[ax][k] = sigma[ax][k] = np.nan
                    continue

                ### ESTIMATE PEDESTAL: mu and sigma 
                ###     apply function (gauss fit or median) to the selected pixel list (img_ax)
                mu[ax][k],emu[ax][k],sigma[ax][k],esigma[ax][k],chi2[ax][k] = func_to_estimate_pedestal(img_ax,nsig=self.n_sigma_win_fit,
                        axis=ax,**{'title':'{},{}'.format(ax,k)},
                        **{"Nbins":self.hist_nbins_method.lower(), "Nbins_num":int(self.hist_nbins)})
                
                if self.exit<0:
                    ### keep track of those unsuccessful groups of pixels 
                    bad_fit.append(k)
                ### add the column or row that has been included
                x_mask_values[ax].append(k)
                
                ### subtract the pedestal from the n-essim skip image
                image_skip_sub[:,:,ax] = image_skip_sub[:,:,ax] - mu[ax].astype(np.float64).reshape(len(mu[ax]),1)

            mu_limits[0] = min(mu_limits[0], min(mu[ax]))
            mu_limits[1] = max(mu_limits[1], max(mu[ax]))
            
            # set __DEBUG__ to user settings
            if self._row_debug:
                self.__DEBUG__ = True           
            self.__verbose__ = __verbose__
        
        setattr(rawdata,"image_mean_compressed_pedestal_subtracted", np.mean(image_skip_sub,axis=2))

        #### PLOTTING OUTPUTS for the mu and sigma
        if self.__display__:
            _isbatch = ROOT.gROOT.IsBatch()
            if self.__display__:
                ROOT.gROOT.SetBatch(0)
            else:
                ROOT.gROOT.SetBatch(1)
            
            n=0
            tg, c = {}, {}
            indx = 0
            for pstr, param in zip(['Baseline (ADUs)','Sigma (ADUs)'],[mu,sigma]):
                c[pstr] = ROOT.TCanvas(amplifier+pstr)
                tg[pstr] = []
                optdraw = "AP PMC PLC"
                for ax in range(image_skip.shape[2]):
                    tg[pstr].append( ROOT.TGraph( len(x_mask_values[ax]), array('d',x_mask_values[ax]), array('d',param[ax][x_mask_values[ax]] )))
                    tg_name = pstr.replace('_','').replace('(','_').replace(')','')+f"{ax}"
                    tg[pstr][-1].SetTitle(f"{tg_name};{self.__axis_name__[0]};{pstr}")
                    tg[pstr][-1].SetName(tg_name)
                    if indx==0:
                        tg[pstr][-1].GetHistogram().GetYaxis().SetRangeUser(mu_limits[0]-mu_limits[0]*0.9,mu_limits[1]+10*mu_limits[1])
                    tg[pstr][-1].Draw(optdraw)
                    optdraw = "PL PMC PLC SAME"
                
                
                # Display only if --display option is set to ON
                c[pstr].Update()
            input("press enter ...")

            ROOT.gROOT.GetListOfCanvases().Delete()
            ROOT.gROOT.SetBatch(_isbatch)
         
        return

    def pedestal_subtraction_on_compressed_image(self,rawdata, image, image_amp, axis, func_to_estimate_pedestal, amplifier, itype):
        
        mu, sigma, emu, esigma, chi2 = {},{},{},{},{}
        x_mask_values = {}
        if axis is None:
            mu0,sigma0 = func_to_estimate_pedestal(image)
            print("   - pedestal estimation is {} and its sigma {} (in ADUs)".format(mu0,sigma0))
            mu[0] = mu0
            sigma[0]=sigma0

        else:
            __verbose__ = self.__verbose__
            self.__verbose__ = False
            #### full size of the image
            full_image_slice = [slice(0,image.shape[0]), slice(0,image.shape[1])]
            ### the function can be apply in one axis, and then in the other axis
            #       for instance estimate the pedestal for each row, subtract the pedestal,
            #           and then estimate the pedestal for each column to be also subtracted
            #           some images has patterns depending on both axis
            self._row_debug = True if (self.__DEBUG__ and self.DEBUG_ROWS is not None) else False
            for ax in axis:
                # initialize the histogram for each axis
                self.th1_guas_fits = []

                print("   - Subtracting PEDESTAL along axis ", 'row' if ax==0 else 'column')
                ### Estimate the pedestal for each axis
                mu[ax]     = np.zeros(image.shape[ax])
                emu[ax]    = np.zeros(image.shape[ax])
                sigma[ax]  = np.zeros(image.shape[ax])
                esigma[ax] = np.zeros(image.shape[ax])
                chi2[ax]   = np.zeros(image.shape[ax])
                x_mask_values[ax] = []
                bad_fit    = []
                
                if ax==1:
                    # subtraction along columns, must be along the full image, pre + active + col
                    if self.in_overscan:
                        print("     WARNING: Subtraction in columns can not be done over the overscan region only, should be done to all columns: pre+act+ovs")
                        image = np.ma.array(image.data, mask=np.zeros_like(image.data))
                        print("         Mask set to False for all pixels")
                slice_ax  = full_image_slice.copy()

                for k in range(0,image.shape[ax],1):
                    if self._row_debug:
                        if k in list(range(self.DEBUG_ROWS[0],self.DEBUG_ROWS[1]+1)):
                            self.__DEBUG__ = True
                        else:
                            self.__DEBUG__ = False

                    ##### estimate the pedestal for each index in the axis
                    #slice_ax[ax]=k
                    ##### select those pixels in the region of interest: [row_k,:] or [:,col_k]
                    #img_ax = image[slice_ax[0],slice_ax[1]]
                    if k+self.rebin_rows/2 > image.shape[ax]:
                        slice_ax[ax] = slice(image.shape[ax]-int(self.rebin_rows/2), image.shape[ax])
                    else:
                        slice_ax[ax] = slice(k, min(k+self.rebin_rows, image.shape[ax]))

                    img_ax = image[slice_ax[0], slice_ax[1]]

                    if img_ax.compressed().size <2:
                        #### note that some of the cols or rows may be fully masked
                        #       in this case add nan
                        mu[ax][k:k+self.rebin_rows]    = np.nan 
                        sigma[ax][k:k+self.rebin_rows] = np.nan
                        continue

                    ### ESTIMATE PEDESTAL: mu and sigma 
                    ###     apply function (gauss fit or median) to the selected pixel list (img_ax)
                    #mu[ax][k],emu[ax][k],sigma[ax][k],esigma[ax][k],chi2[ax][k] 
                    mu_val, emu_val, sigma_val, esigma_val, chi2_val = func_to_estimate_pedestal(img_ax,nsig=self.n_sigma_win_fit,
                                                                            axis=ax,**{'title':'{},{}'.format(ax,k)},
                                                                            **{"Nbins":self.hist_nbins_method.lower(), "Nbins_num":int(self.hist_nbins)})
                    mu[ax][k:k+self.rebin_rows]     = mu_val
                    emu[ax][k:k+self.rebin_rows]    = emu_val
                    sigma[ax][k:k+self.rebin_rows]  = sigma_val
                    esigma[ax][k:k+self.rebin_rows] = esigma_val
                    chi2[ax][k:k+self.rebin_rows]   = chi2_val

                    if self.exit<0:
                        ### keep track of those unsuccessful groups of pixels 
                        for ki in range(k,min(k+self.rebin_rows,image.shape[ax]),1):
                            bad_fit.append(ki)

                    ### add the column or row that has been included
                    for ki in range(k,min(k+self.rebin_rows,image.shape[ax]),1):
                        x_mask_values[ax].append(ki)
                
                # set __DEBUG__ to user settings
                if self._row_debug:
                    self.__DEBUG__ = True

                ##### reaplace nan values, if any, for its mean value
                #mu[ax]     = np.nan_to_num(mu[ax],     nan= np.nanmean(mu[ax]))
                #sigma[ax]  = np.nan_to_num(sigma[ax],  nan= np.nanmean(sigma[ax]))
                
                ### recreate root file before updating it
                if self.save_plots and "root" in self.format_figures:
                    outrootfile = ROOT.TFile.Open(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}_all_params.root","RECREATE")
                    outrootfile.Close()

                self.__verbose__ = __verbose__
                #### PLOTTING OUTPUTS for the mu and sigma
                if self.save_plots or self.__display__:
                    _isbatch = ROOT.gROOT.IsBatch()
                    if self.__display__:
                        ROOT.gROOT.SetBatch(0)
                    else:
                        ROOT.gROOT.SetBatch(1)
                    n=0
                    c,tg = [],[]
                    for pstr, param in zip(['Baseline (ADUs)','Sigma (ADUs)'],[mu,sigma]):
                        c.append(ROOT.TCanvas(f"{amplifier}-{pstr}",f"{amplifier}-{pstr}"))
                        c[-1].cd()
                        tg.append( ROOT.TGraph( len(x_mask_values[ax]), array('d',x_mask_values[ax]), array('d',param[ax][x_mask_values[ax]]) ) )
                        tg[-1].SetTitle(f"amplifier {rawdata.execute_process_in_amp};{self.__axis_name__[ax]};{pstr}")
                        tg[-1].SetName(pstr.lower().replace(" ","_").replace("(","").replace(")",""))
                        tg[-1].Draw("A P")
                        c[-1].Update()
                        c[-1].Draw()

                        if self.save_plots:
                            for fmt in self.format_figures:
                                if fmt=="root":
                                    outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}_all_params.root","UPDATE")
                                    tg[-1].Write()
                                    outrootfile.Close()
                                else:
                                    fname = rawdata.output+"{}_{}_{}.{}".format(self.__name__,pstr.split(' ')[0],rawdata.execute_process_in_amp,fmt)
                                    c[-1].SaveAs(fname)

                    if self.__display__:
                        # Display only if --display option is set to ON
                        input("Baseline and sigma where display,\npress enter ...")

                ### set shape for mu and sigma to be able to subtract them from the image
                if ax==0:
                    mu[ax]    = np.array(mu[ax]).astype(np.float64).reshape(len(mu[ax]),1)
                    emu[ax]   = np.array(emu[ax]).astype(np.float64).reshape(len(emu[ax]),1)
                    sigma[ax] = np.array(sigma[ax]).astype(np.float64).reshape(len(sigma[ax]),1)
                    esigma[ax]= np.array(esigma[ax]).astype(np.float64).reshape(len(esigma[ax]),1)
                else:
                    mu[ax]    = np.array(mu[ax]).astype(np.float64).reshape(1,len(mu[ax]))
                    emu[ax]   = np.array(emu[ax]).astype(np.float64).reshape(1,len(emu[ax]))
                    sigma[ax] = np.array(sigma[ax]).astype(np.float64).reshape(1,len(sigma[ax]))
                    esigma[ax]= np.array(esigma[ax]).astype(np.float64).reshape(1,len(esigma[ax]))
                
                #### If both axis are booked, the second round must be applied on the first-axis-pedestal subtracted image
                if len(axis)>1:
                    image = image - mu[ax]
        
        ############# OUTPUTS
        ######################################################################################################
        results = {'pedestal_mu':mu, 'pedestal_sigma':sigma }
        if self.method in ['gaus_fit']:
            # append errors and chi2, only with this fit method has sense the error
            results.update({ 'pedestal_mu_error':emu, 'pedestal_sigma_error':esigma, 'pedestal_chi2': chi2 })

        # pedestal subtracted image will be save as
        img_ped_sub_name = f"{itype}_pedestal_subtracted"
        _ = rawdata.get_pedestal_substracted_image(image_amp,mu,img_ped_sub_name)
        rawdata.set_image_by_attribute_name(img_ped_sub_name,None,**results)
        
        if self.__display__:
            ROOT.gStyle.SetLegendTextSize(0.05)
            psimage = getattr(rawdata,img_ped_sub_name)
            M,N = psimage.shape

            ## in case high-E pixels are present, remove tails
            qmean   = trim_mean(psimage,0.1)
            qstd    = trimmed_std(psimage,0.1)
            
            c,tg,leg = [],[],[]
            for ax in [0,1]:
                ### columns
                N = psimage.shape[int(not ax)]
                x = np.linspace(0,N-1,N)
                xlabel = 'column' if ax==0 else 'row'
                ymedian = np.median(psimage,axis=ax)
                ymean   = np.ma.mean(np.ma.array(psimage, mask= np.abs(psimage-qmean)>5.0*qstd),axis=ax)

                c.append(ROOT.TCanvas())
                leg.append(ROOT.TLegend(0.65,0.3,0.75,0.5))

                c[-1].cd()
                tg.append( ROOT.TGraph(len(x),array('d',x),array('d',ymedian)) )
                tg[-1].SetTitle(f"MedianChargePerColumn;{xlabel};pixel charge")
                tg[-1].SetMarkerStyle(21)
                tg[-1].Draw("AP")
                leg[-1].AddEntry(tg[-1],'median','P')

                tg.append( ROOT.TGraph(len(x),array('d',x),array('d',ymean)) )
                tg[-1].SetTitle(f"MedianChargePerColumn;{xlabel};pixel charge")
                tg[-1].SetMarkerStyle(20)
                tg[-1].SetMarkerColor(ROOT.kRed+1)
                tg[-1].Draw("P same")
                leg[-1].AddEntry(tg[-1],'mean','P')
                
                leg[-1].Draw("same")
                c[-1].Update()
                c[-1].Draw()

            input("press enter ...")
            ROOT.gROOT.GetListOfCanvases().Delete()
            ROOT.gROOT.SetBatch(_isbatch)

        # UPDATE HEADER ACCORDINGLY
        ######################################################################################################
        rawdata.image_header[rawdata.nskips_keyword] = 1
        rawdata.image_header['PSFUNC'] = (self.method, 'DQM. Func to estimate pedestal')
        rawdata.image_header['PSMAD']  = (int(self.use_mad), 'DQM. 1: MAD, 0:STD')
        rawdata.image_header['PSAXIS'] = (self.axis, 'DQM. Axis to estimate pedestal')
        rawdata.image_header['PSOVS']  = (int(self.in_overscan), 'DQM. 1: in overscan')
        rawdata.image_header['PSNSIG'] = (self.n_sigma_to_mask,'DQM. Nsigma to mask pixels')
        try:
            rawdata.image_header[f"PSSIG{rawdata.execute_process_in_amp}"]  = (np.mean(list(itertools.chain(*sigma.values())) ),'Mean sigma from the PED-SUB porcess')
            rawdata.image_header[f"PSESIG{rawdata.execute_process_in_amp}"] = (np.std(list(itertools.chain(*sigma.values())) ),'Mean sigma from the PED-SUB porcess')
        except (AttributeError, ValueError) as error:
            pass

        rawdata.__process_chain__+="/"+self.__name__
        print("\n") 
        
        # only once per image, not per amplifier
        self._create_ttree_counter +=1
        if self.create_ttree and self._create_ttree_counter==rawdata.n_amp:
            self.save_images_as_root(rawdata)
 
        return


####################################################################################################################
#
#
#   DE-CORRELATION 
#
#
####################################################################################################################
class TwoDGaussProcess(SKImageProcess):
    """De-correlate data taken with two amplifiers by fitting a 2D Gaussian

    """
    __sequence_id__ = 28
    __name__ = 'TwoDGaussProcess'

    def __init__(self):
        """
        """
        super().__init__()
        # this process is independent of the amplifier
        self._per_amp = False
        self.__silent__ = False
        
        self.image = "mean_compressed_pedestal_subtracted"

        self.q_min_ADU = -20
        self.q_max_ADU = 100
        self.bin_size  = 0.5

        self.col_start = 0
        self.row_start = 0

        self.do_fit = True
        self.do_decorrelation = True
        self.do_calibration = True

        # initial guess for the fitting parameters
        # ai, muX, muY, xo, yo, sigma_x, sigma_y, theta, xoff, yoff
        self.dc_L = 3e-3
        self.dc_L_range = (self.dc_L/10,self.dc_L*10)
        self.dc_U = 3e-3
        self.dc_U_range = (self.dc_U/10,self.dc_U*10)
        # calibration
        self.cal_L = 7
        self.cal_L_range = (self.cal_L-3,self.cal_L+3)
        self.cal_U = 7
        self.cal_U_range = (self.cal_U-3,self.cal_U+3)
        # sigma
        self.sig_L = 2
        self.sig_L_range = (self.sig_L-3,self.sig_L+3)
        self.sig_U = 2
        self.sig_U_range = (self.sig_U-3,self.sig_U+3)

        self.theta = np.deg2rad(45.0)
        self.theta_range = (-1,1)

        self.xoff = 0
        self.xoff_range = (-1,1)
        self.yoff = 0
        self.yoff_range = (-1,1)

        self.fit_opt = "Q R LEM"
        self.NpX = 2
        self.NpY = 2

        self.__display__ = False

        # mask any pixel that belongs to a cluster
        self.mask_clusters = True

        self.format_figures = ["pdf"]
        ### 
        self.__units__.update({
            "__display__":1,
            "do_calibration":1,"do_fit":1,
            "q_min_ADU":u.ADC, "q_max_ADU":u.ADC,"bin_size":u.ADC,"col_start":1, "row_start":1,
            "dc_L":1,"dc_U":1,"dc_L_range":1,"dc_U_range":1,
            "cal_L":1,"cal_U":1,"cal_L_range":1,"cal_U_range":1,
            "sig_L":1,"sig_U":1,"sig_L_range":1,"sig_U_range":1,
            "theta":np.deg2rad(1),"theta_range":np.deg2rad(1),"format_figures":1,
            "fit_opt":1, "mask_clusters":1, "NpX":1,"NpY":1
            })

    def execute_process(self,rawdata):
        """
        """
        if not rawdata.n_amp==2:
            raise IOError("Decorrelation only possible if data has been readout with two amplifiers")
        
        ####################################################################################################
        #
        # pars : [I0, theta, lambda_x, lambda_y, sig_x, sig_y, cal_x, cal_y, xoff, yoff ]
        #         0     1       2         3       4       5     6       7     8     9
        ####################################################################################################
        def twoD_gauss(x, pars):
            if not hasattr(twoD_gauss,'_cmpcode'):
                cppcode = """
                #include <cmath>

                double _twoD_gauss(double * x, double * pars)
                {
                    const double _x = x[0]-pars[8];
                    const double _y = x[1]-pars[9];
                
                    const double a = (std::pow(TMath::Cos(pars[1]),2)/(2*std::pow(pars[4],2)) + std::pow(TMath::Sin(pars[1]),2)/(2*std::pow(pars[5],2)));
                    const double b = (TMath::Sin(2*pars[1])/(4*std::pow(pars[5],2)) - TMath::Sin(2*pars[1])/(4*std::pow(pars[4],2)));
                    const double c = (std::pow(TMath::Sin(pars[1]),2)/(2*std::pow(pars[4],2)) + std::pow(TMath::Cos(pars[1]),2)/(2*std::pow(pars[5],2)));
                    
                    double G = 0.0;
                    for(int nx=pars[10]; nx <= pars[11]; ++nx)
                    {
                        for(int ny=pars[12]; ny <= pars[13]; ++ny)
                        {
                            double Axy = pars[0]*TMath::Poisson(nx,pars[2])*TMath::Poisson(ny,pars[3]);
                            G = G + Axy*TMath::Exp(-(a*std::pow((_x-nx*pars[6]),2)+2*b*(_x-nx*pars[6])*(_y-ny*pars[7])+c*std::pow((_y-ny*pars[7]),2)));
                        }
                    }
                
                    return G;
                }
                """
                _ = ROOT.gInterpreter.ProcessLine(cppcode)
                twoD_gauss._cmpcode = ROOT._twoD_gauss
                twoD_gauss.npars = 14
            return twoD_gauss._cmpcode(x,pars)


            
        #### PROCESS STARTS HERE
        amplifier=''
        if not self.__silent__:
            print(f"Process <{self.__name__}>: by fitting a 2D gaussian")
        
        image = rawdata.get_image_by_attribute_name(self.image).data
        # GET BOTH IMAGES: im1(L)(x2d), im2(U)(y2d)
        self.col_start += rawdata.n_cols_prescan
        L = image[self.row_start:,self.col_start:rawdata.ncols//2-rawdata.n_cols_overscan]
        # IMAGE U IS FLIPPED DURING READOUT
        U = np.flip(image[:,rawdata.ncols//2+rawdata.n_cols_overscan:],axis=1)[self.row_start:,self.col_start:]
        if self.mask_clusters:
            mL = rawdata.mask_clusters[self.row_start:,self.col_start:rawdata.ncols//2-rawdata.n_cols_overscan]
            mU = np.flip(rawdata.mask_clusters[:,rawdata.ncols//2+rawdata.n_cols_overscan:],axis=1)[self.row_start:,self.col_start:]
            # ignore any pixel (and its mirrowed pixel) that belongs to a high-E cluster (i.e. cluster from BuildMask)
            mask = np.logical_not(np.logical_or(mL.astype(bool),mU.astype(bool)))
            print(f"    - Apply mask from BuildClusterMask: {np.round(100-(mask.sum())/(mask.shape[0]*mask.shape[1])*100,2)} of the pixels have been masked (in both regions)")
       
        if self.do_fit:
            # fill the 2D histogram in ROOT
            Nbins = int((self.q_max_ADU-self.q_min_ADU)/self.bin_size)
            corr = ROOT.TH2F("L_vs_U","L_vs_U",Nbins,self.q_min_ADU,self.q_max_ADU,Nbins,self.q_min_ADU,self.q_max_ADU)
            corr.SetTitle(";L side [ADU];U side [ADU]")
            corr.Sumw2()

            for vall,valu in zip(L.flatten()[mask.flatten()],U.flatten()[mask.flatten()]):
                _ = corr.Fill(vall,valu)
            
            # function to fit as cppcode            
            fitfunc = ROOT.TF2("ff_2DGauss",twoD_gauss,self.q_min_ADU,self.q_max_ADU,self.q_min_ADU,self.q_max_ADU,14)
            fitfunc.SetNpy(int(Nbins*5))
            fitfunc.SetNpx(int(Nbins*5))
            fitfunc.SetLineColor(2)
            fitfunc.SetLineWidth(2)

            fitfunc.SetParNames("A_{0}","#theta","#lambda_{L}","#lambda_{U}","#sigma_{L}","#sigma_{U}","k_{L}","k_{U}","X0_{L}","Y0_{U}")
            # set fitting range for the different parameters
            fitfunc.SetParameter(0,corr.GetMaximum()/self.bin_size)
            fitfunc.SetParLimits(8,-1,1)
            fitfunc.SetParLimits(9,-1,1)
            for i,pname in enumerate(["theta","dc_L","dc_U","sig_L","sig_U","cal_L","cal_U"]):
                fitfunc.SetParameter(i+1,getattr(self,pname))
                plims = getattr(self,f"{pname}_range")
                fitfunc.SetParLimits(i+1,plims[0],plims[1])
            
            fitfunc.FixParameter(13,int(self.NpY))
            fitfunc.FixParameter(12,0)
            fitfunc.FixParameter(11,int(self.NpX))
            fitfunc.FixParameter(10,0)

            # fit the data
            corr.Fit(fitfunc,self.fit_opt+" N")
             
            ### UPDATE FITTED PARAMETERS
            param_comments = ["angle from the 2D-Gauss fit", "DC from the 2D-Gauss fit, L-side", "DC from the 2D-Gauss fit, U-side", 
                    "sigma from the 2D-Gauss fit, L-side", "sigma from the 2D-Gauss fit, U-side", 
                    "gain from the 2D-Gauss fit, L-side", "gain from the 2D-Gauss fit, U-side", ]

            for i,pname in enumerate(["theta","dc_L","dc_U","sig_L","sig_U","cal_L","cal_U"]):
                # UPDATE PARAMETERS
                setattr(self,pname,fitfunc.GetParameter(i+1))
                setattr(self,f"{pname}_err",fitfunc.GetParError(i+1))
                if not self.__silent__:
                    print(f"  - {pname} =  {fitfunc.GetParameter(i+1)}   +/-  {fitfunc.GetParError(i+1)}  ")
                # ADDING BEST FITTED PARAMS TO THE OUTPUT FITS FILE HEADER
                u._tohdr.update({f"TDG{pname.replace('_','').upper()}": (getattr(self,pname),param_comments[i]) })
                u._tohdr.update({f"TDG{pname.replace('_','').upper()}E":(getattr(self,f"{pname}_err"), f"error {param_comments[i]}" ) })

        # Append other params to header
        u._tohdr.update({"TDGNPIX": (int(corr.GetEntries()),"Total number of pixels corr.GetEntries")})

        if self.do_calibration:
            L_e = np.divide(L,self.cal_L)
            U_e = np.divide(U,self.cal_U)

            img_LU_e = np.concatenate((L_e,np.flip(U_e,axis=1)),axis=1)
            setattr(rawdata,"image_mean_compressed_pedestal_subtracted_e", img_LU_e)
            print(f"  - Calibrated image with L and U gain values {self.cal_L} ADUs and {self.cal_U} ADUs, respectively ")
      

        if self.do_fit:
            _isbatch = ROOT.gROOT.IsBatch()
            ROOT.gROOT.SetBatch(not self.__display__)

            ROOT.gStyle.SetOptStat(0)
            ROOT.gStyle.SetOptFit(111)
           
            c = ROOT.TCanvas()
            c.SetLogz()

            corr.Draw("COLZ")
            # set contour limits
            lmin = int(np.log10(corr.GetMaximum()/1e8))
            lmax = int(np.log10(corr.GetMaximum()*10))
            levels = array('d',[10**i for i in np.arange(lmin,lmax,0.8)])
            fitfunc.SetContour(len(levels), levels)
            fitfunc.SetNpx(500)
            fitfunc.SetNpy(500)
            fitfunc.Draw("same")

            if self.__display__:
                c.Update()
                c.Draw()
                input("press enter ...")
            for fmt in self.format_figures:
                c.SaveAs(f"{rawdata.output}{self.__name__}_BestFit.{fmt}")


            ROOT.gROOT.SetBatch(_isbatch)

        
####################################################################################################################
#
#
#   ANALYSIS OF A GIVEN REGION: FITTING THE PCD TO A GAUSSIAN TO GIVE THE MEAN AND SIGMA
#
#
####################################################################################################################
class GaussianFitProcess(SKImageProcess):
    """
    """
    __sequence_id__ = 22
    __name__ = 'GaussianFitProcess'
    
    def __init__(self):
        """
        """
        super().__init__()
        # should be true by construction
        self.__DEBUG__ = True
        self.image = "mean_compressed"

        self.in_overscan = True
        self.skip_cols = []
        self.skip_rows = []
        
        # to select only pixels within the 10-90 percentile
        self.in_trim_mean = True
        self.n_sigma = 3
        #self.show_outliers = True

        self.hist_nbins_method = "freedman"

        # other params
        self.n_sigma_win_fit = 2
        
        ### 
        self.__units__.update({
            'in_overscan':1, 'in_trim_mean':1,'n_sigma':1,'skip_cols':1,'skip_rows':1,'show_outliers':1,'hist_nbins_method':1, 
            "__DEBUG__":1
            })
    
    def execute_process(self,rawdata):
        """
        """
        
        self.method = 'gaus_fit'
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)

        if not self.hist_nbins_method.lower() in ["default","freedman","sturge"]:
            raise AttributeError(f"<{self.__name__}> ERROR: Available methods to calculate Nbins of the PCD: sturge, freedman or default")

        print("Process <GaussianFitProcess> Fit a Gaussian to the given region for amplifier ".format(amplifier))

        #### GET IMAGE TO ESTIMATE THE PEDESTAL
        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image)
        
        #### SET THE REGION TO BE USED TO FIT A GAUSSIAN
        ################################################################################
        if self.in_overscan:
            print("   - using overscan region")
            #mask_image_overscan_cols = rawdata.mask_image_overscan_cols[user_rows_range,user_cols_range]
            image = np.ma.array(image_amp,mask=rawdata.mask_image_overscan_cols)
        else:
            print("   - using the sensitive region")
            image = np.ma.array(image_amp,mask=rawdata.mask_image_active_region)

            #### skip cols and rows only if possible
            for col in self.skip_cols:
                if type(col)==list:
                    for ci in range(col[0],col[1]+1):
                        image.mask[:,ci] = True
                else:
                    image.mask[:,col] = True
            for row in self.skip_rows:
                if type(row)==list:
                    for ri in range(row[0],row[1]+1):
                        image.mask[ri,:] = True
                else:
                    image.mask[row,:] = True


        #### START THE PROCESS
        ################################################################################
        self.th1_guas_fits = []
        
        # pixels within the user region
        img_ax = image.compressed()
        print(f"        q_min ={img_ax.min()},q_max={img_ax.max()}")

        if self.in_trim_mean:
            # select only pixels within 10-90 percentil
            pmean = trim_mean(img_ax, 0.1)
            pstd  = trimmed_std(img_ax, 0.1)
            img_ax = img_ax[np.logical_and(img_ax>pmean-self.n_sigma*pstd, img_ax<pmean+self.n_sigma*pstd)]

        ### ESTIMATE PEDESTAL: mu and sigma 
        ###     apply function (gauss fit or median) to the selected pixel list (img_ax)
        #self.__DEBUG__ = True
        mu,emu,sigma,esigma,chi2 = self.fit_gaussian(img_ax,nsig=self.n_sigma_win_fit,axis=0,**{"Nbins":self.hist_nbins_method})
        
        print(" best fitted values: ")
        print(" \t STD : {} +/- {} [chi2 {}]\n".format(sigma, esigma, chi2))


        # UPDATE HEADER ACCORDINGLY
        ######################################################################################################
        rawdata.image_header['GFSIG']  = (sigma,  f"Gaussian Fit: sigma (in image units), in OVS: {self.in_overscan}")
        rawdata.image_header['GFESIG'] = (esigma, 'Gaussian Fit: error on the sigma')
        rawdata.image_header['GFCHI2'] = (chi2,   'Gaussian Fit: Chi-square')

        if self.exit<0:
            print("Bad fitting Error Code ", self.exit)

        return

class CalibrationProcess(SKImageProcess):
    """The image will be calibrated: from ADC to electrons using the gain parameter

    """
    __sequence_id__ = 25
    __name__ = 'CalibrationProcess'

    def __init__(self):
        """
        """
        super().__init__()
        self._per_amp = False
        self.__silent__ = False
        self.image = "mean_compressed_pedestal_subtracted"

        # NO default: must be given by the user (json) or taken from the dark current fit
        #   number          -> constant model (ADU/e-)
        #   [a, b, c, ...]  -> ADU = a + b*e + c*e^2 + ...
        #   {amp: number | list}
        self.gain  = 1 * u.ADC/u.e
        self.offset = 0.0
        self.valid_range = None
        self.from_dc_fit = False
        
        self.create_ttree = False
        ### 
        self.__units__.update({'gain':1,'offset':1,'valid_range':1,'from_dc_fit':1, 'create_ttree':1})
    

class CalibrationProcess(SKImageProcess):
    """The image will be calibrated: from ADC to electrons.

    Calibration curve (fit of the position of each peak, in ADU, vs. the number of electrons):

            ADU = offset + g*e + c2*e^2 + c3*e^3 + ...

    `gain` (what goes in the json):
      - a number g: linear model, ADU = offset + g*e           (g in ADU/e-)
      - a list/tuple [g, c2, c3, ...]: polynomial model from first order,
        with units  g: ADU/e-,  c2: ADU/e-^2,  c3: ADU/e-^3, ...
        so that the number g is equivalent to the list [g]
      - a dict {amplifier: number | list}: one of the above per amplifier

    `offset` (optional, separate from `gain`): constant term of the curve in ADU, i.e. the
    position of the zero-electron peak (p0 of the fit). It is NOT part of the `gain` list.
      - a number, or a dict {amplifier: number}. Default 0.

    `valid_range` (optional): range of electrons [e_min, e_max] where the polynomial is
    supported by the fitted peaks (it is written by get_calibration_constant).
      - a list [e_min, e_max], or a dict {amplifier: [e_min, e_max]}. Default None.
      Inside the range the polynomial is inverted; outside it the curve is continued
      linearly with the slope of the polynomial at the edge (continuous in value and
      slope), so any ADU value (negative noise, saturated pixels, ...) gives a finite,
      monotonic result. With None the polynomial is inverted over the whole axis, which
      fails (RuntimeError) for ADU values beyond the extremum of the polynomial.
      Only used for polynomial models (more than one coefficient in `gain`).

    The input image must be in ADU (pedestal subtracted); the output is in e-.
    Default is gain=1, offset=0 (e = ADU).
    The polynomial is only supported by the data in the range of the fitted peaks.
    """
    __sequence_id__ = 25
    __name__ = 'CalibrationProcess'

    # internal settings of the polynomial inversion (not user parameters)
    _MAX_ITER = 50
    _TOL = 1e-9

    def __init__(self):
        """
        """
        super().__init__()
        self._per_amp = False

        self.__silent__ = False
        self.image = "mean_compressed_pedestal_subtracted"
        self.gain  = 1
        self.offset = 0.0
        self.valid_range = None
        self.from_dc_fit = False

        self.create_ttree = False
        ### 
        self.__units__.update({'gain':1,'offset':1,'valid_range':1,'from_dc_fit':1, 'create_ttree':1})

    def _adu_to_e(self, adu, coeffs, erange=None):
        """ADU = poly(e)  ->  e = poly^-1(ADU).

        Inside `erange` the polynomial is inverted with Newton starting from the linear
        solution; outside it is continued linearly with the slope at the edge.
        `erange` None: the polynomial is inverted everywhere.
        """
        from numpy.polynomial import polynomial as P
        adu = np.asarray(adu, dtype=float)
        e = (adu - coeffs[0]) / coeffs[1]
        if coeffs.size == 2:
            return e
        dcoeffs = P.polyder(coeffs)

        if erange is None:
            lo, hi = -np.inf, np.inf
        else:
            lo, hi = float(erange[0]), float(erange[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"{self.__name__}: invalid valid_range {erange}")
            if np.any(P.polyval(np.linspace(lo, hi, 2001), dcoeffs) <= 0):
                raise ValueError(f"{self.__name__}: the calibration polynomial is not monotonic "
                                 f"in its valid range {erange}")

        out = e.copy()
        inside = np.ones(adu.shape, dtype=bool)
        if np.isfinite(lo):
            a_lo, s_lo = P.polyval(lo, coeffs), P.polyval(lo, dcoeffs)
            below = adu < a_lo
            out[below] = lo + (adu[below] - a_lo) / s_lo
            inside &= ~below
        if np.isfinite(hi):
            a_hi, s_hi = P.polyval(hi, coeffs), P.polyval(hi, dcoeffs)
            above = adu > a_hi
            out[above] = hi + (adu[above] - a_hi) / s_hi
            inside &= ~above

        if inside.any():
            x, y = np.clip(e[inside], lo, hi), adu[inside]
            for _ in range(self._MAX_ITER):
                step = (P.polyval(x, coeffs) - y) / P.polyval(x, dcoeffs)
                x = np.clip(x - step, lo, hi)
                if np.max(np.abs(step)) < self._TOL:
                    break
            else:
                raise RuntimeError(f"{self.__name__}: the inversion of the polynomial calibration did not converge")
            out[inside] = x
        return out

    def execute_process(self,rawdata):
        """
        """
        # GET calibration value from FitDarkCurrentProcess or from user config
        ################################################################################
        def get_gain_value(amp_name, gain, from_dc_fit):
            if from_dc_fit and f"MEGAIN{amp_name}" in u._tohdr:
                return np.round(u._tohdr[f"MEGAIN{amp_name}"][0],4)
            if type(gain)==dict:
                # when coming from DQM not all ampl will have its gain, so this should be set to be 1
                try:
                    return gain[amp_name]
                except KeyError:
                    print(f" WARNING: Calibration set to 1 {amp_name} is not found in calibration dictionary ")
                    return 1
            return gain

        def get_offset_value(amp_name, offset):
            if type(offset)==dict:
                return offset.get(amp_name, 0.0)
            return offset

        def get_range_value(amp_name, valid_range):
            if type(valid_range)==dict:
                return valid_range.get(amp_name, None)
            return valid_range

        if not self.__silent__:
            print("Process <CalibrationProcess> INFO.")

        #### IMAGE TO CALIBRATE
        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        try:
            image_to_cal = getattr(rawdata,itype)
        except AttributeError:
            raise IOError(f"{self.__name__}: Image does not exists")
        data = np.ma.getdata(image_to_cal).astype(float)

        #### CALIBRATE EACH AMPLIFIER REGION
        ################################################################################
        calibrated = np.zeros_like(data)
        for amp_name in rawdata.amplifier.keys():
            value  = get_gain_value(amp_name, self.gain, self.from_dc_fit)
            offset = float(get_offset_value(amp_name, self.offset))
            erange = get_range_value(amp_name, self.valid_range)
            region = np.logical_not(rawdata.amplifier[amp_name]['mask_full_extension'])
            if region.shape != data.shape:
                raise IOError(f"{self.__name__}: Unexpected error, calibration mask and image has not the same shape")

            # orders: [g, c2, c3, ...]  (a number g is the list [g])
            if isinstance(value, (list, tuple, np.ndarray)):
                orders = np.array(value, dtype=float).ravel()
                is_poly = orders.size > 1
            else:
                orders = np.array([float(value)])
                is_poly = False
            if orders.size < 1 or not np.all(np.isfinite(orders)) or orders[0] == 0 or not np.isfinite(offset):
                raise ValueError(f"{self.__name__}: invalid calibration for '{amp_name}': gain={value}, offset={offset} "
                                 "(needs finite values and a non-zero first coefficient)")
            coeffs = np.concatenate(([offset], orders))      # [offset, g, c2, c3, ...]

            calibrated[region] = self._adu_to_e(data[region], coeffs, erange if is_poly else None)

            # first order term = gain in ADU/e- (slope of the curve at 0 e-)
            val = float(orders[0])
            if is_poly:
                for k, ck in enumerate(orders, start=1):
                    u._tohdr.update({f"CAL{amp_name}{k}": (float(ck), f"calibration poly coeff e^{k}: ADU = off + sum c_k e^k")})
                if erange is not None:
                    u._tohdr.update({f"CAL{amp_name}R0": (float(erange[0]), "calibration valid range, min e-"),
                                     f"CAL{amp_name}R1": (float(erange[1]), "calibration valid range, max e-")})
            if is_poly or offset != 0:
                u._tohdr.update({f"CAL{amp_name}0": (offset, "calibration offset in ADU (zero-e- peak position)")})
            if not self.__silent__:
                if is_poly or offset != 0:
                    terms = [f"{offset:g}"] + [f"{c:g}*e^{k}" for k,c in enumerate(orders, start=1)]
                    print(f"    amplifier/ccd {amp_name}: ADU = " + " + ".join(terms)
                          + (f"   (valid for {erange[0]:g} <= e- <= {erange[1]:g}, linear outside)" if is_poly and erange is not None else ""))
                else:
                    print(f"    amplifier/ccd {amp_name} with gain of {val} ADC/e-")

            rawdata.execute_process_in_amp = amp_name
            rawdata.set_image_by_attribute_name(None,None,**{'calibration':val,'ADC2e':val})
            #### update calibration/gain in the u singleton
            u.set_gain(amp_name,val)
            #### add parameter to the fits file header
            u._tohdr.update({f"GAIN_{amp_name}": (val,"calibration constant ADU/e (first order term)")})

        setattr(rawdata,"{}_e".format(itype), calibrated)

        if self.create_ttree:
            self.save_images_as_root(rawdata)


##############################################################################################
#
#
#   PROCESS TO ADD NOISE TO THE FULL IMAGE (FOR SIMULATED IMAGES WITHOUT NOISE THAT SHOULD 
#       BE RECONSTRUCTED AS DATA)
#
##############################################################################################
class GaussianNoiseProcess(SKImageProcess):
    """A Gaussian noise will be apply to the Full image

    """
    __sequence_id__ = 21
    __name__ = 'GaussianNoiseProcess'

    def __init__(self):
        """
        """
        super().__init__()
        self._per_amp = False

        self.__silent__ = False
        self.image = "mean_compressed"
        self.sigma = 0.2758
        ### 
        self.__units__.update({'sigma':1})
    
    def execute_process(self,rawdata):
        """
        """
        if not self.__silent__:
            print(f"Process <{self.__name__}> INFO. A gaussian noise with mu=0, and sigma={self.sigma} (in units of {self.image}) will be apply to the full image")

        itype = rawdata.get_image_attribute_name(self.image)
        # from eV to electrons
        image = rawdata.get_image_by_attribute_name(self.image).data
        
        setattr(rawdata,f"{itype}_wgnoise", np.random.normal(image,self.sigma))
        return


#################################################################################################
#                                                                                               #
#       Process Class to fit hte pixel charge distribution                                      #
#           Calculate dark current and readout noise                                            #
#                                                                                               #
#################################################################################################
class FitDarkCurrentProcess(SKImageProcess):
    """Fits the pixel charge distribution
    """

    __sequence_id__ = 30
    __name__ = 'FitDarkCurrentProcess'

    def __init__(self):
        """
        """
        super().__init__()

        self.image          = "mean_compressed_pedestal_subtracted"
        
        # recursive algorithm
        self._ntrails   = 0
        self.redo       = True

        # options to select data from image
        self.in_overscan    = False
        self.mask_clusters  = False
        self.rows_to_mask   = []
        self.cols_to_mask   = []

        self.fit_options    = "QR S L EM"
        self.do_calibration = True
        self.calibration    = 10.0
        self.binning_size   = 0.5

        # model parameters
        #  Norm * Gaussian(x/gain, i+mu0/gain, sigma/gain ) * Poisson(i, lambda/gain ) +  (A + x/gain + B/gain)
        # sorted params:  Norm, mu0, sigma, lambda, gain, CRx, CR0
        # linear model added to dark current: 
        self.add_linear_DQ              = False
        self.CRx                        = 0.0
        self.CR0                        = 0.0
        # dark current params
        self.n_peaks                    = 5
        # center of the zero-e peak
        self.mu_min, self.mu_max        = -1,1
        # noise, single electron resolution
        self.sigma_min, self.sigma_max  = 0.001,100.0
        # gain/calibration
        self.gain_min, self.gain_max    = 1.0,100.0
        # dark current, ratio  between the two first peaks
        self.dc_min, self.dc_max        = 0.0010,10.0
        # limits for the spectrol window
        self.x_min, self.x_max  = -10,50
        # optional upper limit of the plotted histogram (>x_max): the fit still uses [x_min,x_max]
        self.x_max_plot = None

        self.ratio_plot = False

        # OPTION TO ADD A MASK VIA FITS FILE(+EXT)
        self.apply_mask_from_fits = [False,None,None]
        # ADD DATA FO FIT VIA CSV FILE [bool,file name,column name]
        self.data_as_csv = [False,None,None]
        
        # to store results in units of e- instead of ADU, and include other ME
        self.dqm_mode = False

        self.__units__.update({
            'mask_clusters':1,'in_overscan':1,'rows_to_mask':1,'cols_to_mask':1,
            'do_calibration':1,'calibration':1,'binning_size':1,'add_linear_DQ':1,'n_peaks':1,'mu_min':1,'mu_max':1,'dc_min':1,'dc_max':1,
            'sigma_min':1,'sigma_max':1,'gain_min':1,'gain_max':1,'x_min':1,'x_max':1,'x_max_plot':1,
            'ratio_plot':1,'fit_options':1,'apply_mask_from_fits':1,'data_as_csv':1, 'dqm_mode':1
            })

    def execute_process(self,rawdata,**kwargs):
        """
        """
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)

        if not self.__silent__:
            print("Process <FitDarkCurrentProcess> INFO. Fit the n-first peaks to estimate the dark current and the calibration constant {}.".format(amplifier))

        # set batch mode
        _isbatch = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(1)
        
        # n_peaks is used around and could be dependent of the amplifer, keep configuration value
        # under self._n_peaks
        if type(self.n_peaks)==dict:
            self._n_peaks = self.n_peaks[rawdata.execute_process_in_amp]
        else:
            self._n_peaks = self.n_peaks

        data = self.get_data_array(rawdata,**kwargs)
        
        # get calibration and sigma_guess
        gain_guess = self.get_calibration()
        
        # If xmin,xmax is not defined, define the spectral window based on 
        # this sigma_guess and gain_guess values
        if self.x_min is None or np.isnan(self.x_min):
            self.x_min = -gain_guess
        if self.x_max is None or np.isnan(self.x_max):
            self.x_max = self._n_peaks*gain_guess
        
        # select pixels within this spectral window
        pcd_array = data[(data>self.x_min)&(data<self.x_max)]
        self.included_pixels = pcd_array.size

        # define the binning
        if self.binning_size>0:
            n_bins = int(abs(self.x_max-self.x_min)/self.binning_size)
            self.binning_size_used = self.binning_size
        else:
            n_bins = int(2*np.sqrt(len(pcd_array)))
            self.binning_size_used =  abs(self.x_max-self.x_min)/n_bins
        # bins in the fit window (used for the normalization guess)
        self.n_bins_used = n_bins

        # histogram range: [x_min,x_max] unless x_max_plot asks to show more charge on the right.
        # Extra bins have the same width as the fit-window bins, so the bin edges at x_max are unchanged
        self._x_hist_max = self.x_max
        if self.x_max_plot is not None and self.x_max_plot > self.x_max:
            bin_width = abs(self.x_max-self.x_min)/n_bins
            n_extra = int(np.ceil((self.x_max_plot-self.x_max)/bin_width))
            n_bins += n_extra
            self._x_hist_max = self.x_max + n_extra*bin_width
        hist_array = data[(data>self.x_min)&(data<self._x_hist_max)]

        # FITTING USING ROOT
        pcd = ROOT.TH1F("{} [r{},t{}]".format(rawdata._file_name.split(".")[0],np.random.randint(100),self._ntrails),
                "Dark Current Fit {}".format(amplifier),n_bins,self.x_min,self._x_hist_max)
        pcd.SetDirectory(0)

        for qij in hist_array:
            pcd.Fill(qij)

        # get function to fit
        fitfunc = self.get_fitfunc(len(pcd_array),gain_guess,rawdata.execute_process_in_amp)
        # "R": fit only within the function range [x_min,x_max], even if the histogram is wider
        fit_options = self.fit_options + "R"
        # fit PCD
        pcd.Fit(fitfunc,fit_options)
        # re-fit starting with the best parameters
        fitfunc.SetParameters(fitfunc.GetParameter(0),fitfunc.GetParameter(1),fitfunc.GetParameter(2),fitfunc.GetParameter(3),fitfunc.GetParameter(4))
        pcd.Fit(fitfunc,fit_options)

        # add the fitted function
        self.fitfunc = fitfunc   

        # get results
        results = self.get_results(self.fitfunc)        
        # this should be given by the LL 
        # self.chi2 = self.fitfunc.GetChisquare()/self.fitfunc.GetNDF()
        try:
            self.chi2 = pcd.Chisquare(self.fitfunc,"L")/self.fitfunc.GetNDF()
        except ZeroDivisionError:
            self.chi2 = -999
        # some funcitons needs the `pcd` object
        self.pcd_hist = pcd

        # display plot
        nskips = int(rawdata.image_header['NSKIPS']) if 'NSKIPS' in rawdata.image_header else rawdata.nskips
        if nskips==1 and rawdata._NDCM>1:
            nskips = rawdata._NDCM
        
        if self.__display__ or self.save_plots:
            text = [f"CCD-{rawdata.execute_process_in_amp}",
                    f"frame: {rawdata.nrows}x{rawdata.ncols}x{nskips}", 
                    f"bin: {rawdata.npbin}x{rawdata.nsbin}x{nskips}"]
            if rawdata.exposure_time>0:
                text.append("t_{exp}="+str(round(rawdata.exposure_time,1))+"s")
            if rawdata.read_time>0:
                text.append("t_{read}="+str(round(rawdata.read_time,1))+"s")

            self.execute_plot(rawdata,pcd, text=text)
      
        ROOT.gROOT.SetBatch(_isbatch)
        ROOT.gROOT.GetListOfCanvases().Delete()
        
        # ADD parameters to the Units class to be included later on in the fits file header, if an image output is produced
        self.attributes_to_update_fits_header(rawdata.execute_process_in_amp,rawdata.n_image,rawdata.n_run)
        if not self.__silent__:
            print("")

        if self.dqm_mode:
            results_me = self.get_results_in_electrons(rawdata)
            x,y,y_fit  = self.get_dcfit_data()
            results_me.update({'pcd': {'x':x, 'y':y, 'y_fit':y_fit}})
            setattr(rawdata,'me_dcfit_results_{}'.format(rawdata.execute_process_in_amp),results_me )

        return

    def attributes_to_update_fits_header(self,amp,image_id,run_id):
        """ADD to Units class the attributes to properly update fits file header
        """
        
        if self.image[-2:] == "_e":
            values_in = "e-"
        else:
            values_in = "ADU"

        u._tohdr.update({
            'MEGAIN{}'.format(amp)  :(self.dc_gain,'DCFit: fitted gain {}/e-'.format(values_in)),
            'MEDC{}'.format(amp)    :(self.dc_lambda,'DCFit: fitted DC {}/bin/img'.format(values_in)),
            'MESIG{}'.format(amp)   :(self.dc_sigma,'DCFit: fitted sigma {}'.format(values_in)),
            'MEMU0{}'.format(amp)   :(self.dc_mu0,'DCFit: fitted zero-e peak center in {}'.format(values_in)),
            'EMEGAIN{}'.format(amp) :(self.dc_gain_err,'DCFit: error of MEGAIN'),
            'EMEDC{}'.format(amp)   :(self.dc_lambda_err,'DCFit: error of MELAMBDA'),
            'EMESIG{}'.format(amp)  :(self.dc_sigma_err,'DCFit: error of MESIGMA'),
            'EMEMU0{}'.format(amp)  :(self.dc_mu0_err,'DCFit: error of MEMU0'),
            'MEIMGID{}'.format(amp) :(image_id,'DCFit: Assigned image ID'),
            'MERUNID{}'.format(amp) :(run_id,'DCFit: Assigned run ID')
            })

        if hasattr(self,"total_pixels") and hasattr(self,"included_pixels"):
            u._tohdr.update({'MEFRAC{}'.format(amp)  :(self.included_pixels/self.total_pixels,'DCFit: fraction of included pixels')})
                    

        return

    def get_results_in_electrons(self,rawdata,per_day=False):
        """Return the paramters from the fit in units of

        mu0     : electrons
        sigma   : electrons
        dc      : electrons/pix/day if per_day is True, otherwise will be in e/pix/img
        calibration : ADC/e
        """

        if per_day:
            tot_time = rawdata.exposure_time + rawdata.read_time/2.
            tot_time = tot_time if tot_time>0 else 1
            tot_time = (60*60*24) / tot_time
        else:
            tot_time = 1.0


        get_error = lambda p: ((p[1]/p[2])**2 + ((p[0]*p[3]/p[2]**2))**2)**0.5
        # conversion factor 
        convfact = (1/(rawdata.bin_col*rawdata.bin_row))*tot_time
        
        mu0   = self.dc_mu0/self.dc_gain
        sigma = self.dc_sigma/self.dc_gain
        dc    = self.dc_lambda/self.dc_gain * convfact
        calibration = self.dc_gain

        mu0_err   = get_error([self.dc_mu0,self.dc_mu0_err,self.dc_gain,self.dc_gain_err])
        sigma_err = get_error([self.dc_sigma,self.dc_sigma_err,self.dc_gain,self.dc_gain_err])
        dc_err    = get_error([self.dc_lambda,self.dc_lambda_err,self.dc_gain,self.dc_gain_err])*convfact
        calibration_err = self.dc_gain_err

        results = { 'mu0': (mu0,mu0_err), 
                    'sigma': (sigma,sigma_err), 
                    'dc': (dc,dc_err),
                    'calibration': (calibration,calibration_err), 
                    'chi2': self.chi2, 'to_e_pix_day':convfact  
                    }
        
        return results 

    def get_dcfit_data(self):
        ### data used to fit
        n_entries = int(self.pcd_hist.GetNbinsX())
        x = []
        y = []
        y_err = []
        y_fit = []

        ### fitfunc
        model = self.fitfunc
        for i in range(n_entries):
            x.append(self.pcd_hist.GetBinCenter(i+1))
            y.append(self.pcd_hist.GetBinContent(i+1))
            y_fit.append(self.fitfunc(x[-1]))

        return x,y,y_fit



    def execute_plot(self,rawdata,pcd_hist,text=None):

        _isbatch = ROOT.gROOT.IsBatch()
        if self.__display__:
            ROOT.gROOT.SetBatch(0)
 
        can = ROOT.TCanvas("")
        can.Clear()
        
        # pcd_hist.GetXaxis().SetTitle("Charge (ADU)")
        # pcd_hist.GetYaxis().SetTitle("Counts")
        # pcd_hist.GetXaxis().CenterTitle(True)
        # pcd_hist.GetYaxis().CenterTitle(True)
        # pcd_hist.GetXaxis().SetTitleOffset(1.2)  # Adjust as needed
        # pcd_hist.GetYaxis().SetTitleOffset(1.6)  # Adjust for vertical centering
        pcd_hist.SetName("pcd")
        pcd_hist.GetXaxis().SetTitle("Pixel Charge [ADU]")
        pcd_hist.GetYaxis().SetTitle("Counts")
        pcd_hist.SetMarkerStyle(21)
        pcd_hist.Draw()

        # if the histogram extends beyond the fit window, draw the fitted model there as a dashed line
        x_hist_max = pcd_hist.GetXaxis().GetXmax()
        if x_hist_max > self.x_max:
            # evaluated point by point (cloning the TF1 makes ROOT re-parse its formula, which
            # fails with the parameter names set in get_fitfunc)
            xs = np.linspace(self.x_max, x_hist_max, 1000)
            ys = np.array([self.fitfunc.Eval(xi) for xi in xs])
            self._fitfunc_ext = ROOT.TGraph(len(xs), xs, ys)
            self._fitfunc_ext.SetLineColor(self.fitfunc.GetLineColor())
            self._fitfunc_ext.SetLineWidth(self.fitfunc.GetLineWidth())
            self._fitfunc_ext.SetLineStyle(2)
            self._fitfunc_ext.Draw("L SAME")


        if self.mask_clusters and hasattr(self,'_pcd_no_masked'):
            ### Add histogram with the non-masked PCD
            nmhist = ROOT.TH1F("non-masked {}".format(np.random.randint(1000)), "non-masked PCD",pcd_hist.GetNbinsX(),pcd_hist.GetXaxis().GetXmin(),pcd_hist.GetXaxis().GetXmax())
            nmhist.SetName("pcd_raw")
            nmhist.SetMarkerStyle(22)
            nmhist.SetMarkerSize(0.4)
            nmhist.SetMarkerColor(ROOT.kAzure-8)
            nmhist.SetLineStyle(9)
            nmhist.SetLineWidth(2)
            nmhist.SetLineColor(ROOT.kAzure-8)
            nmhist.SetLineColorAlpha(ROOT.kAzure-8,0.4)
            nmhist.SetMarkerColorAlpha(ROOT.kAzure-8,0.4)
            ### FILLING HISTOGRAM using ROOT much easier than Minuit (in python)
            for qij in self._pcd_no_masked:
                nmhist.Fill(qij)
            nmhist.Draw("SAME PE")
        can.Update()

        ### add text
        if text is not None:
            for i,t in enumerate(text):
                ttext = ROOT.TLatex()
                ttext.SetTextSize(0.031)
                ttext.DrawLatexNDC(0.68,0.39+0.052*i, "#color[26]{"+t+"}")
        can.Update()
        
        # ### re-place the TPaveStats
        # # stat_box = pcd_hist.FindObject("stats")
        # # stat_box.SetX1NDC(0.62)
        # # stat_box.SetY1NDC(0.63)
        # # stat_box.SetX2NDC(0.96)
        # # stat_box.SetY2NDC(0.91)

        # pcd_hist.SetStats(0)

        # # Custom stat box
        # stats = ROOT.TPaveText(0.62, 0.63, 0.96, 0.91, "NDC")
        # stats.SetFillStyle(0)
        # stats.SetBorderSize(0)
        # stats.SetTextFont(42)
        # stats.SetTextSize(0.032)
        # stats.SetTextColor(ROOT.kGray+2)  # use grayish tone

        # # Add only the desired entries
        # stats.AddText(f"#sigma = {self.dc_sigma:.4f} ± {self.dc_sigma_err:.4f} ADU")
        # stats.AddText(f"#lambda = {self.dc_lambda:.4f} ± {self.dc_lambda_err:.4f} ADU/bin/img")
        # stats.AddText(f"gain = {self.dc_gain:.4f} ± {self.dc_gain_err:.4f} ADU/e⁻")

        # stats.Draw()
        ### re-place the TPaveStats
        stat_box = pcd_hist.FindObject("stats")
        stat_box.SetX1NDC(0.62)
        stat_box.SetY1NDC(0.63)
        stat_box.SetX2NDC(0.96)
        stat_box.SetY2NDC(0.91)
        can.Update()
        
        can.SetLogy()
        can.Update()

        if self.save_plots:
            for fmt in self.format_figures:
                if fmt=="root":
                    # rename fitfunc parameters to avoid wrong expression error
                    pcd_hist.GetFunction(f"fitfunc_{rawdata.execute_process_in_amp}").SetParNames("norm", "mu", "sigma", "lambda", "gain", "DCx", "DC0")
                    outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}.root","UPDATE")
                    pcd_hist.Write()
                    if self.mask_clusters and hasattr(self,'_pcd_no_masked'):
                        nmhist.Write()
                    outrootfile.Close()
                else:
                    can.SaveAs(rawdata.output+"{}_DCfit_{}.{}".format(self.__name__,rawdata.execute_process_in_amp,fmt))

        if self.__display__:
            can.Draw()
            input("\nPress Enter ........ ")
        
        if self.ratio_plot and self.__display__:
            create_ratio_plot(self.fitfunc, pcd_hist, display=True, xlabel='Pixel Charge', ylabel='Events')

        ROOT.gROOT.GetListOfCanvases().Delete()
        ROOT.gROOT.SetBatch(_isbatch)

        return


    def get_results(self,fitfunc):
        """Creates a dictionary with all fitted parameters
        """
        par_names = ['dc_norm','dc_mu0','dc_sigma','dc_lambda','dc_gain']
        par_names_msm = ["Normalization", "zero-e- peak position [ADU]", "single-e- resolution [ADU]", "dark current [ADU/bin/img]","gain [ADU/e-]"]
        results = {}
        for i,par in enumerate(par_names):
            results[par]        = fitfunc.GetParameter(i)
            results[par+"_err"] = fitfunc.GetParError(i)
            # append as attributte
            setattr(self,par,fitfunc.GetParameter(i))
            setattr(self,par+"_err",fitfunc.GetParError(i))

            if not self.__silent__:
                print("     - {} = {} +/- {}".format(par_names_msm[i],results[par],results[par+"_err"]))
        print("")
        print(f"     - Noise [e-] = {results['dc_sigma']/results['dc_gain']:.3f} e-")
                
        # set 0 if calibration was not fitted
        if self.do_calibration:
            results['dc_gain_err'] = 0.0
        
        return results

    def get_fitfunc(self,Npoints,gain_guess,amp):
        """Set the funtion to fit the pixel charge distribution
        """
        
        fitfunc = []
        for peak_i in range(int(self._n_peaks)):
            fitfunc.append("[0]*(TMath::Gaus(x/[4],{0}+[1]/[4],[2]/[4],1) * (TMath::Poisson({0},[3]/[4]) + ([5]*x/[4]+[6]/[4]) ))".format(peak_i))

        fitfunc = ROOT.TF1(f"fitfunc_{amp}", " + ".join(fitfunc), self.x_min, self.x_max, 7)
        ### plotting options
        fitfunc.SetLineColor(ROOT.kRed)
        fitfunc.SetLineWidth(4)
        fitfunc.SetLineStyle(1)

        # re-name names of fitted params
        if self.do_calibration:
            fitfunc.SetParNames("Norm", "#mu_{0}", "#sigma[ADU]", "#lambda[ADU/bin/img]", "gain[ADU/e]", "CR_{x}", "CR_{0}")
        else:
            fitfunc.SetParNames("Norm", "#mu_{0}", "#sigma", "#lambda", "gain[ADU/e]", "CR_{x}", "CR_{0}")
        self.norm_guess = Npoints/self.n_bins_used
        
        # center of the first guassian
        mu_min = self.mu_min if not type(self.mu_min)==dict else self.mu_min[amp]
        mu_max = self.mu_max if not type(self.mu_max)==dict else self.mu_max[amp]
        # noise
        sigma_min = self.sigma_min if not type(self.sigma_min)==dict else self.sigma_min[amp]
        sigma_max = self.sigma_max if not type(self.sigma_max)==dict else self.sigma_max[amp]
        # dark current
        dc_min = self.dc_min if not type(self.dc_min)==dict else self.dc_min[amp]
        dc_max = self.dc_max if not type(self.dc_max)==dict else self.dc_max[amp]
        # calibration/gain
        gain_min = self.gain_min if not type(self.gain_min)==dict else self.gain_min[amp]
        gain_max = self.gain_max if not type(self.gain_max)==dict else self.gain_max[amp]
        
        #if hasattr(self,'normalization_HC'):
        #    self.norm_guess = self.normalization_HC * self.binning_size

        fitfunc.SetParameters(
                self.norm_guess,
                np.mean([mu_min,mu_max]),
                self.sigma_guess,
                gain_guess,
                self.CRx,
                self.CR0)
        
        # harcodding normalization
        #if hasattr(self,'normalization_HC'):
        #    fitfunc.FixParameter(0,self.normalization_HC * self.binning_size)

        # add limits for the paramters
        fitfunc.SetParLimits(1,mu_min,mu_max)
        fitfunc.SetParLimits(2,sigma_min,sigma_max)
        fitfunc.SetParLimits(3,dc_min,dc_max)

        # calibration can be frozen
        if self.do_calibration:
            fitfunc.SetParLimits(4,gain_min,gain_max)
        else:
            if self.calibration>0:
                fitfunc.FixParameter(4,self.calibration)
            else:
                fitfunc.FixParameter(4,1)

        # linear model to the Dark Current
        if self.add_linear_DQ:
            fitfunc.SetParLimits(5,-1,1)
            fitfunc.SetParLimits(6,0,50)
        else:
            fitfunc.FixParameter(5,0)
            fitfunc.FixParameter(6,0)

        return fitfunc
    
    def get_calibration(self):
        # define calibration and rewrite gain interval
        if self.calibration is None:
            if hasattr(rawdata,'itgtime'):
                calibration = rawdata.itgtime
            else:
                calibration = 1.05
            
            # override gain interval
            self.gain_min, self.gain_max = rawdata.itgtime-3, rawdata.itgtime+3
        else:
            calibration = self.calibration
        
        self.sigma_guess = calibration*0.2
        return calibration


    def get_data_array(self,rawdata,**kwargs):
        
        if self.data_as_csv[0]:
            # data is given in a CSV file, the second argument the file name, and 3rd the column name 
            import pandas as pd
            return pd.read_csv(self.data_as_csv[1])[self.data_as_csv[2]].values

        if 'data' in kwargs:
            #setattr(self,'normalization_HC', len(kwargs['data']))
            return kwargs['data']
        
        # set and get image for the current amplifier
        ##################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image).copy()
        
        # Use the input image, and apply mask from fits file extension
        # any other option (overscan, hot columns, ... will be ignored)
        if self.apply_mask_from_fits[0]:
            if self.apply_mask_from_fits[1] == "same":
                print(f"  Appply extension {self.apply_mask_from_fits[2]} from {rawdata.file_name} as mask" )
                mask = fits.getdata(rawdata.file_name, ext=self.apply_mask_from_fits[2]).astype(bool)
            else:
                print(f"  Appply extension {self.apply_mask_from_fits[2]} from {self.apply_mask_from_fits[1].split('/')[-1]} as mask" )
                mask = fits.getdata(self.apply_mask_from_fits[1],ext=self.apply_mask_from_fits[2]).astype(bool)

            # MASK THE GIVEN SET OF COLUMNS/ROWS
            #  this is gien by amplifier
            ##################################################################################
            _rows_to_mask = self.rows_to_mask[rawdata.execute_process_in_amp] if type(self.rows_to_mask)==dict else self.rows_to_mask
            for r in _rows_to_mask:
                if type(r)==list and len(r)==2:
                    mask[r[0]:r[1],:] = True
                else:
                    mask[r,:] = True

            _cols_to_mask = self.cols_to_mask[rawdata.execute_process_in_amp] if type(self.cols_to_mask)==dict else self.cols_to_mask
            for c in _cols_to_mask:
                if type(c)==list and len(c)==2:
                    mask[:,c[0]:c[1]] = True
                else:
                    mask[:,c] = True

            return np.ma.array(image_amp.data, mask=mask).compressed()

        # select between sensitive region or overscan
        ##################################################################################
        if self.in_overscan:
            image = np.ma.array(image_amp, mask=rawdata.mask_image_overscan_cols).copy()
        else:
            image = np.ma.array(image_amp, mask=rawdata.mask_image_active_region).copy()
            ### XXX by default mask column 0 and 1
            #image.mask[:,0] = True
            #image.mask[:,1] = True
            # MASK CLUSTERS ONLY HAS SENSE IF THE ACTIVE REGION IS SELECTED
            ##############################################################################
            if self.mask_clusters:
                # keep track of the non-masked pixels
                self._pcd_no_masked = image.compressed()
                if not hasattr(rawdata,'mask_clusters'):
                    raise AttributeError("To use this option 'mask_clusters' run also ClusterFinder and BuildClusterMask")
                # joing both masks
                image = np.ma.array(image.data, mask=np.logical_or(image.mask,rawdata.mask_clusters)).copy()
   
        # MASK THE GIVEN SET OF COLUMNS/ROWS
        #  this is gien by amplifier
        ##################################################################################
        _rows_to_mask = self.rows_to_mask[rawdata.execute_process_in_amp] if type(self.rows_to_mask)==dict else self.rows_to_mask
        for r in _rows_to_mask:
            if rawdata.ACM:
                ### rows are joined along rows, user use local coordinates for the different CCDs, 
                #       and offset is needed to properly mask the correct row
                indx = rawdata.ampl.index(rawdata.execute_process_in_amp)
                offset = indx*rawdata.ACM_amp_rows
            else:
                offset = 0

            if type(r)==list and len(r)==2:
                image.mask[r[0]+offset:r[1]+offset,:] = True
            else:
                image.mask[r+offset,:] = True

        _cols_to_mask = self.cols_to_mask[rawdata.execute_process_in_amp] if type(self.cols_to_mask)==dict else self.cols_to_mask
        for c in _cols_to_mask:
            if type(c)==list and len(c)==2:
                image.mask[:,c[0]:c[1]] = True
            else:
                image.mask[:,c] = True

        if self.__DEBUG__:
            self.execute_debug_plot(rawdata,image)
        
        self.total_pixels = image.shape[0]*image.shape[1]
        return image.compressed()
    
    def execute_debug_plot(self,rawdata,image):

        ROOT.gROOT.SetBatch(0)
        if self.calibration is not None:
            neg_rows = np.ma.where(image<-self.calibration/2.0)[0]
        else:
            neg_rows = np.ma.where(image<-8.0)[0]
        
        try:
            thr = ROOT.TH1F("hrow_{}".format(rawdata.execute_process_in_amp),"position of pixels with Q<-{} ADUs".format(int(self.calibration/2.0)),                    
                    int(max(neg_rows)-min(neg_rows)), min(neg_rows),max(neg_rows))
        except ValueError:
            ROOT.gROOT.SetBatch(1)
            return

        for r in neg_rows:
            thr.Fill(r)
        c = ROOT.TCanvas()
        thr.GetXaxis().SetTitle("Rows")
        thr.GetYaxis().SetTitle("Entries")
        thr.Draw("HIST")
        c.Update()
        c.Draw()
        input("press enter ... ")

        if self.calibration is not None:
            neg_cols = np.ma.where(image<-self.calibration/2.0)[1]
        else:
            neg_cols = np.ma.where(image<-8.0)[1]
        
        thc = ROOT.TH1F("hcol_{}".format(rawdata.execute_process_in_amp),"position of pixels with Q<-{} ADUs".format(int(self.calibration/2.0)),
                int(max(neg_cols)-min(neg_cols)), min(neg_cols),max(neg_cols))
        for r in neg_cols:
            thc.Fill(r)
        c = ROOT.TCanvas()
        thc.GetXaxis().SetTitle("Columns")
        thc.GetYaxis().SetTitle("Entries")
        thc.Draw("HIST")
        c.Update()
        c.Draw()
        input("press enter ... ")
        ROOT.gROOT.SetBatch(1)


class FitDarkCurrentPerRow(SKImageProcess):
    """Fits the pixel charge distribution 

    """
    __sequence_id__ = 31
    __name__ = 'FitDarkCurrentPerRow'
    
    def __init__(self):
        """
        """
        super().__init__()
        self.DEBUG_ROWS = None
        self.row_start = 0
        self.row_end   = -1
        self.row_step  = 10

        self.rows_to_mask = []
        self.cols_to_mask = []

        ### The DC can be estimated by using the pedestal_subtracted image or the mean_compressed one
        self.image = "mean_compressed_pedestal_subtracted"

        ### python method needs the number of peaks to be plotted
        self.n_peaks = 2
        self.binning_size = -1
        self.calibration = 10.0

        self.mask_clusters = False
        self.do_calibration = False

        ### ADD an extra linear component to the DC
        self.x_min = np.nan
        self.x_max = np.nan

        ### LIMITS FOR THE PARAMETER SPACE
        self.mu_min,self.mu_max         = -1,1
        self.sigma_min,self.sigma_max   = 0,100
        self.dc_min,self.dc_max         = 0.000001,500
        self.gain_min,self.gain_max     = 1.0,500

        # add variable to read data from fits file, and apply mask
        # flag, fits file name, extension
        self.add_mask = [False,None,None]
        
        self.q_threshold = None
        ### 
        self.__units__.update({
            'row_start':1, 'row_end':1, 'row_step':1,
            'image':1, 'n_peaks':1,'calibration':1,'binning_size':1,
            'x_min':u.ADC,'x_max':u.ADC,
            'mu_min':1,'mu_max':1,'sigma_min':1,'sigma_max':1,'dc_min':1,'dc_max':1,'gain_min':1,'gain_max':1,
            'mask_clusters':1,'rows_to_mask':1, 'cols_to_mask':1,'do_calibration':1,
            'add_mask':1, 'q_threshold':1, 'DEBUG_ROWS':1
            })
    
    def execute_process(self,rawdata,**kwargs):
        """
        """
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)
       
        # THIS IS A SILENT MODE TO RUN AS BATTTTCH
        if not self.__silent__:
            print("Process <FitDarkCurrentPerRow> INFO. Fit the n-first peaks to estimate the dark current and the calibration constant {}.".format(amplifier))
           
        # PIXEL CHARGE DISTRIBUTION AS AN ARRAY OR AS PIXELS FROM AN IMAGE
        #### GET IMAGE AND APPLY MASK (OVERSCAN, ACTIVE REGION, OR USING THE CLUSTER MASK)
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image).copy()
        # only sensitive region
        image = np.ma.array(image_amp, mask=rawdata.mask_image_active_region).copy()
        
        if not self.add_mask[0]:
            #### masking rows and columns
            _rows_to_mask = self.rows_to_mask[rawdata.execute_process_in_amp] if type(self.rows_to_mask)==dict else self.rows_to_mask
            for r in _rows_to_mask:
                if rawdata.ACM:
                    ### rows are joined along rows, user use local coordinates for the different CCDs, 
                    #       and offset is needed to properly mask the correct row
                    indx = rawdata.ampl.index(rawdata.execute_process_in_amp)
                    offset = indx*rawdata.ACM_amp_rows
                else:
                    offset = 0

                if type(r)==list and len(r)==2:
                    image.mask[r[0]+offset:r[1]+offset,:] = True
                else:
                    image.mask[r+offset,:] = True

            _cols_to_mask = self.cols_to_mask[rawdata.execute_process_in_amp] if type(self.cols_to_mask)==dict else self.cols_to_mask
            for c in _cols_to_mask:
                if type(c)==list and len(c)==2:
                    image.mask[:,c[0]:c[1]] = True
                else:
                    image.mask[:,c] = True
        else:
            print(f"File {self.add_mask[1]} (ext. {self.add_mask[2]}) is used as mask")
            image.mask = fits.getdata(self.add_mask[1],ext=self.add_mask[2]).astype(bool)


        # set row region
        self.row_end = self.row_end if self.row_end>0 else self.row_end+image.shape[0]

        # function to get the error of function f = p[0]/p[2] with p[0]+/-p[1], p[2]+/-p[3]
        get_error = lambda p: ((p[1]/p[2])**2 + ((p[0]*p[3]/p[2]**2))**2)**0.5
        # to convert DC to e-/pix/img
        # convfact = (60*60*24) / (rawdata.bin_col*rawdata.bin_row) / (rawdata.exposure_time+rawdata.read_time/2.)
        convfact = 1 / (rawdata.bin_col*rawdata.bin_row)
        
        results = {'xrows':[[],[]], 'dc':[[],[]], 'mu':[[],[]],'sigma':[[],[]],'calibration':[[],[]],'chi2':[], 'has_highE_pix':[[],[]]}


        for row in range(self.row_start, self.row_end, self.row_step):
            sliced_rows = image[slice(row,row+self.row_step),:].compressed()    
            if len(sliced_rows)==0:
                continue

            dcfit = FitDarkCurrentProcess()
            # set all parameters related to the fit
            for par in ['n_peaks','calibration', 'x_min', 'x_max', 'mu_min', 'mu_max', 'sigma_min','sigma_max', 'dc_min',
                    'dc_max','gain_min','gain_max','binning_size','mask_clusters','do_calibration']:
                setattr(dcfit,par,getattr(self,par))
            dcfit.__silent__ = True
            if self.DEBUG_ROWS is not None:
                dcfit.__display__ = True if row in range(*self.DEBUG_ROWS) else False
            else:
                # plot single DCFits
                dcfit.__display__ = self.__DEBUG__

            if len(sliced_rows)==0 or len(sliced_rows[(sliced_rows>dcfit.x_min) & (sliced_rows<dcfit.x_max)])==0:
                continue
            
            dcfit.execute_process(rawdata, **{'data':sliced_rows})
            
            results['mu'][0].append(dcfit.dc_mu0/dcfit.dc_gain)
            results['sigma'][0].append(dcfit.dc_sigma/dcfit.dc_gain)
            results['dc'][0].append(dcfit.dc_lambda/dcfit.dc_gain * convfact)
            results['calibration'][0].append(dcfit.dc_gain)
            ## FIT :: AND ERRORS
            results['mu'][1].append(get_error([dcfit.dc_mu0,dcfit.dc_mu0_err,dcfit.dc_gain,dcfit.dc_gain_err]))
            results['sigma'][1].append(get_error([dcfit.dc_sigma,dcfit.dc_sigma_err,dcfit.dc_gain,dcfit.dc_gain_err]))
            results['dc'][1].append(get_error([dcfit.dc_lambda, dcfit.dc_lambda_err,dcfit.dc_gain,dcfit.dc_gain_err])*convfact)
            results['calibration'][1].append(dcfit.dc_gain_err)
            
            results['xrows'][0].append(row + self.row_step/2. )
            results['xrows'][1].append(self.row_step/2.)

            if self.q_threshold is not None:
                results['has_highE_pix'][1].append( np.any(sliced_rows>self.q_threshold) )
                results['has_highE_pix'][0].append( np.sum(sliced_rows>self.q_threshold) )
            
            try:
                results['chi2'].append( dcfit.fitfunc.GetChisquare()/dcfit.fitfunc.GetNDF() )
            except ZeroDivisionError:
                results['chi2'].append( -999 )

        
        if self.save_plots or self.__display__:
            self.draw_results(rawdata,results)
            self.draw_results_as_hist(rawdata,results)
        
        if 'results' in kwargs:
            return results


    def draw_results_as_hist(self,rawdata,results):
        
        _isbatch = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(0)

        ystring = {'mu':'mu_e','sigma':'sigma_e','dc':'lambda_e_pix_img','calibration':'gain_ADU_e','chi2':'chis2',
                'has_highE_pix':'has_highE_pix'}
        ylabel = {'mu':'zero-e^{-} peak position (e^{-})','sigma':'Resolution (e^{-})',
                'dc':'Dark current (e^{-}/pix/img)','calibration':'Calibratoin (ADU/e^{-})', 
                'has_highE_pix':f"N(q) > {self.q_threshold}"}
        
        Nbins = lambda x: int(1 + np.ceil(np.log2(len(x))))

        c,thist,ff = [],[],[]
        for par in results.keys():
            if par in ['xrows','chi2']:
                continue
            if par=='has_highE_pix' and self.q_threshold is None:
                continue
            c.append(ROOT.TCanvas(f"{par}_xrows",f"{par}_xrows"))
            c[-1].cd()
            
            thist.append(ROOT.TH1D(f"h{par}_xrows",f"h{par}_xrows",Nbins(results[par][0]),min(results[par][0]),max(results[par][0])))
            for yi in results[par][0]:
                _ = thist[-1].Fill(yi)
            
            ff.append( ROOT.TF1(f"fitfunc_{par}_xrows","gaus",min(results[par][0]),max(results[par][0])) )
            ff[-1].SetLineColor(2)

            thist[-1].Fit(ff[-1],'Q EM')
            thist[-1].SetTitle(f"amplifier {rawdata.execute_process_in_amp};{ylabel[par]};counts")
            thist[-1].SetDirectory(0)
            thist[-1].Draw()
            
            c[-1].Update()

        input("press enter ...")
        # set display mode to the one by default
        ROOT.gROOT.SetBatch(_isbatch)
        return


    def draw_results(self,rawdata,results):
        
        _isbatch = ROOT.gROOT.IsBatch()
        if self.__display__:
            ROOT.gROOT.SetBatch(0)
        if self.save_plots and not self.__display__:
            ROOT.gROOT.SetBatch(1)

        ystring = {'mu':'mu_e','sigma':'sigma_e','dc':'lambda_e_pix_img','calibration':'gain_ADU_e','chi2':'chis2',
                'has_highE_pix':'has_highE_pix'}
        ylabel = {'mu':'zero-e^{-} peak position (e^{-})','sigma':'Resolution (e^{-})',
                'dc':'Dark current (e^{-}/pix/img)','calibration':'Calibratoin (ADU/e^{-})', 
                'has_highE_pix':f"N(q) > {self.q_threshold}"}
        
        ylabel_csv = {
                'mu':'mu',
                'sigma':'resolution',
                'dc':'dc',
                'calibration':'gain'
                }

        # recreate the file before updating it, only if forat is root
        if self.save_plots and "root" in self.format_figures:
            outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}_all_params_vsRows.root","RECREATE")
            outrootfile.Close()
        
        c,tg,fitfunc,tqth = [],[],[],[]
        results_to_csv = {}
        for par in results.keys():
            if par in ['xrows','chi2']:
                continue
            if par=='has_highE_pix' and self.q_threshold is None:
                continue
            c.append(ROOT.TCanvas(f"{par}_xrows",f"{par}_xrows"))
            c[-1].cd()
            tg.append(ROOT.TGraphErrors( 
                    int(len(results['xrows'][0])),
                    array('d',results['xrows'][0]),
                    array('d',results[par][0]),
                    array('d',results['xrows'][1]),
                    array('d',results[par][1])
                    ))
            
            results_to_csv[f"x_{ylabel_csv[par]}"] = results['xrows'][0]
            results_to_csv[f"{ylabel_csv[par]}"] = results[par][0]
            results_to_csv[f"err_x_{ylabel_csv[par]}"] = results['xrows'][1]
            results_to_csv[f"err_{ylabel_csv[par]}"] = results[par][1]

            fitfunc.append(ROOT.TF1(f"fitfunc_{par}",'pol1', 
                    min(results['xrows'][0])-results['xrows'][1][0], 
                    max(results['xrows'][0])-results['xrows'][1][0]))
            fitfunc[-1].SetLineColor(ROOT.kRed-3)
            fitfunc[-1].SetLineWidth(3)
            fitfunc[-1].SetLineStyle(9)
            tg[-1].Fit(fitfunc[-1],"QREL0")
            
            tg[-1].SetName(ystring[par])
            tg[-1].SetTitle(f"amplifier {rawdata.execute_process_in_amp};sliced rows;{ylabel[par]}")
            tg[-1].Draw("A PEZ")
            if par!='has_highE_pix':
                fitfunc[-1].Draw("L same")
            c[-1].Update()

            if self.q_threshold is not None and par!='has_highE_pix':
                mask = np.array(results['has_highE_pix'][1]).astype(bool)
                if mask.sum()>0:
                    c[-1].cd()
                    tqth.append( ROOT.TGraphErrors(
                            int(mask.sum()),
                            array('d',np.array(results['xrows'][0])[mask]),
                            array('d',np.array(results[par][0])[mask]),
                            array('d',np.array(results['xrows'][1])[mask]),
                            array('d',np.array(results[par][1])[mask])
                            ))
                    tqth[-1].SetLineColor(ROOT.kRed+1)
                    tqth[-1].SetMarkerColor(ROOT.kRed+1)
                    tqth[-1].Draw("PE SAME")
                    c[-1].Update()

            if self.save_plots:
                for fmt in self.format_figures:
                    if fmt=="root":
                        outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}_all_params_vsRows.root","UPDATE")
                        tg[-1].Write()
                        outrootfile.Close()
                    else:
                        fname = rawdata.output+"{}_plot_{}vsRows_{}.{}".format(self.__name__,par,rawdata.execute_process_in_amp,fmt)
                        c[-1].SaveAs(fname)

            #ROOT.gROOT.GetListOfCanvases().Delete()

        if "csv" in self.format_figures:
            outcsvfile = f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}.csv"
            print(f"   -- Results from fit save as {outcsvfile}")
            df = pd.DataFrame(results_to_csv)
            df.to_csv(outcsvfile, index=False)

        if self.q_threshold is not None:
            print(f"In RED those axis that has at least 1 pixel along the [binned]-axis with q>{self.q_threshold} (in input image units)")

        if self.__display__:
            input("press enter ...")

        # set display mode to the one by default
        ROOT.gROOT.SetBatch(_isbatch)
        return

class FitDarkCurrentPerCol(SKImageProcess):
    """Fits the pixel charge distribution 

    """
    __sequence_id__ = 32
    __name__ = 'FitDarkCurrentPerCol'
    
    def __init__(self):
        """
        """
        super().__init__()
        self._per_amp = False 
        self.__silent__ = False
        self.DEBUG_COLS = None        

        self.col_start = 0
        self.col_end   = -1
        self.col_step  = 10

        self.rows_to_mask = []
        self.cols_to_mask = []

        ### The DC can be estimated by using the pedestal_subtracted image or the mean_compressed one
        self.image = "mean_compressed_pedestal_subtracted"

        ### python method needs the number of peaks to be plotted
        self.n_peaks = 2
        self.binning_size = -1
        self.calibration = 10.0

        ### ADD an extra linear component to the DC
        self.x_min = np.nan
        self.x_max = np.nan

        ### LIMITS FOR THE PARAMETER SPACE
        self.mu_min,self.mu_max         = -1,1
        self.sigma_min,self.sigma_max   = 0,100
        self.dc_min,self.dc_max         = 0.000001,500
        self.gain_min,self.gain_max     = 1.0,500
        
        # add variable to read data from fits file, and apply mask
        # flag, fits file name, extension
        self.add_mask = [False,None,None]
        
        # degree of the polynomial to fit the dark current
        self.degree_pol = 1

        ### Fit option
        self.fit_options = "QSMEFR"

        # to save results as csv
        self.save_results = False

        self.q_threshold = None
        ### 
        self.__units__.update({
            'col_start':1, 'col_end':1, 'col_step':1,
            'image':1, 'n_peaks':1,'calibration':1,'binning_size':1,
            'x_min':u.ADC,'x_max':u.ADC,
            'mu_min':1,'mu_max':1,'sigma_min':1,'sigma_max':1,'dc_min':1,'dc_max':1,'gain_min':1,'gain_max':1,
            'mask_clusters':1,'rows_to_mask':1, 'cols_to_mask':1,
            'add_mask':1, 'degree_pol':1, 'save_results':1, 'fit_options':1,
             'q_threshold':1, 'DEBUG_COLS':1
            })
    
    def execute_process(self,rawdata,**kwargs):
        """
        """
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)
       
        # THIS IS A SILENT MODE TO RUN AS BATTTTCH
        if not self.__silent__:
            print("Process <FitDarkCurrentPerCol> INFO. Fit the n-first peaks to estimate the dark current and the calibration constant {}.".format(amplifier))
           
        # PIXEL CHARGE DISTRIBUTION AS AN ARRAY OR AS PIXELS FROM AN IMAGE
        #### GET IMAGE AND APPLY MASK (OVERSCAN, ACTIVE REGION, OR USING THE CLUSTER MASK)
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image).copy()
        # full region
        image = np.ma.array(image_amp, mask=np.zeros_like(image_amp).astype(bool)).copy()
        
        if not self.add_mask[0]:
            #### masking rows and columns
            _rows_to_mask = self.rows_to_mask[rawdata.execute_process_in_amp] if type(self.rows_to_mask)==dict else self.rows_to_mask
            for r in _rows_to_mask:
                if rawdata.ACM:
                    ### rows are joined along rows, user use local coordinates for the different CCDs, 
                    #       and offset is needed to properly mask the correct row
                    indx = rawdata.ampl.index(rawdata.execute_process_in_amp)
                    offset = indx*rawdata.ACM_amp_rows
                else:
                    offset = 0
    
                if type(r)==list and len(r)==2:
                    image.mask[r[0]+offset:r[1]+offset,:] = True
                else:
                    image.mask[r+offset,:] = True

            #### masking columns
            _cols_to_mask = self.cols_to_mask[rawdata.execute_process_in_amp] if type(self.cols_to_mask)==dict else self.cols_to_mask
            for c in _cols_to_mask:
                if type(c)==list and len(c)==2:
                    image.mask[:,c[0]:c[1]] = True
                else:
                    image.mask[:,c] = True
        else:
            image.mask = fits.getdata(self.add_mask[1],ext=self.add_mask[2]).astype(bool)

        # set row region
        self.col_end = self.col_end if self.col_end>0 else self.col_end+image.shape[1]

        # function to get the error of function f = p[0]/p[2] with p[0]+/-p[1], p[2]+/-p[3]
        get_error = lambda p: ((p[1]/p[2])**2 + ((p[0]*p[3]/p[2]**2))**2)**0.5
        # to convert DC to e-/pix/img
        # convfact = (60*60*24) / (rawdata.bin_col*rawdata.bin_row) / (rawdata.exposure_time+rawdata.read_time/2.)
        convfact = 1 / (rawdata.bin_col*rawdata.bin_row)
        
        results = {'xcols':[[],[]], 'dc':[[],[]], 'mu':[[],[]],'sigma':[[],[]],'calibration':[[],[]], 'chi2':[],'has_highE_pix':[[],[]]}
        
        for col in range(self.col_start, self.col_end, self.col_step):
            sliced_cols = image[:,slice(col,col+self.col_step)].compressed()    
            if len(sliced_cols)==0:
                continue

            dcfit = FitDarkCurrentProcess()
            # set all parameters related to the fit
            for par in ['n_peaks', 'calibration', 'x_min', 'x_max', 'mu_min', 'mu_max', 'sigma_min','sigma_max', 'dc_min',
                    'dc_max', 'gain_min', 'gain_max','binning_size','fit_options']:
                setattr(dcfit, par, getattr(self,par))
            dcfit.__silent__ = True
            # plot single DCFits
            if self.DEBUG_COLS is not None:
                dcfit.__display__ = True if col in range(*self.DEBUG_COLS) else False
            else:
                # plot single DCFits
                dcfit.__display__ = self.__DEBUG__

            if len(sliced_cols)==0 or len(sliced_cols[(sliced_cols>dcfit.x_min) & (sliced_cols<dcfit.x_max)])==0:
                continue
            dcfit.execute_process(rawdata, **{'data':sliced_cols})
            
            results['mu'][0].append(dcfit.dc_mu0/dcfit.dc_gain)
            results['sigma'][0].append(dcfit.dc_sigma/dcfit.dc_gain)
            results['dc'][0].append(dcfit.dc_lambda/dcfit.dc_gain * convfact)
            results['calibration'][0].append(dcfit.dc_gain)
            ## FIT :: AND ERRORS
            results['mu'][1].append(get_error([dcfit.dc_mu0,dcfit.dc_mu0_err,dcfit.dc_gain,dcfit.dc_gain_err]))
            results['sigma'][1].append(get_error([dcfit.dc_sigma,dcfit.dc_sigma_err,dcfit.dc_gain,dcfit.dc_gain_err]))
            results['dc'][1].append(get_error([dcfit.dc_lambda, dcfit.dc_lambda_err,dcfit.dc_gain,dcfit.dc_gain_err])*convfact)
            results['calibration'][1].append(dcfit.dc_gain_err)
            
            results['xcols'][0].append(col + self.col_step/2. )
            results['xcols'][1].append(self.col_step/2.)
            if self.q_threshold is not None:
                results['has_highE_pix'][1].append( np.any(sliced_cols>self.q_threshold) )
                results['has_highE_pix'][0].append( np.sum(sliced_cols>self.q_threshold) )

            
            try:
                results['chi2'].append( dcfit.fitfunc.GetChisquare()/dcfit.fitfunc.GetNDF() )
            except ZeroDivisionError:
                self.chi2 = -999
        
        if self.save_plots or self.__display__:
             self.draw_results(rawdata,results)

        if self.save_results:
            import pandas as pd
            _r = pd.DataFrame.from_dict({
                'cols'  : list(results['xcols'][0]),
                'ecols' : list(results['xcols'][1]),
                'noise' : list(results['sigma'][0]),
                'enoise': list(results['sigma'][1]),
                'gain'  : list(results['calibration'][0]),
                'egain' : list(results['calibration'][1]),
                'dc'    : list(results['dc'][0]),
                'edc'   : list(results['dc'][1]),
                'chi2'  : list(results['chi2'][0])
                })
            _r.to_csv(rawdata.output+"{}_DCvsCOLS_results_amp{}.csv".format(self.__name__,rawdata.execute_process_in_amp),index=False)

        if 'results' in kwargs:
            return results

    def draw_results(self,rawdata,results):
        
        _isbatch = ROOT.gROOT.IsBatch()
        if self.__display__:
            ROOT.gROOT.SetBatch(0)
        if self.save_plots and not self.__display__:
            ROOT.gROOT.SetBatch(1)

        
        ystring = {'mu':'mu_e','sigma':'sigma_e','dc':'lambda_e_pix_img','calibration':'gain_ADU_e','chi2':'chis2',
                'has_highE_pix':'has_highE_pix'}
        ylabel = {'mu':'zero-e^{-} peak position (e^{-})','sigma':'Resolution (e^{-})',
                'dc':'Dark current (e^{-}/pix/img)','calibration':'Calibratoin (ADU/e^{-})', 'chi2':'chi^{2}_{red}',
                'has_highE_pix':f"N(q) > {self.q_threshold}"}

        # recreate the file before updating it, only if forat is root
        if self.save_plots and "root" in self.format_figures:
            outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}_all_params_vsCols.root","RECREATE")
            outrootfile.Close()


        c,tg,fitfunc,tqth,thist = [],[],[],[],[]
        for par in results.keys():
            if par in ['xcols']:
                continue
            if par=='has_highE_pix' and self.q_threshold is None:
                continue
            if par in ['chi2']:
                results['chi2'] = [results['chi2'],[0]*len(results['chi2'])]
            
            c.append(ROOT.TCanvas(f"{par}_xrows",f"{par}_xrows"))
            c[-1].cd()

            tg.append(ROOT.TGraphErrors( 
                    int(len(results['xcols'][0])),
                    array('d',results['xcols'][0]),
                    array('d',results[par][0]),
                    array('d',results['xcols'][1]),
                    array('d',results[par][1])
                    ))

            fitfunc.append( ROOT.TF1(f"fitfunc_{par}",'pol{}'.format(self.degree_pol), 
                min(results['xcols'][0])-results['xcols'][1][0], max(results['xcols'][0])-results['xcols'][1][0]))
            fitfunc[-1].SetLineColor(ROOT.kMagenta-5)
            fitfunc[-1].SetLineWidth(3)
            fitfunc[-1].SetLineStyle(9)
            tg[-1].Fit(fitfunc[-1],"QREL0")

            tg[-1].SetName(ystring[par])
            tg[-1].SetTitle(f"amplifier {rawdata.execute_process_in_amp};sliced cols;{ylabel[par]}")
            tg[-1].Draw("A PEZ")
            if par != 'has_highE_pix':
                fitfunc[-1].Draw("L same")
            c[-1].Update()

            if self.q_threshold is not None and par!='has_highE_pix':
                mask = np.array(results['has_highE_pix'][1]).astype(bool)
                if mask.sum()>0:
                    c[-1].cd()
                    tqth.append( ROOT.TGraphErrors(
                        int(mask.sum()),
                        array('d',np.array(results['xcols'][0])[mask]),
                        array('d',np.array(results[par][0])[mask]),
                        array('d',np.array(results['xcols'][1])[mask]),
                        array('d',np.array(results[par][1])[mask])
                        ))
                    tqth[-1].SetLineColor(ROOT.kRed+1)
                    tqth[-1].SetMarkerColor(ROOT.kRed+1)
                    tqth[-1].Draw("PE SAME")
                    c[-1].Update()

            if self.save_plots:
                for fmt in self.format_figures:
                    if fmt=="root":
                        outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_side{rawdata.execute_process_in_amp}_all_params_vsCols.root","UPDATE")
                        tg[-1].Write()
                        outrootfile.Close()
                    else:
                        fname = rawdata.output+"{}_plot_{}vsCols_{}.{}".format(self.__name__,par,rawdata.execute_process_in_amp,fmt)
                        c[-1].SaveAs(fname)
            
        if self.q_threshold is not None:
            print(f"In RED those axis that has at least 1 pixel along the [binned]-axis with q>{self.q_threshold} (in input image units)")


        if self.__display__:
            input("press enter ...")

        ROOT.gROOT.GetListOfCanvases().Delete()
        # set display mode to the one by default
        ROOT.gROOT.SetBatch(_isbatch)

class CorrectElectronicColumnTransient(SKImageProcess):
    """The first columns transient is corrected by fitting the median per row to an exponential
    decay and subtracting the fit.
    If the correct_image option is True it will subtract the fitted model to the data. 
    It will add _correct_cols to the image id.
    If is False. It will only do the fit and mantain the previous image id.
    """
    __sequence_id__ = 31
    __name__ = 'CorrectElectronicColumnTransient'

    def __init__(self):
        """
        """
        super().__init__()

        self.image = "mean_compressed_pedestal_subtracted"
        self.col_start = 1
        self.col_end   = 50
        self.subtract_median = True

        ### Decaying parameters
        self.n_exp = 1
        self.mu = [0.12]
        self.amplitude = [0.013]

        ### Fit option
        self.fit_options = "QSMEFR"
        
        self.correct_image = True 
        ### 
        self.__units__.update({'col_start':u.pixel,'col_end':u.pixel, 
            'subtract_median':1, 'n_exp':1, 
            'mu':1, 'amplitude':1, 'fit_options':1, 'correct_image':1})
    
    def execute_process(self,rawdata):
        """
        """
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)

        print("Process <CorrectElectronicColumnTransient> INFO. Corrects the electronic transient {}".format(amplifier))
        print("   - Using column region: {}-{}".format(self.col_start,self.col_end))

        #### GET IMAGE 
        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image)
        
        _is_amp_L = self.define_column_region(rawdata)

        ### Calculate pixel charge median and MAD of columns
        image_col_median = np.ma.median(image_amp, axis=0)
        if self.subtract_median:
            if _is_amp_L:
                image_col_median = np.ma.subtract(image_col_median,np.ma.median(image_col_median[:self.col_start]))
            else:
                image_col_median = np.ma.subtract(image_col_median,np.ma.median(image_col_median[self.col_end:]))
        image_col_MAD = np.ma.median(np.ma.abs(image_amp-np.ma.median(image_amp)), axis=0)
        
        ### Creates the TGraph
        n_cols_list = np.linspace(self.col_start,self.col_end,self.col_end)
        n_cols = len(n_cols_list)
        col_median_graph = ROOT.TGraph(
                n_cols,
                array('d',n_cols_list),
                array('d',image_col_median.compressed().tolist())
                )

        ### Fitting Formula according to the number of exponentials to be fitted
        func_formula = []
        sign = 1.0 if _is_amp_L else -1.0
        for i in range(self.n_exp):
            func_formula.append( "[{0}]*TMath::Exp(-[{1}]*x)".format(2*i, 1+2*i))
        func_formula = "+".join(func_formula)

        ### Define the fitting function in the column range to be applied
        fitfunc = ROOT.TF1("fitfunc", func_formula, self.col_start, self.col_end)
        fitfunc.SetLineColor(2)
        self.set_fit_pars(fitfunc)

        ## Checkout if range it's in Fit options
        if not "R" in self.fit_options:
            self.fit_options = self.fit_options+"R"
        f0 = col_median_graph.Fit(fitfunc, self.fit_options)
        ### Get the fitted parameters
        self.mu_fit, self.amplitude_fit = self.get_fit_pars(fitfunc) 
        ## Gets the fitted model
        image_median_fit_model = self.get_fitted_model(fitfunc)
        
        # append best fit function
        self.fitfunc = fitfunc

        #### OUTPUTS
        ######################################################################################################
        results = { 'tgraph':col_median_graph, 
                'mu_fit':self.get_fit_pars(fitfunc)[0], 'amplitude_fit':self.get_fit_pars(fitfunc)[1],
                'mu_err_fit':self.get_fit_pars_err(fitfunc)[0], 'amplitude_err_fit':self.get_fit_pars_err(fitfunc)[1],
                'ect_mu0':self.mu_fit[0],'ect_A0':self.amplitude_fit[0],'ect_chi2':fitfunc.GetChisquare(),
                'colTransfit':self }
        
        img_corr = image_amp.copy()
        img_corr[:,self.col_start:self.col_end] = image_amp[:,self.col_start:self.col_end] - image_median_fit_model
        rawdata.set_image_by_attribute_name("{}_correct_cols".format(itype),img_corr,**results) 
        
        # RESULTS
        ### Append best fit parameters: mu and amplitud, and tgraph
        ### record the fitted electronic transient effect
        output = rawdata.output+"{}_ETfit.eps".format(self.__name__)
        if self.__verbose__:
            self.draw_fit(output,results['tgraph'])

    def define_column_region(self,rawdata):

        ### DATA CAN BE READ WITH MORE THAN ONE AMPLIFIER, AND THE NUMBER OF ROWS ARE RELATIVE TO THE AMPLIFIER USED
        _is_amp_L = False
        if hasattr(rawdata,'execute_process_in_amp') and rawdata.n_amp==2:
            if rawdata.execute_process_in_amp == 'U':
                ### when readout is done by amplifier U, the last rows are the first to be read out
                nrows,ncols = image_amp.shape
                col_start = self.col_start
                col_end   = self.col_end
                self.col_start = nrows - col_end
                self.col_end   = -3 
                _is_amp_L = True

        return _is_amp_L

    def get_ECTfit_data(self):

        ### data used to fit
        x = np.array(self.tgraph.GetX()).tolist()
        y = np.array(self.tgraph.GetY()).tolist()

        y_fit = []
        for xi in x:
            y_fit.append(self.fitfunc(xi))

        return x,y,y_fit


    def set_fit_pars(self, fit_func):
        """Sets the initial parameters and name
        """
        for i in range(self.n_exp):
            ### Set Amplitude 
            fit_func.SetParameter(2*i, self.amplitude[i])
            fit_func.SetParName(2*i, "A_{"+str(i+1)+"}")
            #   only possitive
            fit_func.SetParLimits(2*i, 0, 50)
            ### Set Decay Constant
            fit_func.SetParameter(2*i+1, self.mu[i])
            fit_func.SetParName(2*i+1, "#mu_{"+str(i+1)+"}")
            #   defined as postive
            fit_func.SetParLimits(2*i+1, 0, 50)
            
    def get_fit_pars(self, fit_func):
        """Gets the parameters obtained from the fit
        """
        mu, amplitude = [], []
        for i in range(self.n_exp):
            ### Amplitude
            amplitude.append( fit_func.GetParameter(2*i) )
            ### Decay Constant
            mu.append(fit_func.GetParameter(2*i+1))
        return mu, amplitude

    def get_fit_pars_err(self, fit_func):
        """Gets the parameters errors obtained from the fit
        """
        mu_err, amplitude_err = [], []
        for i in range(self.n_exp):
            ### Amplitude
            amplitude_err.append( fit_func.GetParError(2*i) )
            ### Decay Constant
            mu_err.append(fit_func.GetParError(2*i+1))
        return mu_err, amplitude_err


    def get_fitted_model(self, fitfunc):
        """ Creates the numpy array of the fitted model
        """
        y = np.zeros([self.col_end-self.col_start])
        for i in range(len(y)):
            y[i] = fitfunc.Eval(i+self.col_start)

        return y

    def draw_fit(self,output,tgraph,amp='U'):
        
        if self.__verbose__:    
            ##### PLOTTING 
            ROOT.gROOT.SetBatch(0)

        can = ROOT.TCanvas("final fit")
        can.Clear()

        ### Plotting
        hist = tgraph.GetHistogram()
        hist.GetXaxis().SetTitle(" column number ")
        hist.GetYaxis().SetTitle("(electronic-tail) baseline [ADU]")

        tgraph.Draw("AP")
        can.Update()
        
        can.SaveAs(output)

        if self.__verbose__:
            can.Draw()
            input("Press Enter ........ ")
    
        return

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#
#                   FOR SIMULATED IMAGES ONLY
#
#################################################################################################
#                                                                                               #
#       Process Class to ADD an extra source of BKG IN SIMULATED IMAGES                         #
#                                                                                               #
#################################################################################################

class AddBKGComponent(SKImageProcess):
    """
    """
    __name__ = 'AddBKGComponent'
    __sequence_id__ = 15

    def __init__(self):
        super().__init__()

        self.image = "mean_compressed"
        
        # parameters of the model
        self.function = 'uniform'
        # value depends on function
        self.value = 0
        
        # parameters of the model
        self.params = [np.nan,np.nan]

        # if linear
        self.Ncls = None
        self.Emin = 0.04
        self.Emax = 0.15
        
        self.__units__.update({'function':1,'value':u.ADC,'image':1,'Ncls':1,'Emin':1,'Emax':1})
        
    def execute_process(self,rawdata):
        """
        """
        print("Process <AddDCComponent> INFO. Add an extra souce of DC acting only in the active region: {},{}".format(self.function,self.value))
        
        #### GET IMAGE 
        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image = getattr(rawdata,itype).copy().astype(float)

        Nrow,Ncol = image.shape
        
        setattr(rawdata,itype+"_woExtraDC",image.copy())

        if self.function.lower() in ['uniform','uni']:
            # value is assumed to be the same units as the input image (no conversion is applied!)
            image[rawdata.mask_image_active_region] += float(self.value)
        elif self.function.lower() in ['dc']:
            # ADD DC ROW DEPENDENT COMPONENT
            DCslope = self.params[0]
            DCint   = self.params[1]
            noise   = self.params[2]
            model = lambda row: np.random.poisson(DCslope/4*row + DCint/4,Ncol)
            img = [model(row) for row in range(Nrow)]

        elif self.function.lower() in ['linear']:
            # POSITIONS: random positions following a linear distribution with rows (to emulate the
            # exposure time due to readout)
            # ENERGY: flat distribution between 0 up to 0.2 keV
            # CLUSTER SIZE: 1-pix cluster
            rows,cols,energy = self._random_clusters(rawdata,self.Ncls)
            image[rows,cols] += energy

        setattr(rawdata,itype,image.copy())
        
        if not rawdata.__process_chain__.split("/")[-1] == self.__name__:
            rawdata.__process_chain__+="/"+self.__name__

    def _random_clusters(self,rawdata,Ncls):
        """
        """

        prob = lambda x: 0.0050 + 0.0040*x
        ener = lambda x: np.random.uniform(self.Emin,self.Emax)
        
        
        rows = []
        cols = []
        E    = []
        for i in range(Ncls):
            #### random position followin a linear distribution
            nrow = None
            while (nrow is None):
                _rx,_ry = np.random.uniform(0,1), np.random.uniform(0,prob(1))
                _row = int(rawdata.slice_active_region_rows.start + rawdata.slice_active_region_rows.stop*_rx)
                if _ry<prob(_rx) and (_row<rawdata.slice_active_region_rows.stop and _row>rawdata.slice_active_region_rows.start):
                    nrow = _rx
                    ### ADD ROW
                    rows.append( _row )
                    ### ADD COLUMN
                    cols.append( int(np.random.uniform(rawdata.slice_active_region_cols.start,rawdata.slice_active_region_cols.stop)) )
                    ### ADD ENERGY
                    E.append(ener(0))

        return rows,cols,E


####################################################################################################################
#
#
#   MEAN CHARGE IN ROWS AND/OR COLUMNS
#
#
####################################################################################################################
class MeanPixelChargePerAxis(SKImageProcess):
    """
    """
    __sequence_id__ = 500
    __name__ = 'MeanPixelChargePerAxis'
    
    def __init__(self):
        """
        """
        super().__init__()

        self.image = "mean_compressed"

        self.function = "mean"
        self.axis = "col"
        self._per_amp = False
        self.mask = False
        self.charge = 100

        ### 
        self.__units__.update({
            'function': 1, 'axis':1, 'image':1, 'mask':1, 'charge':1
            })
    
    def execute_process(self,rawdata):
        """
        """
        
        if self.axis in ["row","rows"]:
            amplifier=''
            if hasattr(rawdata,'execute_process_in_amp'):
                amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)
            self._per_amp = True

        else:
            self._per_amp = False
            amplifier = 'both'

        print("Process <MeanPixelChargePerAxis> {} pixel charge per {} [amplifier {}]".format(self.function,self.axis,amplifier))

        #### image in which the data should be given
        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image)
        

        #### START THE PROCESS
        ################################################################################
        
        function = getattr(np,self.function)
        axis = 1 if self.axis in ["row","rows"] else 0
        
        if self.mask:
            _image = np.ma.array(image_amp.data, mask=image_amp.data>self.charge)
        else:
            _image = np.ma.array(image_amp.data, mask=False)

        q_axis = function(_image, axis=axis)

        if self.save_plots or self.__display__:
            self.execute_plot(rawdata,np.arange(0,len(q_axis)),q_axis)
               
        return

    def execute_plot(self,rawdata,x,y):
        """
        """
        #### PLOTTING OUTPUTS for the mu and sigma
        _isbatch = ROOT.gROOT.IsBatch()
        if self.__display__:
            ROOT.gROOT.SetBatch(0)
        else:
            ROOT.gROOT.SetBatch(1)

        c = ROOT.TCanvas(rawdata.execute_process_in_amp)

        tg = ROOT.TGraph( len(x), array('d',x), array('d',y) )
        tg.Draw("A P")
        tg.SetTitle(" {} pixel charge amplifier: {}".format(self.function,rawdata.execute_process_in_amp))
        tg.GetHistogram().GetYaxis().SetTitle(self.function+"(pix)")
        tg.GetHistogram().GetXaxis().SetTitle(self.axis)
        c.Update()

        # Display only if --display option is set to ON
        if self.__display__:
            c.Update()
            c.Draw()
            input("press enter ...")

        ROOT.gROOT.GetListOfCanvases().Delete()
        ROOT.gROOT.SetBatch(_isbatch)
    
        return

#######################################################################################################
#
#
#
#
#######################################################################################################
class EvalCTIProcess(SKImageProcess):
    """The image will be evaluated from CTI, both vertical and horizontal

    """
    __sequence_id__ = 510
    __name__ = 'EvalCTIProcess'

    def __init__(self):
        """
        """
        super().__init__()
        self._per_amp = False
        self.__silent__ = False

        self.image = "mean_compressed_pedestal_subtracted_e"
        self.VCTI = [155,10]
        self.HCTI = [155,10]

        self.Qrange_e = [0.75,1.5]
        self.cumulative = True

        ### 
        self.__units__.update({'VCTI':1,'HCTI':1, 'Qrange_e':1, 'cumulative':1})
    
    def execute_process(self,rawdata):
        """
        """
        
        if not hasattr(rawdata,'Nclusters'):
            raise IOError(f"ClusterFinder process should be booked before {self.__name__}.")

        if not self.__silent__:
            print(f"Process <{self.__name__}> INFO.")
        

        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image = getattr(rawdata, f"image_{self.image}")
        maxR,maxC  = image.shape
        if self.__DEBUG__:
            from matplotlib import pyplot as plt

        for idx in range(rawdata.Nclusters):
            ##### initialization of the event_rate
            setattr(rawdata.evt_clusters[idx],"vcti_pix",np.arange(self.VCTI[1],self.VCTI[0],self.VCTI[1]).astype(int))
            setattr(rawdata.evt_clusters[idx],"vcti_rate", np.zeros_like(rawdata.evt_clusters[idx].vcti_pix).astype(float))

            setattr(rawdata.evt_clusters[idx],"hcti_pix",np.arange(self.HCTI[1],self.HCTI[0],self.HCTI[1]).astype(int))
            setattr(rawdata.evt_clusters[idx],"hcti_rate", np.zeros_like(rawdata.evt_clusters[idx].hcti_pix).astype(float))

            #if not rawdata.evt_clusters[idx].has_seed:
            #    continue

            # coordenates are shifted in the output root file!
            rows = rawdata.evt_clusters[idx].pixels_y-1
            cols = rawdata.evt_clusters[idx].pixels_x-1

            # Evaluate VERTICAL CTI: set of columns with the maximum value of the row
            ###########################################################################################3
            #  sort the structured array by 'first' (cols) then by 'second' (rows) in descending order
            points = np.sort( np.array(list(zip(cols,rows)), dtype=[('first', int), ('second', int)]), order=['first', 'second'] )[::-1]
            #  find the unique 'first' (col) element, and extract the tuples with the highest 'second' (rows) values for each unique 'first'
            _, unique_indices = np.unique(points['first'], return_index=True)
            results = points[unique_indices]
             
            for j,dr in enumerate(range(self.VCTI[1],self.VCTI[0],self.VCTI[1])):
                for ci,ri in results:
                    r_end   = min(ri+dr, maxR)
                    r_start = ri+1 if self.cumulative else r_end-self.VCTI[1]+1
                    r_start = max(0,r_start)
                    rawdata.evt_clusters[idx].vcti_rate[j] += np.sum(np.logical_and(image[r_start:r_end+1,ci] > self.Qrange_e[0], 
                        image[r_start:r_end+1,ci] < self.Qrange_e[1])) / ((r_end-r_start+1)*len(results))

                if self.__DEBUG__:
                    _selR,_selC = zip(*results)
                    if j==len(rawdata.evt_clusters[idx].hcti_rate)-1:
                        plt.figure(ri+ci)
                        plt.imshow( image[min(_selR):max(_selR)+self.VCTI[0]+1,min(_selC):max(_selC)+1] > self.Qrange_e[0], origin='lower', aspect='auto')
                        plt.colorbar()
                        plt.show(block=True)
                   

            # Evaluate HORITZONTAL CTI: set of rows  with the maximum value of the column
            ###########################################################################################3
            #  sort the structured array by 'first' (rows) then by 'second' (cols) in descending order
            points = np.sort( np.array(list(zip(rows,cols)), dtype=[('first', int), ('second', int)]), order=['first', 'second'])[::-1]
            #  find the unique 'first' (row) element, and extract the tuples with the highest 'second' (cols) values for each unique 'first'
            _, unique_indices = np.unique(points['first'], return_index=True)
            results = points[unique_indices]
            
            for j,dc in enumerate(range(self.HCTI[1],self.HCTI[0],self.HCTI[1])):
                
                for ri,ci in results:
                    c_end   = min(ci + dc, maxC)
                    c_start = ci+1 if self.cumulative else c_end-self.HCTI[1]+1
                    c_start = max(0,c_start)

                    Ncti = np.sum( np.logical_and(image[ri,c_start:c_end+1] > self.Qrange_e[0],
                                                     image[ri,c_start:c_end+1] < self.Qrange_e[1]))
                    rawdata.evt_clusters[idx].hcti_rate[j] += np.sum(np.logical_and(image[ri,c_start:c_end+1] > self.Qrange_e[0], 
                        image[ri,c_start:c_end+1] < self.Qrange_e[1])) / ((c_end-c_start+1)*len(results))

                    if self.__DEBUG__:
                        print(dc,ri,ci, Ncti, c_end-c_start+1, len(results), Ncti/((c_end-c_start+1)*len(results)), rawdata.evt_clusters[idx].hcti_rate[j] )
                
                if self.__DEBUG__:
                    _selR,_selC = zip(*results)
                    if j==len(rawdata.evt_clusters[idx].hcti_rate)-1:
                        plt.figure(ri+ci)
                        plt.imshow( image[min(_selR):max(_selR)+1,min(_selC):max(_selC)+self.HCTI[0]+1] > self.Qrange_e[0], origin='lower', aspect='auto')
                        plt.colorbar()
                        plt.show(block=True)


        return


#######################################################################################################
#
#
#
#
#######################################################################################################
class EvalHaloProcess(SKImageProcess):
    """The image will be evaluated from Halo

    """
    __sequence_id__ = 511
    __name__ = 'EvalHaloProcess'

    def __init__(self):
        """
        """
        super().__init__()
        self._per_amp = False
        self.__silent__ = False

        self.image = "mean_compressed_pedestal_subtracted_e"
        self.radius = [155,10]

        self.Qrange_e = [0.75,1.5]
        self.cumulative = True

        ### 
        self.__units__.update({'radius':1,'Qrange_e':1, 'cumulative':1})
    
    def execute_process(self,rawdata):
        """
        """
        
        if not hasattr(rawdata,'Nclusters'):
            raise IOError(f"ClusterFinder process should be booked before {self.__name__}.")

        if not self.__silent__:
            print(f"Process <{self.__name__}> INFO.")
        

        ################################################################################
        itype = rawdata.get_image_attribute_name(self.image)
        image = getattr(rawdata, f"image_{self.image}")
        maxR,maxC = image.shape
        # Create a grid of coordinates
        _Y,_X = np.ogrid[:image.shape[0],:image.shape[1]]

        for idx in range(rawdata.Nclusters):
            ##### initialization of the event_rate
            setattr(rawdata.evt_clusters[idx],"halo_pix",  np.arange(self.radius[1],self.radius[0],self.radius[1]).astype(int))
            setattr(rawdata.evt_clusters[idx],"halo_rate", np.zeros_like(rawdata.evt_clusters[idx].halo_pix).astype(float))

            if not rawdata.evt_clusters[idx].has_seed:
                continue

            # coordenates are shifted in the output root file!
            rows = rawdata.evt_clusters[idx].pixels_y-1
            cols = rawdata.evt_clusters[idx].pixels_x-1

            # Evaluate HALO in a set of columns with the minimum value of the row
            ###########################################################################################3
            points = np.sort( np.array(list(zip(cols,rows)), dtype=[('first', int), ('second', int)]), order=['first', 'second'] )
            _, unique_indices = np.unique(points['first'], return_index=True)
            results = points[unique_indices]
            
            halo_rate_cols = np.zeros_like(rawdata.evt_clusters[idx].halo_pix).astype(float)
            for j,dr in enumerate(range(self.radius[1],self.radius[0],self.radius[1])):
                for ci,ri in results:
                    r_start = max(0,ri - dr)
                    r_end   = ri if self.cumulative else (r_start+self.radius[1]-1)
                    r_end = min(r_end,maxR)
                    halo_rate_cols[j] += np.sum(np.logical_and(image[r_start:r_end+1,ci] > self.Qrange_e[0], 
                        image[r_start:r_end+1,ci] < self.Qrange_e[1])) / ((r_end-r_start+1)**len(results))
                    
            # Evaluate HALO in a set of rows  with the minimum value of the column
            ###########################################################################################3
            #  sort the structured array by 'first' (rows) then by 'second' (cols) in descending order
            points = np.sort( np.array(list(zip(rows,cols)), dtype=[('first', int), ('second', int)]), order=['first', 'second'])
            #  find the unique 'first' (row) element, and extract the tuples with the highest 'second' (cols) values for each unique 'first'
            _, unique_indices = np.unique(points['first'], return_index=True)
            results = points[unique_indices]
            
            halo_rate_rows = np.zeros_like(rawdata.evt_clusters[idx].halo_pix).astype(np.float64)
            for j,dc in enumerate(range(self.radius[1],self.radius[0],self.radius[1])):
                for ri,ci in results:
                    c_start = max(0,ci - dr)
                    c_end   = ci if self.cumulative else (c_start+self.radius[1]-1)
                    c_end   = min(c_end,maxC)
                    halo_rate_rows[j] += np.sum(np.logical_and(image[ri,c_start:c_end+1] > self.Qrange_e[0], 
                        image[ri,c_start:c_end+1] < self.Qrange_e[1])) / ((c_end-c_start+1)**len(results))
                    
            
            # Evaluate HALO in the lower-left quadrant of the minimum cluster position (center of the circle)
            ###########################################################################################3
            i,j = rawdata.evt_clusters[idx].minY, rawdata.evt_clusters[idx].minX
            # Calculate the squared distances from the center
            dist_from_center = (_X - j)**2 + (_Y - i)**2
            # Create a mask for the negative quadrant (lower-left quadrant)
            negative_quadrant_mask = np.logical_and(_X <= j, _Y >= i)
           
            halo_rate_q3 = np.zeros_like(rawdata.evt_clusters[idx].halo_pix).astype(float)
            for j,dc in enumerate(range(self.radius[1],self.radius[0],self.radius[1])):
                # Create a mask for pixels within the radius
                dc_prev = 0 if self.cumulative else dc-self.radius[1]+1
                within_radius_mask = np.logical_and(dist_from_center>=dc_prev**2, dist_from_center <= dc**2)
                # Combine both masks
                final_mask = np.logical_and(within_radius_mask, negative_quadrant_mask)
                
                num_pix = np.pi*dc**2/4. if self.cumulative else np.pi*(dc**2 - dc_prev**2)/4.

                try:
                    ri,ci = list(zip(*np.argwhere(final_mask)))
                    halo_rate_q3[j] += np.sum(np.logical_and(image[ri,ci] > self.Qrange_e[0], image[ri,ci] < self.Qrange_e[1])) / num_pix
                except ValueError:
                    continue
                

            # total event rate at the halo (we do not account for quadrant1, as may we have CTI)
            rawdata.evt_clusters[idx].halo_rate = halo_rate_cols + halo_rate_rows + halo_rate_q3
            
        return

