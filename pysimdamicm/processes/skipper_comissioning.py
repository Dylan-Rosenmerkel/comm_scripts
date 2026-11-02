""":mod:`pysimdamicm.skiat.ccdimg`
    
    skipperimg is a collection of algorithms for CCD image processing
    Module of the main methods for the skiat of a CCD image      
    
.. moduleauthor:: Nuria Castello-Mor
"""

from pysimdamicm.processes.absp import SKImageProcess
from pysimdamicm.processes import skipper_analysis
from pysimdamicm.utils.libplot4ana import ImagePlot, XYPlot,PCDPlot
from pysimdamicm.utils.units import Units
u=Units()


from matplotlib import pyplot as plt
import numpy as np
from scipy import fftpack,stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import norm
from array import array

import ROOT

#from memory_profiler import profile
#################################################################################################
#                                                                                               #
#       Fit Calibration Constant                                                                #
#                                                                                               #
#################################################################################################
class FitCalibrationConstant(SKImageProcess):
    """
    """
    __name__ = 'FitCalibrationConstant'
    __sequence_id__ = 400
    def __init__(self):
        super().__init__()

        # parameters of the process
        self.image = "mean_compressed"
        self.n_peaks         = 100
        self.calibration     = None
        self.n_sigma_win_fit = 3
        self.show_fit        = True
        self.__verbose__     = False
        self.pcd_bin_width   = 1/20.

        self.use_pcd = None

        self.__units__.update({'n_peaks':1,'calibration':1,'n_sigma_win_fit':1,'show_fit':1,
            'use_pcd':1, 'pcd_bin_width':1})

    def execute_process(self,rawdata):
        """
        """
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)
        print("Process <FitCalibrationConstant> INFO. Fit all peaks to a gaussian to find the calibration constant {}.".format(amplifier))
        
        # add to the dictionary all desired outputs 
        self.results = {}

        #### GET IMAGE TO ESTIMATE THE CALIBRATION CONSTANT: pedestal_subtracted or compressed image
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image)

        #### convert into a flat array (only one axis)
        if np.ma.is_masked(image_amp):
            #### avoid masked pixels
            image = image_amp.compressed()
        else:
            image = image_amp.flatten()

        if self.use_pcd is not None:
            image = self.use_pcd
        
        ### FITTING PROCESS TO self.n_peaks PEAKS STARTS
        #######################################################################################
        #### estimate calibration within the firts two peaks
        cal,mu0 = self.estimate_calibration(image,self.calibration)
        if cal<0:
            self.print_warning("Calibration constant is negative: {}ADU/e-. Should the image be inverted?".format(round(cal,2)))
        ### calibrate 
        image = image/cal
        
        ### OUTPUTS
        # center of the gaussian peaks 
        mu_e         = np.zeros(self.n_peaks)
        # sigma of the gaussian peaks
        sigma_e      = np.zeros(self.n_peaks)
        # sigma scaled by 1/sqrt(number of points used in the fit, degree of freedoms dof)
        sigma_mean_e = np.zeros(self.n_peaks)
        # sigma of the sigma: 2.0*sigma**2 / sqrt(dof-1)
        sigma_sigma_e= np.zeros(self.n_peaks)

        #### STARTING POINT: fitting the zero=charged peak (WITHOUT SHOWING OUTPUTS)
        image_0 = image[(image>image.min()) & (image<2)]
        mu_e[0],sigma_e[0] = self.fit_gaussian_peak(image_0,nsig=7,normed=True,axis=0,distance=1)
        
        print("    - Starting process with peak of zero charge at {} electrons".format(mu_e[0]))
        for i in range(1,self.n_peaks):
            # spectral window centered to i-electrons, with 3sigma width
            e_low  = mu_e[i-1] + self.n_sigma_win_fit*sigma_e[0]
            e_high = e_low + 1

            # only pixels within this spectral window
            img_peak_i = image[(image>e_low) & (image<=e_high)]

            # fitting selected data
            if len(img_peak_i) == 0:
                self.print_warning("No pixels with charge {} electrons. If this is not expected check input parameters (invert polarity? skips? ...)".format(i))
                pass
            mu_e[i],sigma_e[i] = self.fit_gaussian_peak(img_peak_i,distance=1)

            # errors
            sigma_mean_e[i]  = sigma_e[i]/np.sqrt(len(img_peak_i))
            sigma_sigma_e[i] = np.sqrt(2.*sigma_e[i]**4./np.sqrt(len(img_peak_i)-1))
            e_low = e_high
        

        #### fitting to get the fitted calibration constant, and calibrate our signal
        def linefit(x,a,b):
            return a*x + b
        
        x_nelec = np.arange(0,len(mu_e))
        popt, pcov = curve_fit(linefit, x_nelec, mu_e)
        
        print("Best fit ", popt, pcov)

        self.results.update({"mu_e": mu_e, "sigma_e": sigma_e, "n_e":x_nelec, "calibration":cal})

        perr = np.sqrt(np.diag(pcov))
        perr[0] = 0.0 if pcov[0,0] is np.inf else np.sqrt(np.diag(pcov))[0]
        perr[1] = 0.0 if pcov[1,1] is not np.inf else 0
    
        #### set calibration (ADC2e) to the Units class, singleton u 
        u.ADC2e = popt[0]*cal
        u.ADC2e_err = perr[0]
        print("     - Fitted Calibration constant:")
        print("         from lineal fit {} ADU/e- (error {})".format(u.ADC2e, u.ADC2e_err))
        print("         from guasian fit, mean of median(mu(i)-mu(j)) = {} ADU/e-".format( np.median(np.diff(mu_e)*cal)))
        print("         from weighted (1/n_e) average = {} ADC/e-\n\n".format(np.average(np.diff(mu_e)*cal,weights=1/x_nelec[1:])) )

        #### PLOTTING ....
        # 1) CALIBRATION LINEALITY
        x_list       = [x_nelec, x_nelec, x_nelec, x_nelec]
        y_list       = [mu_e, (mu_e - linefit(x_nelec,*popt))/(x_nelec+1) , mu_e-linefit(x_nelec,*popt), sigma_e ]
        y_err_list   = [sigma_mean_e, sigma_mean_e/(x_nelec+1) , sigma_mean_e, sigma_sigma_e ]
        y_label_list = ["Reconstructed Charge [$e^{-}$]", "Relative difference from fit", "Fit residual [$e^{-}$]", "sigma [$e^{-}]$"]

        plot_title = "Linearity from single electron peaks\n image: {}\nprocess chain: {}".format(rawdata._file_name.replace("_","\_"),
                ", ".join(rawdata.__process_chain__.split("/")[1:]))
        figCal = XYPlot(self.__sequence_id__+100,plot_title, xlabel="True Charge [$e^{-}$]", ylabel=None,nplots=4)
        figCal.Draw(x_list,y_list,nrows=2,ncols=2, Yerr=y_err_list, ylabel=y_label_list)
        figCal.cd()
        plt.subplot(2,2,1)
        plt.plot(x_nelec,linefit(x_nelec,*popt),'black', linestyle='--', label="calibration: {} ADU/e-".format(
            round(popt[0]*cal,2)))
        plt.grid()
        figCal.resize((12,7))
        figCal.DrawLegend()
        
        if self.save_plots:
            _ = figCal.SaveAs(rawdata.output+"{}_lineality_from_single_electron_peaks.eps".format(self.__name__,rawdata.execute_process_in_amp))
            _ = figCal.SaveAs(rawdata.output+"{}_lineality_from_single_electron_peaks.pdf".format(self.__name__,rawdata.execute_process_in_amp))

        # 2) PCD plot
        title = "Pixel Charge Distribution\n image: {}".format(
                rawdata._file_name.replace("_","\_"))
        title=title.replace("Process","")
        figPCD = PCDPlot(self.__sequence_id__+100+1,title,xlabel='pixel charge [$e^{-}$]',random=False,Nbins=self.pcd_bin_width)
        figPCD.resize((14,5))

        new_cal = np.average(np.diff(mu_e)*cal,weights=1/x_nelec[1:])
        img_calibrated = image*cal/new_cal
        try:
            Nbins = int((self.n_peaks-img_calibrated.min())/self.pcd_bin_width)
            figPCD.Draw(img_calibrated,region=(img_calibrated.min(),self.n_peaks), Nbins=Nbins)
        except ValueError:
            Nbins = int((self.n_peaks-200)/self.pcd_bin_width)
            figPCD.Draw(img_calibrated,region=(img_calibrated.min(),200), Nbins=Nbins)

        plt.xscale = figPCD.xlabel
        plt.yscale('log')
        plt.ylabel('counts')
        plt.xlabel('pixel charge [e$^{-}$]')
        
        if self.save_plots:
            _ = figPCD.SaveAs(rawdata.output+"{}_PCD_low_and_high_EnergyRange_AMPLIFIER_{}.eps".foramt(self.__name__,rawdata.execute_process_in_amp))
            _ = figPCD.SaveAs(rawdata.output+"{}_PCD_low_and_high_EnergyRange_AMPLIFIER_{}.pdf".foramt(self.__name__,rawdata.execute_process_in_amp))

        rawdata.__process_chain__+="/"+self.__name__

    def estimate_calibration(self,image,cal,prominence=10,n_sig=3.):
        
        ## image has zero-charge pixels to zero (this process is after subtraction), eliminate the 
        #   que of this peak at zero
        #### the left side of the tail of the first-gaussian peak (charge zero)
        #   can have noise, and the detection of the second peak can be problematic
        below_zero_cut =image.min()-(image.min()/2.0)
        image = image[image > below_zero_cut]
        med = np.median(image)
        mad = np.median(np.abs(image - med))
        bins = np.arange( image.min(), med+n_sig*mad)

        peaks = np.array([])
        prominence_try=prominence
        cal_try = cal
        n_tries = 0
        while (peaks.size<2 and n_tries<100): 
            bins = np.arange( image.min(), med+n_sig*mad)

            hpix, edges = np.histogram(image, bins=bins)
            centers = edges[0:-1] + np.diff(edges)[0]/2.

            peaks, prop = find_peaks(hpix, distance=cal_try, prominence=prominence_try)
            n_tries+=1
            n_sig+=1

        prop,peaks = zip(*sorted(zip(prop['prominences'],peaks),reverse=True))        

        if len(peaks)<2:
            return cal,None
        else:
            cal_guess = round(centers[peaks[1]] - centers[peaks[0]],2)
            mu0       = centers[peaks[0]]
            print("    - Estimated calibration from the two first peaks: {}-{}={}\n".format(
                round(centers[peaks[0]],2),round(centers[peaks[1]],2),cal_guess))
        
        if self.__verbose__:
            plt.figure(1521)
            plt.title("Starting point for the calibration constant fit:\n Distance between the two first peaks\n no exit? re-run changing the calibration constant", fontsize=11)
            plt.plot( centers, hpix, 'rx' )
            plt.xlabel("pixel charge [ADU]")
            plt.ylabel("counts")
            plt.axvline( below_zero_cut, ls='solid', label='ignoring pixels below' )
            plt.axvline( centers[peaks[1]], ls='dotted', label='second found peak' )
            plt.legend()
            plt.yscale('log')
            plt.show(block=True)
       
        return cal_guess,mu0


#################################################################################################
#                                                                                               #
#       Study of noise versus number of single skips                                            #
#                                                                                               #
#################################################################################################
class RNvsNskipsPlot(SKImageProcess):
    """
    """
    __name__ = 'NoiseVSSkipPlot'
    __sequence_id__ = 300
    def __init__(self):
        super().__init__()
        
        # parameters of the process
        self.n_skips_per_block = -100
        self.is_blank = False
        ## 
        self.__units__.update({'n_skips_per_block':1,'is_blank':1})

    def execute_process(self,rawdata):
        """
        """
        # Define the working range for the 3 axes: rows, cols, skips 
        # user range of rows
        user_rows_range = self.get_user_axis_range(rawdata,'row')
        # user range of cols
        user_cols_range = self.get_user_axis_range(rawdata,'col')
        # user range of skips
        user_skips_range = self.get_user_axis_range(rawdata,'skip')

        if not self.__silent__:
            amplifier=''
            if hasattr(rawdata,'execute_process_in_amp'):
                amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)

            if self.n_skips_per_block>0:
                print("Process <NoiseVSSkipPlot> INFO. Running study of readout noise as a function of the number os averaged samples {} \n".format(amplifier),
                        "   - using blocks of {} skips, in the range {}:{}".format(self.n_skips_per_block,user_skips_range.start,user_skips_range.stop))
            else:
                print("Process <NoiseVSSkipPlot> INFO. Running study of readout noise as a function of the number os averaged samples {}\n".format(amplifier),
                        " - in log scale, in the range {}:{}".format(user_skips_range.start,user_skips_range.stop))

        ### instanciate <CompressSkipperProcess> to compress image in blocks of n skips
        comp = skipper_analysis.CompressSkipperProcess()
        comp.__silent__  = True
        comp.__verbose__ = False
        ## only the mean function is required
        comp.func_to_compress = ['mean']
        #  first skip to start with
        comp.id_skip_start = user_skips_range.start if self.id_skip_start is None else self.id_skip_start
        
        ### get the number of blocks with <self.n_skips_per_block> skips per block
        #       in the skip range id_skip_start:id_skip_end
        #   Do not include a block with less number of skips per block
        if self.n_skips_per_block>0:
            n_blocks = int( (user_skips_range.stop - user_skips_range.start)/self.n_skips_per_block)
            id_skip_end_list = [self.id_skip_start + 1 + i*self.n_skips_per_block for i in range(n_blocks)]
            id_skip_end_list[-1] = id_skip_end_list[-1] if id_skip_end_list[-1]>rawdata.nskips else rawdata.nskips 

        else:
            #self.n_skips_per_block = abs(self.n_skips_per_block)
            id_skip_end_list = sorted(list(set( np.round(np.exp(np.linspace( np.log(comp.id_skip_start+1),
                np.log(rawdata.nskips),abs(self.n_skips_per_block))),0).astype(int)  )))
            n_blocks = len(id_skip_end_list)

        ### initialization of the mean measurements per block
        std_image_mean_compressed_nskips = np.zeros(n_blocks,dtype=np.float64)
        x_Nskips = []
        
        for i,id_skip_end in enumerate(id_skip_end_list):
            ### last skip for this block
            #id_skip_end   = self.id_skip_start + 1 + i*self.n_skips_per_block
            #if id_skip_end > rawdata.nskips:
            #    comp.id_skip_end = rawdata.nskips
            #else:
            #    comp.id_skip_end   = self.id_skip_start + 1 + i*self.n_skips_per_block
            comp.id_skip_end = id_skip_end

            ### number of skips used in this block
            x_Nskips.append(comp.id_skip_end-self.id_skip_start)
            # print(self.id_skip_start,comp.id_skip_end,x_Nskips[-1])
            ### compresse only this block of images
            ##      the compressed image, is stored in the data member 'image_mean_compressed' of rawdata
            comp.execute_process(rawdata)
            
            
            if self.is_blank:
                ### the input image is only noise (no clusters on it)
                std_image_mean_compressed_nskips[i]=rawdata.image_mean_compressed.std()
            else:
                ### the input image has clusters, use only the overscan region:
                #       overscan region is defined with a mask (boolean image): rawdata.mask_image_overscan_cols
                #       mask the mean image with this mask, and compress all inputs non masked
                # MAD value
                image = rawdata.get_image_by_attribute_name('mean_compressed')
                _median, mad = self.get_med_and_mad(np.ma.array(rawdata.image_mean_compressed,mask=rawdata.mask_image_overscan_cols).compressed())
                std_image_mean_compressed_nskips[i] = mad * 1.4826

        self.results = {'NDCM': x_Nskips, 'noise':std_image_mean_compressed_nskips.tolist()}
        if self.__display__ or self.save_plots:
            self.draw_plot(x_Nskips,std_image_mean_compressed_nskips,rawdata.output, rawdata.execute_process_in_amp)
        rawdata.__process_chain__+="/"+self.__name__

    def draw_plot(self,x_Nskips,std_image_mean_compressed_nskips,output, amplifier):
        
        batch = ROOT.gROOT.IsBatch()
        if self.__display__:
            # window mode
            ROOT.gROOT.SetBatch(0)
        else:
            # batch mode
            ROOT.gROOT.SetBatch(1)

        c = ROOT.TCanvas("Readout Noise","Reaadout Noise",900,850)
        leg = ROOT.TLegend(0.68,0.68,0.83,0.83)
        x = array('d', x_Nskips)
        tg = ROOT.TGraph(len(x), x, array('d', std_image_mean_compressed_nskips) )
        tg.SetMarkerStyle(20)
        tg.SetMarkerSize(0.9)
        tg.SetMarkerColor(ROOT.kBlack)
        tg.GetHistogram().GetXaxis().SetTitle("N_{skips}")
        tg.GetHistogram().GetYaxis().SetTitle("Pixel readout noise (ADU)")
        tg.SetTitle("Amplifier "+amplifier)
        tg.Draw("A P")
        leg.AddEntry(tg,"Data","P")
        
        # expected: 1 over square N
        y0 = std_image_mean_compressed_nskips[0] / np.sqrt(np.array(x_Nskips))
        # expected 1 over square N assuming the last point (with N skips) is the correct value
        #   the first skip measurements has more noise than should be
        sigma_1 = std_image_mean_compressed_nskips[-1] * np.sqrt(x_Nskips[-1])
        y = sigma_1 / np.sqrt(np.array(x_Nskips))
        tgm = ROOT.TGraph(len(x), x, array('d',y))
        tgm.SetLineColor(ROOT.kRed)
        tgm.SetLineStyle(1)
        tgm.SetLineWidth(2)
        leg.AddEntry(tgm,"1/\sqrt{N_{skip}}","L")
        tgm.Draw("L SAME")
        
        tgm0 = ROOT.TGraph(len(x), x, array('d',y0))
        tgm0.SetLineColor(ROOT.kRed)
        tgm0.SetLineStyle(9)
        tgm0.SetLineWidth(2)
        tgm0.Draw("L SAME")

        leg.Draw("SAME")
        c.SetLogx()
        c.SetLogy()
        c.Update()
        
        if self.save_plots:
            for fmt in ['png','root','pdf']:
                c.SaveAs(output+"{}_{}.{}".format(self.__name__,amplifier,fmt))
        
        if self.__display__:
            c.Draw()
            input("press enter ...")

        # Return same ROOT conditions
        ROOT.gROOT.SetBatch(batch)
        ROOT.gROOT.GetListOfCanvases().Delete()

#################################################################################################
#                                                                                               #
#       FFTNoise :: STUDY NOISE SPECTRUM BY FFT                                                 #
#                                                                                               #
#################################################################################################
class FFTNoisePlot(SKImageProcess):
    """
    """
    __name__ = 'FFTNoisePlot'
    __sequence_id__ = 200

    def __init__(self):
        super().__init__()
        
        ### all attributes from default

    def execute_process(self,rawdata):
        """
        """
        amplifier=''
        if hasattr(rawdata,'execute_process_in_amp'):
            amplifier= '[amplifier {}]'.format(rawdata.execute_process_in_amp)
        print("Process <FFTNoisePlot> INFO. Running 'study of noise spectrum by FFT. {}.".format(amplifier))
        

        self.image = "mean_compressed"
        itype = rawdata.get_image_attribute_name(self.image)
        image_amp = rawdata.get_image_by_attribute_name(self.image)
        if image_amp is None:
            raise IOError("Process CompressSkipperProcess should be booked!")

        ####   FFT in a given range of skips, rows and columns                  ############
        rstart, rend = rawdata.amplifier[rawdata.execute_process_in_amp]['rows']
        user_rows_range = slice(rstart,rend)
        # user range of cols
        cstart, cend = rawdata.amplifier[rawdata.execute_process_in_amp]['cols']
        user_cols_range = slice(cstart,cend)

        if rawdata.nskips>1:
            # user range of skips
            user_skips_range = self.get_user_axis_range(rawdata,'skip')
            Nskips_FFT = user_skips_range.stop - user_skips_range.start
            print(" - FFT: for the skip noise spectrum using skip range:  {}:{} ({} skips)".format(user_skips_range.start,user_skips_range.stop,Nskips_FFT))
        else:
            Nskips_FFT = 1
            print(" - FFT: for the skip noise spectrum using skip range: 1 skip")

        ##### FAST FOURIER TRANSFER FOR EACH SINGLE SKIP IMAGE
        ##########################################################################
        Ncols_FFT = user_cols_range.stop - user_cols_range.start + 1
        Nrows_FFT = user_rows_range.stop - user_rows_range.start + 1
        print(" - FFT: for the row noise spectrum using column range: {}:{} ({} rows)".format(user_rows_range.start,user_rows_range.stop,Nrows_FFT))
        print(" - FFT: for the col noise spectrum using column range: {}:{} ({} cols)\n".format(user_cols_range.start,user_cols_range.stop,Ncols_FFT))

        #### output of the FFT for the single skip measurement
        image_single_skips_FFT = np.zeros(Nskips_FFT, dtype=np.float64)
        #### output of the FFT for the rows
        image_avg_row_FFT = np.zeros(Ncols_FFT, dtype=np.float64)

        #################################################################################################
        ####   MEMORY ALLOCATION PROBLEMS USING THE FULL CCD SIZE     ###################################
        #################################################################################################
        # fftpack.fft allows to do the FFT in a given list of axis in once, however a very high price is#
        #   paid in terms of memory, and some systems will just colapse                                 #
        #                                                                                               #
        #   SOLUTION: do skip by skip and row by row to get the list of FFT per skip and per row!!!!    #
        #       this solution increase the execution time a little bit                                  #
        #################################################################################################
        ### image axis are Nrows x Ncols x Nskips
        if rawdata.nskips>1:
            for row in range(rstart,rend,1):
                for col in range(cstart,cend,1):
                    ### temporal information of the charge in pixel i,j = row,col
                    #    - as an input to the FFT we pass the temporal list of all skips for a given pixel after subtract its mean value
                    #    - to reduce memory consumption, no intermediate variable is defined for this subtraction
                    fft_row_col = fftpack.fft( rawdata.image[row,col,:].astype(float) - rawdata.image[row,col,:].mean(), n=Nskips_FFT )
                    #    -   sum_i (sum_j (| FFT_k(q_{ij}^k - <q_{ij}^k) |))
                    image_single_skips_FFT += np.abs(fft_row_col)
                # the same but now per rows and using the already compressed mean image
                fft_row = fftpack.fft(image_amp[row,:].compressed() - np.ma.mean(image_amp[row,:]), n=Ncols_FFT)
                image_avg_row_FFT += np.abs(fft_row)
        else:
            for row in range(rstart,rend,1):
                # the same but now per rows and using the already compressed mean image
                fft_row = fftpack.fft(image_amp[row,:].compressed()-np.ma.mean(image_amp[row,:]), n=Ncols_FFT)
                image_avg_row_FFT += np.abs(fft_row)
       
        ### Get the average FFT power
        #       for the single skips
        if rawdata.nskips>1: 
            image_single_skips_FFT /= float(Ncols_FFT*Nrows_FFT)
        #       for the rows
        image_avg_row_FFT /= float(Nrows_FFT)
        print("FFT done.\n")
        
        print("Preparing plots .... ")
        #####################################################################################################
        #####                                                                                               #
        #####   PLOTING                                                                                     #
        #####################################################################################################
        ### for skips: frequency from the total time used to read the full CCD
        read_time_per_measurement = rawdata.read_time/(rawdata.nrows * rawdata.nallcols/rawdata.n_amp)
        skip_sampling_frequency = 1./read_time_per_measurement
        x_skips_fft = np.arange(0,skip_sampling_frequency,skip_sampling_frequency/float(Nskips_FFT))
        ### for rows
        read_time_per_row = rawdata.read_time/float(rawdata.ncols/rawdata.n_amp * rawdata.nskips)
        row_sampling_frequency = 1./read_time_per_row
        x_row_fft = np.arange(0,row_sampling_frequency,row_sampling_frequency/float(Ncols_FFT))
        
        if rawdata.nskips>1:
            self.draw_plot(x_skips_fft[1:int(Nskips_FFT/2)], image_single_skips_FFT[1:int(Nskips_FFT/2)],
                    "FFT magnitude (of the skips)",rawdata.output+"_"+rawdata.execute_process_in_amp,"_skips_{}".format(rawdata.execute_process_in_amp), amplifier )
        self.draw_plot(x_row_fft[1:int(Ncols_FFT/2)],image_avg_row_FFT[1:int(Ncols_FFT/2)],
            "FFT magnitude (of the rows)",rawdata.output+"_"+rawdata.execute_process_in_amp,"_rows_{}".format(rawdata.execute_process_in_amp), amplifier )

        rawdata.__process_chain__+="/"+self.__name__
 

    def draw_plot(self,xvals,yvals,xlabel,output=None,dtype='_skips',title=""):
        
        batch=ROOT.gROOT.IsBatch()
        if self.__display__:
            # window mode
            ROOT.gROOT.SetBatch(0)
        else:
            # batch mode
            ROOT.gROOT.SetBatch(1)

        from pysimdamicm.utils.root_plot_styles import damicmStyle
        style = damicmStyle()
        style.cd()
        ROOT.gROOT.ForceStyle()

        c = ROOT.TCanvas("FFT_magnitude","FFT_magnitude",1400,850)
        
        x = array('d', xvals)
        ymin,ymax = 10**np.log10(int(min(yvals)*0.8)), 10**np.log10(int(max(yvals)*10))
        tg = ROOT.TGraph(len(x), x, array('d',yvals) )
        tg.SetMarkerStyle(20)
        tg.SetMarkerSize(0.7)
        tg.SetMarkerColor(ROOT.kBlack)
        tg.GetHistogram().GetXaxis().SetTitle("Frequency (Hz)")
        tg.GetHistogram().GetYaxis().SetTitle(xlabel)
        tg.GetYaxis().SetRangeUser(ymin,ymax)
        tg.GetXaxis().SetNoExponent(False)
        tg.SetTitle(title)
        tg.Draw("A PL")
        
        #c.SetLogx()
        c.SetLogy()
        c.Update()
        
        if self.save_plots:
            outname = output+"{}{}".format(self.__name__,dtype).replace("__","_")
            for fmt in ['png','root','pdf']:
                c.SaveAs(outname+".{}".format(fmt))
        if self.__display__:
            c.Draw()
            input("press enter ...")

        # Return same ROOT conditions
        ROOT.gROOT.SetBatch(batch)
        ROOT.gROOT.GetListOfCanvases().Delete()


#################################################################################################
#                                                                                               #
#       ChargeLossPlot :: CHARGE NON-CONSERVATION BETWEEN SKIPS                                 #
#                                                                                               #
#################################################################################################
class ChargeLossPlot(SKImageProcess):
    """
    """
    __name__ = 'ChargeLossPlot'
    __sequence_id__ = 100

    def __init__(self):
        super().__init__()
        
        # this process is independent of the amplifier
        self._per_amp = False


        # parameters of the process
        self.skip_id_list = [0,1,2]
        self.skip_id_baseline = -1
        self.gray_palette = True
        self.histequ = True
        self.pcd_nbins = -1
        self.pcd_ylog = True
        self.pcd_charge_range = [-100*u.ADC,100*u.ADC]

        self.__units__.update({'skip_id_list':1,'skip_id_baseline':1,'gray_palette':1,'histequ':1,'pcd_nbins':1,'pcd_charge_range':1,
            'pcd_ylog':1})
    
    def execute_process(self,rawdata):
        """
        """
        print("Process <ChargeLossPlot> INFO. Running 'Charge non-conservation between skips':\n",
                "     zoom in any cluster to see if there is a charge loss/gain\n",
                "     (so either see black or white ghost track pixels)\n")

        ### at least two skips to plot
        if len(self.skip_id_list) == 0:
            self.skip_id_list = [0,1,2]

        ### include the first skip to be account for (is in the rawdata class)
        if not rawdata.id_skip_start in self.skip_id_list:
            self.skip_id_list.append( rawdata.id_skip_start )
            self.skip_id_list = sorted(self.skip_id_list)

        ### in case there is only two skips, include one more
        if len(self.skip_id_list)<2:
            self.skip_id_list.append(int(max(self.skip_id_list)+1))

        ### Include the "baseline" skip number 
        if self.skip_id_baseline<0:
            self.skip_id_list.append(rawdata.nskips-1)
        elif self.skip_id_baseline>rawdata.nskips:
            raise TypeError("ChargeLossPlot: skip_id_baseline > number of skips")
        else:
            self.skip_id_list.append(self.skip_id_baseline)
        
        ######################################################################################
        ### DO A LIST OF IMAGES TO SHOW IN THE FINAL PLOT
        ##      * images from single skips (self.skip_id_list)
        ##      * compressed images (mean, and/or std)
        ##      * subtraction of the single skip images with a baseline skip image
        #####################################################################################
        imgs = []
        imgs_subtitle = []
        ### SINGLE SKIP IMAGES
        for num in self.skip_id_list:
            imgs.append(rawdata.image[:,:,num])
            imgs_subtitle.append("skip num. {}".format(num))

        image_baseline = imgs[-1]
        imgs_subtitle[-1]+=" (image baseline)"
        
        ### MEAN AND/OR COMPRESSED IMAGES (the name of function[s] used to compressed 
        #       the full skips is listed in the data member 'func_to_compress'
        if rawdata.compressed:
            for stat in rawdata.func_to_compress:
                imgs.append(getattr(rawdata,'image_{}_compressed'.format(stat)))
                imgs_subtitle.append("{} compressed image".format(stat))
        else:
            print("RuntimeWARNING. CompressSkipperProcess is not active or will be executed after ChargeLossPlot\n",
                    "[compressed images will not be shown, change order or include process].")
        
        ### SUBTRACT THE BASELINE IMAGE TO THE SINGLE SKIP IMAGES (objective: look for shadows
        #       or ghost surrounding a ionizing particle track)
        for k,skip in enumerate(self.skip_id_list[:-1]):
            imgs.append(np.subtract(imgs[k],image_baseline,dtype=np.float32))
            imgs_subtitle.append("(skip {}) - (baseline skip)".format(skip))
        
        ######################################################################################
        #
        #   CREATE A FIGURE WITH SUBPLOTS: ONE FOR EACH IMAGE
        #
        ### (IF WAS BOOKED) ###################################################################
        if rawdata.__display__:
            ### MEAN and STD of the full pixels for each single skip image
            mean_pixel_charge_skip      = np.zeros(rawdata.nskips)
            std_pixel_charge_skip       = np.zeros(rawdata.nskips)
            ### mean of the basline subtracted skip image pixels
            mean_diff_pixel_charge_skip = np.zeros(rawdata.nskips)
            for num in range(rawdata.image.shape[2]):
                img_num = rawdata.image[:,:,num]
                mean_pixel_charge_skip[num]      = img_num.mean()
                std_pixel_charge_skip[num]       = img_num.std()
                mean_diff_pixel_charge_skip[num] = np.subtract(img_num,image_baseline,dtype=np.float32).mean()
            
            ### Use the ImagePlot class to plot a list of images, and then append these 3
            #       distributions (the one computed before)
            ## define the number of columns and rows for the subplot figure
            nplots=len(imgs)+4
            ncols=nrows=round(nplots**0.5)
            if nplots > ncols*nrows:
                ncols +=1
            if nplots > ncols*nrows:
                ncols = len(self.skip_id_list)
                nrows = int(len(imgs)/ncols)+1
            ## instanciate the ImagePlot class with a figure number (the id of this class, to avoid
            #       overlaping figures
            plot_title = "Charge non-conservation between skips\nimage:{}\nprocess chain: {}".format(
                    rawdata._file_name.replace("_","\_"),
                    ", ".join(rawdata.__process_chain__.split("/")[1:]))
            implot = ImagePlot(self.__sequence_id__,nrows=nrows,ncols=ncols,htitle=plot_title.replace("Process",""))
            # resize figure size to better looking
            implot.resize((4*ncols,4*nrows))
            # some parameters from the user palette, histequ
            if self.gray_palette:
                implot.palette = 'gray'
            # plot the full list of images
            implot.Draw( imgs, imgs_subtitle, histequ=self.histequ )
            
            # add in the same figure the other distributions
            implot.cd()
            # mean pixel charge vs skips
            plt.subplot(nrows,ncols,nplots-3)
            _ = plt.plot(mean_pixel_charge_skip, marker='o', markeredgecolor='#156e31',markerfacecolor='None', linestyle='None')
            plt.xlabel("skip number")
            plt.ylabel("mean($q_{ij}^{skip}$)")
            plt.axvline(self.skip_id_list[-1], color='black',lw=1,ls='dashed')
            # std pixel charge vs skips
            plt.subplot(nrows,ncols,nplots-2)
            _ = plt.plot(std_pixel_charge_skip, marker='o', markeredgecolor='#de9050',markerfacecolor='None', linestyle='None')
            plt.xlabel("skip number")
            plt.ylabel("std($q_{ij}^{skip}$)")
            plt.axvline(self.skip_id_list[-1], color='black',lw=1,ls='dashed') 
            # mean of the baseline subtracted pixel charge vs skips
            plt.subplot(nrows,ncols,nplots-1)
            _ = plt.plot(mean_diff_pixel_charge_skip,marker='o', markeredgecolor='#4287f5',markerfacecolor='None', linestyle='None')
            plt.xlabel("skip number")
            plt.ylabel("mean($q_{ij}^{skip=k} - q_{ij}^{baseline}$)")
            plt.axhline(0,color='black',lw=1,ls='dotted') 
            plt.axvline(self.skip_id_list[-1], color='black',lw=1,ls='dashed')
            # pixel charge distribution
            if hasattr(rawdata,'image_mean_compressed'):
                plt.subplot(nrows,ncols,nplots)
                if self.pcd_nbins>1:
                    _ = plt.hist(np.ma.array(rawdata.image_mean_compressed,
                        mask=rawdata.mask_image_active_region).flatten(), bins=self.pcd_nbins, range=self.pcd_charge_range )
                else:
                    _ = plt.hist(np.ma.array(rawdata.image_mean_compressed,
                        mask=rawdata.mask_image_active_region).flatten(), bins=500)
                if self.pcd_ylog:
                    plt.yscale('log')
                plt.xlabel("pixel charge [ADU]")
                plt.ylabel("counts")
                plt.xticks(np.linspace(self.pcd_charge_range[0], self.pcd_charge_range[1], num=5))

            ## save plot (if was booked)
            if self.save_plots:
                implot.SaveAs(rawdata.output+"_{}.eps".format(self.__name__))
                implot.SaveAs(rawdata.output+"_{}.png".format(self.__name__))       

        ### save all images (NOT DISTRIBUTIONS) in a multi-extension fits file (one for each image)
        if self.save_image:
            self.SaveAsFits(rawdata.output+"_skipper_chargeloss.fits",imgs,rawdata.image_header.copy())
        
        ### add a footprint on the process chain
        rawdata.__process_chain__+="/"+self.__name__
        



class ChargeLossSkewnessProcess(SKImageProcess):
    """
    """
    __name__ = 'ChargeLossSkewnessProcess'
    __sequence_id__ = 101

    def __init__(self):
        super().__init__()
        
        # parameters of the process
        self.use_overscan = False
        self.id_skip_reference = 1
        self.id_skip_start = 0
        self.id_skip_end = -1
        self.skip_step = 50
        self.kcl_threshold = 3.2
        self.kcl_n_sig = 8
        self.display = True
        
        self.__units__.update({'use_overscan':1,
                'id_skip_reference':1,'skip_step':1,'kcl_threshold':1,'kcl_n_sig':8,'display':1})
    
    def execute_process(self,rawdata):
        """
        """
        print("Process <ChargeLossSkewnessProcess> INFO. Process to estimate the possible charge loss between skip measurements. \n")
        
        if rawdata.nskips==1:
            # nothing to do, it does not have skips!
            return

        skip_ref  = rawdata.nskips+self.id_skip_reference if self.id_skip_reference<0 else self.id_skip_reference
        last_skip = rawdata.nskips+self.id_skip_end if self.id_skip_end<0 else self.id_skip_end
        if int(last_skip/2.)>self.skip_step:
            self.skip_step = 1

        # list of skip images to compute the skewness (and other parameters)
        skips = np.arange(self.id_skip_start,last_skip-1,self.skip_step).tolist()
        if last_skip in skips:
            skips.remove(last_skip)
        
        if self.use_overscan:
            mask_region = rawdata.amplifier[rawdata.execute_process_in_amp]['mask_image_overscan_cols']
        else:
            mask_region = rawdata.amplifier[rawdata.execute_process_in_amp]['mask_image_active_region']

        img_reference = np.ma.array( rawdata.image[:,:,skip_ref], mask=mask_region ).astype(float)
        self.mean_pcdd = []
        self.sum_pcd   = []
        self.kcl_coeff = []
        self.kcl_skew = [[],[]]
        self.kcl_gauss_fit = [[],[],[]]
        self.kcl_extrainfo = []
        self.skips = []
        for s in skips:
            diff_image = (np.ma.array(rawdata.image[:,:,s],mask=mask_region)-img_reference).compressed()
            try:
                results = self.get_skewness_coefficient(diff_image,self.kcl_threshold,self.kcl_n_sig,False,rawdata.correct_polarity)
            except RuntimeError:
                continue
            # skewnessPCDD, skewnessPCDDuncertainty, kcl, kcluncertainty, ampPCDD, muPCDD, stdPCDD
            self.skips.append(s)
            #diff_image = diff_image[row_start:row_end,col_start:col_end].compressed()
            self.mean_pcdd.append( np.ma.mean(diff_image))
            self.sum_pcd.append(np.ma.array(rawdata.image[:,:,s],mask=mask_region).compressed().sum())
            
            self.kcl_coeff.append(results[2]/results[3])
            self.kcl_skew[0].append(results[0])
            self.kcl_skew[1].append(results[0])
            self.kcl_gauss_fit[0].append(results[4])
            self.kcl_gauss_fit[1].append(results[5])
            self.kcl_gauss_fit[2].append(results[6])
            self.kcl_extrainfo.append(results[7])

        if self.display:
            try:
                q_total_img   = np.ma.array( rawdata.image_mean_compressed, mask=mask_region).compressed().sum()
            except AttributeError:
                q_total_img = 1
            amp = rawdata.execute_process_in_amp
            fig, axs = plt.subplots(3,2,figsize=(12,5))
            fig.suptitle("amp {}, Single-skip pixel charge image difference: {}-{}, {}-{}, ... {}-{} NDCMs".format(amp,
                self.skips[0],skip_ref,self.skips[1],skip_ref,self.skips[-1],skip_ref))

            #xlabel = "NDCM ordinal (image diff is NDCM-{})".format()
            axs[0,0].scatter(self.skips, self.kcl_coeff, c='#1c82d6',edgecolors='black', alpha=0.9, s=5)
            axs[0,0].set_xlabel('NDCM ordinal')
            axs[0,0].set_ylabel('$k_{CL}/uk_{CL}$')
            axs[0,1].errorbar(self.skips, self.kcl_skew[0], yerr=np.abs(self.kcl_skew[1]), mfc='#1c82d6', mec='black',alpha=0.9)
            axs[0,1].set_xlabel('NDCM ordinal')
            axs[0,1].set_ylabel('skewness')
            axs[1,0].scatter(self.skips, np.array(self.mean_pcdd)/q_total_img, c='#1c82d6', edgecolors='black', alpha=0.9,s=5)
            axs[1,0].set_xlabel('NDCM ordinal')
            axs[1,0].set_ylabel('mean($ q_{NDCM} - q_{'+str(skip_ref)+'}$)')
            axs[1,1].scatter(self.skips, self.kcl_gauss_fit[1], c='#1c82d6', edgecolors='black', alpha=0.9, s=5)
            axs[1,1].set_xlabel('NDCM ordinal')
            axs[1,1].set_ylabel('$\mu_{0}$')
            axs[2,0].scatter(self.skips,self.sum_pcd, c='#1c82d6', edgecolors='black', alpha=0.9, s=5)
            axs[2,0].set_xlabel('NDCM ordinal')
            axs[2,0].set_ylabel('total image charge')
            axs[2,0].ticklabel_format(useOffset=False,style='plain')

            position = np.where(img_reference == np.ma.max(img_reference))
            r,c = position[0][0], position[1][0]
            axs[2,1].plot(rawdata.image[r,c,:], c='red', label='$q_{'+','.join([str(r),str(c)])+'}$')
            axs[2,1].set_xlabel('NDCM ordinal')
            axs[2,1].set_ylabel('pixel charge')
            # add minimum value
            position = np.where(img_reference == np.ma.min(img_reference))
            r,c = position[0][0], position[1][0]
            axs[2,1].plot(rawdata.image[r,c,:], c='blue', label='$q_{'+','.join([str(r),str(c)])+'}$')
            axs[2,1].legend(loc=1)
            axs[2,1].set_yscale('log')
            plt.show(block=True)


    @staticmethod
    def get_skewness_coefficient(diff_imag,kcl_threshold,n_sig=8,display=False,reverse=True):
        """MEthod to estimate the skewness and its uncertanty parameter to evaluate possible charge loss between skip measurements
        It is an optimization of the Michellangelo's code https://github.com/mikyphy/CCDTesting/blob/main/m_chargeloss.py
        """

        ravelleddifference = diff_imag.flatten()

        #use either negative or positive semiaxis as a range to estimate mean and stdev before actual fit, in order to exclude possible saturation peak at zero
        # XXX OPTIMIZED
        #countpositive=countnegative=0
        #for component in range(len(ravelleddifference)):
        #    if ravelleddifference[component] > 0: countpositive+=1
        #    elif ravelleddifference[component] < 0: countnegative+=1
        # XXX TO
        countpositive = np.sum(ravelleddifference>0)
        countnegative = np.sum(ravelleddifference<0)

        if countnegative > countpositive:
            rangeadhoc = (min(ravelleddifference),-5)
            nbins = int(-5 - min(ravelleddifference))
        elif countnegative < countpositive:
            rangeadhoc = (5, max(ravelleddifference))
            nbins = int(max(ravelleddifference) - 5)
        else:
            raise RuntimeError('Few-skip image charge loss check failed at PCDD parameter estimation stage. Please review image')
        
        #prepare PCDD histogram for mean (~pedestal) and sigma estimation probing bin value
        differencehistogram, binedges = np.histogram(ravelleddifference, int(nbins), range=rangeadhoc, density=False)
        mostlikelydifference = (binedges[np.argmax(differencehistogram)] + binedges[np.argmax(differencehistogram)+1])/2
        mostlikelydifferencecounts = differencehistogram[np.argmax(differencehistogram)]
        #estimate Half Maximum abscissa (first-last skip difference) to get FWHM and then std deviation
        bincounter = 1
        try:
            condition = np.ones(10)
            while(any(condition) and countnegative > countpositive):
                bincounter=bincounter+1
                for i in range (0,10):
                    condition[i] = differencehistogram[np.argmax(differencehistogram)-bincounter-(i+1)] > 0.14*mostlikelydifferencecounts
            while(any(condition) and countnegative < countpositive):
                bincounter=bincounter+1
                for i in range (0,10):
                    condition[i] = differencehistogram[np.argmax(differencehistogram)+bincounter+(i+1)] > 0.14*mostlikelydifferencecounts
        except:
            print('Search for half maximum abscissa for PCDD fit failed.')
        #estimate FWHM and then std deviation
        if countnegative>countpositive:
            twosigmadifference = (binedges[np.argmax(differencehistogram)-bincounter]+binedges[np.argmax(differencehistogram)-bincounter-1])/2
        elif countnegative<countpositive:
            twosigmadifference = (binedges[np.argmax(differencehistogram)+bincounter]+binedges[np.argmax(differencehistogram)+bincounter+1])/2
        #HM: estimate sigma using FWHM, twosigma: estimate sigma using 0.14Maximum (where 2*sigma should be)
        #HMdifferencecounts = differencehistogram[np.argmax(differencehistogram) - bincounter]
        twosigmadifferencecounts = differencehistogram[np.argmax(differencehistogram) - bincounter]
    
        #stdPCDDestimate = abs(mostlikelydifference - HMdifference)
        #stdPCDDestimate /= np.sqrt(2*np.log(2))
        stdPCDDestimate = abs(mostlikelydifference - twosigmadifference)/2
    
        #now find more accurate values for mean (~pedestal) and standard deviation by fitting PCDD in ad hoc range chosen based on estimates above
        #remove counts at zero from saturation
        # XXX optimized
        # ravelleddifferenceinrange = [s for s in ravelleddifference if s > mostlikelydifference - 3*stdPCDDestimate and s < mostlikelydifference + 3*stdPCDDestimate and s != 0]
        # XXX TO
        m1 = np.logical_and(ravelleddifference > mostlikelydifference-3*stdPCDDestimate, ravelleddifference < mostlikelydifference+3*stdPCDDestimate)
        m1 = np.logical_and(m1,ravelleddifference!=0)
        ravelleddifferenceinrange = ravelleddifference[m1].tolist()
    

        def gauss(x,*p):
            A,mu,sigma = p
            return A*np.exp(-(x-mu)**2/(2*sigma**2))

        pguess = [mostlikelydifferencecounts,mostlikelydifference,stdPCDDestimate]
        ravelleddifferenceinrangehist,binedges = np.histogram(ravelleddifferenceinrange, bins=int(max(ravelleddifferenceinrange)-min(ravelleddifferenceinrange)), density=False)
        try:
            bincenters=(binedges[:-1] + binedges[1:])/2
            pfit, varmatrix = curve_fit(gauss, bincenters, ravelleddifferenceinrangehist, p0=pguess)
            pcddhistfit = gauss(bincenters,*pfit)
            ampPCDD, muPCDD, stdPCDD = pfit[0],pfit[1],pfit[2]
        except:
            ampPCDD, muPCDD, stdPCDD = mostlikelydifferencecounts, mostlikelydifference, stdPCDDestimate

        #skewness and uncertainty computation. Will use wider range to exclude isolated far outliers and retain all of the tails
        # XXX optimized
        # ravelleddifferenceinwiderrange = [s for s in ravelleddifference if s < mostlikelydifference + 8*stdPCDD]
        # XXX TO
        ravelleddifferenceinwiderrange = ravelleddifference[ravelleddifference<mostlikelydifference+n_sig*stdPCDD]

        skewnessPCDD = stats.skew(ravelleddifferenceinwiderrange)
        #use expression for skewness var in case of sample extracted from normal distribution
        skewnessPCDDuncertainty = 6*( len(ravelleddifferenceinwiderrange) - 2 )
        skewnessPCDDuncertainty /= ( len(ravelleddifferenceinwiderrange) + 1 )
        skewnessPCDDuncertainty /= ( len(ravelleddifferenceinwiderrange) + 3 )
        skewnessPCDDuncertainty = np.sqrt(skewnessPCDDuncertainty)
    
        #charge loss coefficient and uncertainty computation. Quantity based on lower-upper tail symmetry in absence of charge loss
        #pedestal subtraction and sorting in ascending order
        centeredsortedravelleddifference = np.array(sorted(ravelleddifference-muPCDD))
        #remove saturation counts at muPCDD after pedestal subtraction
        # XXX optimized
        # centeredsortedravelleddifference = [s for s in centeredsortedravelleddifference if s!= - muPCDD]
        # XXX to
        centeredsortedravelleddifference = centeredsortedravelleddifference[centeredsortedravelleddifference != -muPCDD]

        # XXX optimized 
        #count entries below and above symmetry threshold
        #countsbelowsymmetrythreshold = countsabovesymmetrythreshold = 0
        #component = 0
        #while centeredsortedravelleddifference[component] < -kcl_threshold*stdPCDD:
        #    countsbelowsymmetrythreshold += 1
        #    component += 1
        #component = len(centeredsortedravelleddifference) - 1
        #while centeredsortedravelleddifference[component] > kcl_threshold*stdPCDD:
        #    countsabovesymmetrythreshold += 1
        #    component -= 1
        # XXX to
        try:
            countsbelowsymmetrythreshold = np.where(centeredsortedravelleddifference<-kcl_threshold*stdPCDD)[0][-1]+1
        except IndexError:
            raise RuntimeError("Probably low statistics")
        N = len(centeredsortedravelleddifference)
        try:
            countsabovesymmetrythreshold = N - np.where(centeredsortedravelleddifference > kcl_threshold*stdPCDD )[0][0]
        except IndexError:
            raise RuntimeError("Probably low statistics")

        #compute charge loss coefficient
        if countsbelowsymmetrythreshold + countsabovesymmetrythreshold != 0:
            kcl = (countsabovesymmetrythreshold-countsbelowsymmetrythreshold)/(countsbelowsymmetrythreshold+countsabovesymmetrythreshold)
            kcluncertainty = (np.sqrt((countsabovesymmetrythreshold*countsbelowsymmetrythreshold**2) + (countsbelowsymmetrythreshold*countsabovesymmetrythreshold**2)))
            kcluncertainty = kcluncertainty*2/(countsabovesymmetrythreshold+countsbelowsymmetrythreshold)**2
        else:
            kcl = np.nan
            kcluncertainty = np.nan
            print('The charge loss coefficient cannot be computed as there is no counts on either tail beyond the set threshold.')
        
        if display:
            if countnegative>countpositive:
                if reverse:
                    print('negative baseline shift (more charge) in start skip')
                else:
                    print('negative baseline shift (less charge) in start skip')
            elif countnegative<countpositive: 
                if reverse:
                    print('positive baseline shift (less charge) in start skip')
                else:
                    print('positive baseline shift (more charge) in start skip')
            print('Most likely difference (~mean and ~pedestal) is:', mostlikelydifference)
            print('Most likely difference counts are:', mostlikelydifferencecounts)
            print('2*sigma abscissa (first-last skip difference) is:', twosigmadifference)
            print('2*sigma ordinata (counts) is:', twosigmadifferencecounts)
            print('Value of first-last skip PCDD gaussian std estimate is:', round(stdPCDDestimate,4))
            print("Here's skewness of PCDD: ",skewnessPCDD,'+-',skewnessPCDDuncertainty)
            print('The entries below the symmetric threshold are:',countsbelowsymmetrythreshold)
            print('The entries above the symmetric threshold are:',countsabovesymmetrythreshold)
            print('The charge loss coefficient is:', kcl, '+-', kcluncertainty)
            plt.plot(bincenters,ravelleddifferenceinrangehist,label='pcdd')
            try:
                plt.plot(bincenters, pcddhistfit, label='fit curve')
            except:
                print('Gaussian fit not successful. Guess values printed in plot title')
            plt.title('$\mu_{pcdd}=$' + str(round(muPCDD,1)) + ' ADU, $\sigma_{pcdd}=$' + str(round(stdPCDD,1)) + ' ADU')
            plt.show(block=True)

        extrainfo = "N_blw={},N_abv={}".format(countsbelowsymmetrythreshold,countsabovesymmetrythreshold)

        return skewnessPCDD, skewnessPCDDuncertainty, kcl, kcluncertainty, ampPCDD, muPCDD, stdPCDD, extrainfo, countsbelowsymmetrythreshold, countsabovesymmetrythreshold




