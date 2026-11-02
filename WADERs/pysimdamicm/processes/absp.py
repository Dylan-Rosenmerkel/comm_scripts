""":mod:`pysimdamicm.processes.abs`
    
   ABSTRACT CLASSES TO DEFINE PROCESS FOR SIMULATIONS AND ANALYSIS
 
.. moduleauthor:: Nuria Castello-Mor
"""

from pysimdamicm.utils.libplot4ana import XYPlot

from abc import ABCMeta, abstractmethod

import ROOT
import numpy as np
from astropy.io import fits
import warnings
warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)

from scipy.stats import norm
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

__DEBUG__ = False

###############################################################################################
#####       ABSTRACT CLASS TO DEFINE THE DIFFERENT PROCESS 
#####           PART OF THE DETECTOR RESPONSE
###############################################################################################
class SKImageProcess(metaclass=ABCMeta):
    """Abstract Class for the digitilize process.
    
    Each process involved in the response of the DAMIC-M detector (based on CCDs) will 
    be created from this abstract class.

    """
    __axis_cartesian__ = {'x':0, 'y':1} 
    __axis_name__      = {1:'col',0:'row',2:'skip'}
    __axis_id__        = {'col':1,'row':0,'both':[0,1]}
    __fig_num__        = 5000 
    # display plots related with debug
    __verbose__        = False
    # display final plots
    __display__        = False
    # display debug plots
    __DEBUG__          = False
    # to unset standard outputs when funciton is recurrent
    __silent__         = False

    def __init__(self):
        """
        """
        ### variable to define if a process should be done amplifier per amplifier
        self._per_amp       = True

        ### not include in units (it is not for all the process!)
        self.use_mad        = True

        ### Add some attributes common for all process
        self.image = "raw"
        self.save_image = False
        self.save_plots = False
        
        ### define range for the 3 axis
        self.id_skip_start = np.nan
        self.id_skip_end   = np.nan

        self.id_col_start = np.nan
        self.id_col_end   = np.nan

        self.id_row_start = np.nan
        self.id_row_end   = np.nan

        self.exit = 0

        ## used in fit gaussian method
        self.highT = False

        ## include figure format to be save via ROOT.TCanvas, for default pdf
        self.format_figures = ["pdf"]

        self.__units__ = {
                "__verbose__":1,'__display__':1,'__DEBUG__':1,'__silent__':1,
                "image":1,"__sequence_id__":1,
                "save_image":1,"save_plots":1,
                "id_skip_start":1,"id_skip_end":1,
                "id_row_start":1,"id_row_end":1,
                "id_col_start":1,"id_col_end":1,
                "format_figures":1
                }

    def info(self):
        """ Show all attributes of the process
        """ 
        print("<{}> with sequence id {}.\n List of public data members: ".format(self.__name__,self.__sequence_id__))
        for attr in sorted(self.__units__.keys()):
            print("\t * {} = {} ".format(attr, getattr(self,attr)))
    
    def print_warning(self,msm):
        print("\x1b[93m    WARNING. {} \x1b[m".format(msm))
    
    @staticmethod
    def find_peak_position(image,n_moving_avg=50, min_distance=1,in_ADCs=True, verbose=False):
        """
        """
        ### spectral window to find peaks
        if in_ADCs:
            x_max = 150
        else:
            x_max = 10
        x_min = image.min()
        if x_min> x_max:
            x_max = np.median(image)+3.0*np.std(image)

        ### only pixels with charge between xmin and xmax
        image = image[(image>x_min) & (image<x_max)]
        n_bins = int(np.sqrt(len(image)))

        ### distribution
        hpix, edges = np.histogram( image.ravel(), n_bins )
        centers = edges[1:] + np.diff(edges)[0]/2.

        ### smoth signal
        hist_smooth = np.convolve(hpix, np.ones(n_moving_avg)/n_moving_avg, mode="same")

        ### compute derivative
        derivative = np.diff(hist_smooth)

        ### detect sign change
        #       from possitive to negative: maximum
        #       from negative to positive: minimum
        # first create an array with only the sign 
        derivative_sign = np.array([1 if s>=0 else -1 for s in derivative]).astype(int)
        # now rest derivative_sign from derivative_sign shifted one position
        #       1 1 1 1 1 1 -1 -1 -1  1 1 -1 -1
        #         1 1 1 1 1  1 -1 -1 -1 1  1 -1 -1
        #         0 0 0 0 0 -2  0  0  2 0 -2  0  
        #
        # -2: from positive to negatives
        #  2: from negative to positives
        peaks_max = np.where( np.diff(derivative_sign).astype(int) == -2 )[0]
        peaks = centers[peaks_max]
        
        # peaks with a minimum distance
        peaks_distance = np.diff(peaks)
        found_peaks = [peaks[0]]
        for p,d in zip(peaks[1:],peaks_distance):
            if d > min_distance and p>-0.5:
                found_peaks.append( p )

        if verbose:
            plt.figure(SKImageProcess.__fig_num__)
            SKImageProcess.__fig_num__ +=1
            fig,axs = plt.subplots(2, sharex=True)
            fig.suptitle("Find algorithm by using: $f''(x_i) - f''(x_{i+1})>0$ \n(vertical lines: found peaks)",fontsize=11 )
            axs[0].plot( centers, hpix, 'ro', label="data", markersize=1.5 )
            axs[1].plot( centers[1:], derivative, 'gx', label="derivative" )
            for x in found_peaks:
                plt.axvline( x )

        return np.array(found_peaks)

    def set_parameters(self, **model_parameters):
        """Runtime method to set all atributes of the digitize process listed on the configuration
        json file.

        Those process's parameters not listed on the configuration json file will set to the default
        values.

        Parameters
        ----------
        **model_parameters : dict
            Keyword arguments for all attributes realted with the digitizer process (see examples).

        """
        for keyword,val in model_parameters.items():
            if keyword in ["units","unit"]:
                continue
            if hasattr(self,keyword):
                if type(val)==dict:
                    # XXX Some parameters are amplifier dependent and a dictionary must be pass by,
                    # in this case units can not be applied, this is the case of the Calibration
                    # process XXX
                    setattr(self,keyword,val)
                else:
                    try:
                        setattr(self,keyword,val*self.__units__[keyword])
                    except TypeError:
                        if type(val)==list:
                            setattr(self,keyword,[vi*self.__units__[keyword] for vi in val])
            else:
                raise(AttributeError('"{}" invalid attribute for class {}'.format(keyword,self.__class__.__name__)))
   
    def SaveAsFits(self,output,hdus,header,dtype=None,overwrite=True):
        """Store a list of HDU data into a fits file
        """

        print("     INFO. Data have been save as fits file: {}\n".format(output))
	    ### header
        hdu_list = [fits.PrimaryHDU(data=hdus[0], header=header)]

        if len(hdus)>1:
            for data in hdus[1:]:
                if dtype is not None:
                    data = data.astype(dtype)
                hdu_list.append(fits.ImageHDU(data=data))
        
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(output,overwrite=overwrite)        

    def get_user_axis_range(self,rawdata,axis):
        """
        """
        if axis in ['skips','skip',2]:
            slice_range = self.get_user_skip_range(rawdata)
        elif axis in ['col','cols',1]:
            slice_range = self.get_user_col_range(rawdata)
        elif axis in ['row','rows',0]:
            slice_range = self.get_user_row_range(rawdata)

        return slice_range

    def get_user_skip_range(self,rawdata):
        nmax = rawdata.nskips

        if np.isnan(self.id_skip_start):
            self.id_skip_start = rawdata.id_skip_start
            if np.isnan(rawdata.id_skip_start):
                rawdata.id_skip_start = 0
                self.id_skip_start = 0
        if np.isnan(self.id_skip_end):
            self.id_skip_end = rawdata.id_skip_end
            if np.isnan(rawdata.id_skip_end):
                rawdata.id_skip_end = -1
                self.id_skip_end = -1
        if self.id_skip_end<0:
            self.id_skip_end=nmax

        if self.id_skip_start > nmax:
            raise ValueError("ERROR. skip start {} > number of skips {}".format(
                self.id_skip_start,nmax))
        if self.id_skip_end > nmax:
            raise ValueError("ERROR. skip end {} > number of skips {}".format(
                self.id_skip_end,nmax))

        user_skip_range = slice(self.id_skip_start,self.id_skip_end)
        return user_skip_range

    def get_user_col_range(self,rawdata):
        nmax = rawdata.ncols

        if np.isnan(self.id_col_start):
            self.id_col_start = rawdata.id_col_start
            if np.isnan(rawdata.id_col_start):
                rawdata.id_col_start = 0
                self.id_col_start = 0
        if np.isnan(self.id_col_end):
            self.id_col_end = rawdata.id_col_end
            if np.isnan(rawdata.id_col_end):
                rawdata.id_col_end = -1
                self.id_col_end = -1
        if self.id_col_end<0:
            self.id_col_end=nmax

        if self.id_col_start > nmax:
            raise ValueError("ERROR. col start {} > number of cols {}".format(
                self.id_col_start,nmax))
        if self.id_col_end > nmax:
            raise ValueError("ERROR. col end {} > number of cols {}".format(
                self.id_col_end,nmax))

        user_col_range = slice(self.id_col_start,self.id_col_end)
        return user_col_range

    def get_user_row_range(self,rawdata):
        nmax = rawdata.nrows

        if np.isnan(self.id_row_start):
            self.id_row_start = rawdata.id_row_start
            if np.isnan(rawdata.id_row_start):
                rawdata.id_row_start = 0
                self.id_row_start = 0
        if np.isnan(self.id_row_end):
            self.id_row_end = rawdata.id_row_end
            if np.isnan(rawdata.id_row_end):
                rawdata.id_row_end = -1
                self.id_row_end = -1
        if self.id_row_end<0:
            self.id_row_end=nmax

        if self.id_row_start > nmax:
            raise ValueError("ERROR. col start {} > number of cols {}".format(
                self.id_row_start,nmax))
        if self.id_row_end > nmax:
            raise ValueError("ERROR. col end {} > number of cols {}".format(
                self.id_row_end,nmax))

        user_row_range = slice(self.id_row_start,self.id_row_end)
        return user_row_range


    def get_image_attribute_name(self):

        #### GET IMAGE TO SUBSTRACT PEDESTAL
        if self.image != 'raw':
            image_attribute_name = 'image_{}'.format(self.image)
        else:
            image_attribute_name = 'image'

        return  image_attribute_name

    def median_abs_deviation(self,data,axis=None,scale=1.4826):
        """Compute the median absolute deviation of the data along the given axis as an estimator
        of the standard deviation. 
        
        The scaling factor applied to the MAD. The default scale (1.4826) ensures consistency with 
        the standard deviation for normally distributed data.

        """
        if axis is None:
            mad = np.ma.median(np.abs( data - np.ma.median(data))) * scale
        else:
            mad = np.ma.median(np.abs( data - np.ma.median(data, axis=axis)), axis=axis) * scale

        return mad

    def fit_gaussian(self,pc_array,nsig=None,axis=None,do_cumulative_distribution=False,**kwargs):
        """
        """
        ROOT.gROOT.SetBatch(1)
        pc_array = pc_array.compressed() if np.ma.isMaskedArray(pc_array) else pc_array.flatten()
        
        # number of binning to draw the distribution
        Nbins_methods = {
                "sturge":   lambda x: int(1 + np.ceil(np.log2(len(x)))),
                "freedman": lambda x: int(np.ceil((max(x)-min(x))/(2*(np.quantile(x,0.75)-np.quantile(x,0.25))/len(x)**(1/3)))),
                "default":  lambda x: int(0.5*(max(x)-min(x))) if (max(x)-min(x))>5 else 10,
                "manual":   lambda x: kwargs["Nbins_num"] if "Nbins_num" in kwargs else 0 
                }
        try:
            nbins_method = kwargs["Nbins"]
        except KeyError:
            nbins_method = "default" 
        Nbins = Nbins_methods[nbins_method](pc_array)

        title = '' if not 'title' in kwargs else kwargs['title']
        thist = ROOT.TH1F('axis,row {}'.format(title),'Gaussian Fit', Nbins, min(pc_array), max(pc_array) )
        for qij in pc_array:
           _ =  thist.Fill(qij)
        
        # gaussian function
        fitfunc = ROOT.TF1('fitfunc','gaus')
        fitfunc.SetParNames('norm','mu','sigma')
        fitfunc.SetLineColor(2)
        #fitfunc.SetLineWidth(3)
        #fitfunc.SetLineStyle(1)
        # mean: median of the pixels within the 10,90 percentile
        q10,q90 = np.percentile(pc_array,[10,90])
        mu_guess = np.median( pc_array )
        mu_std   = np.std( pc_array )


        ########################################################## 
        #    for images where DC is quite high, more than one 
        #    predominant peaks is present in the fitting window, 
        #    and difficult to ignore with method <mean> + n * std
        mu_guess = None
        n_peaks_found = 0
        if self.highT:
            def _dhistf(x,par):
                return thist.Interpolate(x[0]) 
            dhist = ROOT.TF1("dhist",_dhistf, min(pc_array), max(pc_array))
            derivative = [0.0]
            for i in range(1,int(thist.GetEntries())+1):
                derivative.append( dhist.Derivative(thist.GetBinCenter(i)) )
                if derivative[-1] < 0 and derivative[-2]>0:
                    if n_peaks_found==0:
                            mu_guess = (thist.GetBinCenter(i) + thist.GetBinCenter(i-1))/2.
                            q90_max  = thist.GetBinCenter(i) + 3*(thist.GetBinCenter(i)-thist.GetBinCenter(i-1))
                            q10_min  = np.min(pc_array)
                            n_peaks_found +=1
                    if n_peaks_found==1:
                        xmin = np.min(pc_array)
                        mu_std     = (thist.GetBinCenter(i)-xmin)/4.
                        mu_std_max = (thist.GetBinCenter(i)-xmin)/2.
                        mu_std_min = (thist.GetBinCenter(i)-xmin)/10.
                        n_peaks_found +=1    
                if n_peaks_found>=2:
                    break

        if mu_guess is None:
            # standard method
            mu_guess = np.median( pc_array )
            q90_max = q90
            q10_min = q10
            if n_peaks_found <=1:
                mu_std_max = mu_std*1.5
                mu_std_min = mu_std*0.5

        ##########################################################


        fitfunc.SetParameters(len(pc_array)/Nbins, mu_guess, mu_std )
        # setting ranges for the parameters
        fitfunc.SetParLimits(1, q10_min, q90_max)
        fitfunc.SetParLimits(2, mu_std_min, mu_std_max)
        if self.__DEBUG__:
            print("mu_min, mu_guess, mu_max : ", q10_min, mu_guess, q90_max)
            print("std_min, std_guess, std_max : ", mu_std_min,mu_std,mu_std_max)


        # fitting with the option 'L' for binned data -- with option 'L' I have some outliers ...
        # XXX why? XXX Option L is too senstitive to other bins with lower statistics, 
        # if they are outliers, they will really affect the final fit
        # For a gaussian fit, where the bulk of the data set is more important than tails, 
        # Chi-sqaure fit should be the correct option
        # Option B is to use SetParLimits only needed if fitfunc is a gauss
        thist.Fit(fitfunc,'Q EM B')

        mu,emu,sigma,esigma = fitfunc.GetParameter(1),fitfunc.GetParError(1),fitfunc.GetParameter(2),fitfunc.GetParError(2)
        try:
            chi2 = fitfunc.GetChisquare()/fitfunc.GetNDF()
        except ZeroDivisionError:
            chi2 = -999
        
        if do_cumulative_distribution:
            self.th1_guas_fits.append( thist.Clone("Gauss Fit: {}".format(kwargs['title'])) )
            self.th1_guas_fits[-1].SetDirectory(0)
        
        if self.__DEBUG__:
            inovs = self.in_overscan if hasattr(self,"in_overscan") else False
            self.draw_fit(None,[thist],True,inovs=inovs,**kwargs)
        
        if 'get_histogram' in kwargs:
            return mu,emu,sigma,esigma,chi2, thist

        return mu,emu,sigma,esigma,chi2
    
    def save_images_as_root(self,rawdata):

        ## INITIALIZE ROOT TTREE
        _outrootfile = ROOT.TFile(f"{rawdata.output}{self.__name__}_tree.root","RECREATE")
        tree = ROOT.TTree("pixels","pixels")

        # ROOT TTree with branches: row, col and for each tuple: MEAN, STD and PED
        meanimg = getattr(rawdata,"image_mean_compressed")
        stdflag = False
        psflag  = False
        neflag  = False


        # BRANCHES by default
        params = ['row','col','Q','STD'] if rawdata.n_amp==1 else ['row','col','Q_L','Q_U']
        
        if hasattr(rawdata,"image_std_compressed"):
            stdimg  = getattr(rawdata,"image_std_compressed")
            params += ['STD'] if rawdata.n_amp==1 else ['STD_L','STD_U']
            stdflag = True

        if hasattr(rawdata,"image_mean_compressed_pedestal_subtracted_e"):
            neimg = getattr(rawdata,"image_mean_compressed_pedestal_subtracted_e") 
            params += ['Ne'] if rawdata.n_amp==1 else ['Ne_L','Ne_U']
            neflag = True
        else:
            if hasattr(rawdata,"image_mean_compressed_pedestal_subtracted"):
                psimg = getattr(rawdata,"image_mean_compressed_pedestal_subtracted")
                params += ['psQ'] if rawdata.n_amp==1 else ['psQ_L','psQ_U']
                psflag = True

        # INITIALIZATION OF BRANCHES
        _branches = {}
        for bname in params:
            _branches[bname] = np.array( [-1], float)
            tree.Branch( bname, _branches[bname], f"{bname}/D")

        # FILL TTREE
        if rawdata.n_amp==2:
            for r in range(meanimg.shape[0]//2):
                for c in range(meanimg.shape[1]//2):
                    _branches['row'][0]  = r
                    _branches['col'][0]  = c
                    _branches['Q_L'][0]  = meanimg[r,c]
                    _branches['Q_U'][0]  = meanimg[r,meanimg.shape[1]-(c+1)]
                    if stdflag:
                        _branches['STD_L'][0] = stdimg[r,c]
                        _branches['STD_U'][0] = stdimg[r,meanimg.shape[1]-(c+1)]
                    if psflag:
                        _branches['psQ_L'][0] = psimg[r,c]
                        _branches['psQ_U'][0] = psimg[r,meanimg.shape[1]-(c+1)]
                    _ = tree.Fill()
        elif rawdata.n_amp==1:
            for r in range(meanimg.shape[0]):
                for c in range(meanimg.shape[1]):
                    _branches['row'][0] = r
                    _branches['col'][0] = c
                    _branches['Q'][0]   = meanimg[r,c]
                    #_branches['Q'][0]   = meanimg[r,meanimg.shape[1]-(c+1)]
                    if stdflag:
                        _branches['STD'][0] = stdimg[r,c]
                    if psflag:
                        _branches['psQ'][0] = psimg[r,c]
                    if neflag:
                        _branches['Ne'][0] = neimg[r,c]
                    _ = tree.Fill()

        tree.Write()
        _outrootfile.Close()
        print(f"    {self.__name__}: mean, std [and ped. sub.] pixel information has been saved as a ROOT.TTree")
        print(f"    {rawdata.output}{self.__name__}_tree.root")

        return

    @staticmethod
    def draw_fit(outfile_ovs_pcd,thist_list, verbose=False,inovs=False,**kwargs):
        if verbose:
            ROOT.gROOT.SetBatch(0)

        c = ROOT.TCanvas()
        ROOT.gStyle.SetPalette(ROOT.kBird)

        copt = 'PMC PLC '
        opt = copt
        htotal = thist_list[0]
        fitfunc = htotal.GetFunction("fitfunc")
 
        inovs_txt = " [in ovs]" if inovs else " [in active]"
        if 'xlabel' in kwargs:
            htotal.GetXaxis().SetTitle(kwargs['xlabel']+inovs_txt)
        else:
            htotal.GetXaxis().SetTitle('pixel charge'+inovs_txt)

        if 'ylabel' in kwargs:
            htotal.GetYaxis().SetTitle(kwargs['ylabel'])
        else:
            htotal.GetYaxis().SetTitle('counts')
        htotal.SetTitle('')
        htotal.Draw()
        
        c.Update()
        if verbose:
            c.Draw()
            input("press enter ... ")
        else:
            c.SaveAs(outfile_ovs_pcd)
        
        return


    def fit_gaussian_peak(self,image,nsig=7,normed=True,axis=0,distance=None,Npoints=50):
        """
        """
        ### function to get the spectral window to fit the gaussian
        def get_gauss_window(mu,mad,nsig):
            #if atzero:
            #    left_gauss_bound = image.min()
            #    return (left_gauss_bound, left_gauss_bound+nsig*mad)
            left_gauss_bound  = mu - nsig*mad
            right_gauss_bound = mu + nsig*mad
            return (left_gauss_bound,right_gauss_bound)

        ### allowing for masked arrays also
        image = image.compressed() if np.ma.is_masked(image) else image.flatten()

        ### STARTING POINTS FOR THE FIT
        #########################################################################
        mu_guess = np.ma.median(image)
        ### MAD or MAD as an estimator of the SIGMA?
        mad_guess = self.median_abs_deviation(image,scale=1) if self.use_mad else self.median_abs_deviation(image)
        
        ### SPECTRAL WINDOWS FOR THE FITTING
        #########################################################################
        gauss_bounds_win = get_gauss_window(mu_guess,mad_guess,nsig)

        ### LOOK FOR PEAKS IN THE SPECTRAL WINDOWN ONLY IF THE NUMBER OF POINTS IS LARGE ENOUGH
        #########################################################################
        if len(image)>Npoints:
            ### normalized histogram
            bins =int(np.sqrt(len(image)))
            y_counts,x_b = np.histogram(image,bins,range=gauss_bounds_win,density=normed)
        
            ### found peaks in the spectral window for the gaussian peak
            distance = distance or mad_guess*nsig
            peaks = np.array([],dtype=int)
            prominence_try=0.2
            n_tries=1
            while (peaks.size<1 and prominence_try>0 and n_tries<10000):
                prominence_try/=2.
                peaks, _ = find_peaks(y_counts,height=0,prominence=prominence_try,distance=distance)
                n_tries+=1
            
            if peaks.size==0:
                print("     WARNING. Peak not found, using the median and MAD values")
                self.exit = -1
                return mu_guess,mad_guess
            elif peaks.size >1:
                peaks = peaks[np.where( y_counts[peaks] == max(y_counts[peaks]))[0]]

            #### Once the peak have been found, fit a gaussian
            ##      fitting in a smaller window
            gauss_bounds_fit = get_gauss_window(x_b[peaks[0]],mad_guess,int(round(nsig/2.)))

        else:
            gauss_bounds_fit = gauss_bounds_win
        
        if self.__verbose__:
            print( " Fitting in the region: ", gauss_bounds_fit )
        
        (mu0, sigma0) = norm.fit(image[np.logical_and(image<gauss_bounds_fit[1],image>gauss_bounds_fit[0])])
        
        if __DEBUG__:
            print(" spectral window ", gauss_bounds_fit)
            print(" n. points ", len(image[np.logical_and(image<gauss_bounds_fit[1],image>gauss_bounds_fit[0])] ))
            plt.figure(10000)
            plt.clf()
            a = image[np.logical_and(image<gauss_bounds_fit[1],image>gauss_bounds_fit[0])]
            for yi in image:
                plt.axvline(yi,ymin=0,ymax=0.15, linestyle='dotted', alpha=0.5, color='#f20713')
            for yi in a:
                plt.axvline(yi,ymin=0,ymax=0.1, linestyle='dotted', alpha=0.5, color='black')
            _ = plt.hist(a,4,density=1,histtype='step')
            xf = np.arange(int(min(a)),round(max(a)))
            yf = norm.pdf(np.arange(int(min(a)),round(max(a))), loc=mu0,scale=sigma0)
            plt.plot(xf,yf,label=" mu={} ADCs, sigma={} ADCs".format(round(mu0,3),round(sigma0,3)))
            plt.legend()
            plt.show(block=True)
            input("....")

        #### outputs
        if self.__verbose__:
            print("  Peak Search Summary: ")
            if len(image)>50:
                print("     found {} peaks with a prominence {} in the range {}".format(peaks.size,prominence_try,gauss_bounds_win))
                print("     Calibration: ADU/e- ")
                print("     Peaks heights: ", y_counts[peaks])
                print("     Peaks at: ", x_b[peaks])
            print("  Gaussian Fit Summary:")
            print("     mu (pedestal): ", mu0)
            print("     error: ", sigma0)
            print("")
        
        #### if booked, plot individual gaussian fits
        if self.show_fit:
            if not normed:
                Norm = sum(y_counts)/np.sqrt(np.pi)
            else:
                Norm = 1

            if len(image)>Npoints:
                title = "Gaussian fit to the zero-charge pixel distribution"
                if not hasattr(self,'figGausFit'):
                    self.figGausFit = XYPlot(self.__fig_num__, "Gaussian fit to the zero-charge pixel distribution\n (over {})".format(
                        self.__axis_name__[axis]), xlabel=" ADUs ", ylabel="counts")
                    self.__fig_num__ +=1

                    self.figGausFit.Draw(x_b[1:],y_counts, ylog=True, xlog=False)
                    self.figGausFit.pdf(x_b, lambda x,p: p[2]*norm.pdf(x,loc=p[0],scale=p[1]), [mu0,sigma0,Norm], N=500)
                else:
                    self.figGausFit.alpha=0.5
                    self.figGausFit.append(x_b[1:],y_counts, pdf=(x_b, lambda x,p: p[2]*norm.pdf(x,loc=p[0],scale=p[1]),[mu0,sigma0,Norm]), rand_color=True)
       
        self.exit = 0
        return (mu0,sigma0)
    
    def get_med_and_mad(self,data,axis=None,mad=True,median=True):
        """
        """
        keepdims=None
        if axis==1:
            keepdims=1
        if median:
            dmed = np.ma.median( data, axis=axis, keepdims=keepdims )
        else:
            if axis is not None:
                dmed = np.ma.mean( data, axis=axis, keepdims=keepdims )
            else:
                dmed = np.ma.mean( data )


        if mad:
            dmad = np.ma.median(np.ma.abs(data - dmed), axis=axis )
        else:
            dmad = np.ma.std( data, axis=axis )
        
        #### shape according to data
        if axis is not None:
            if len(data.shape)>1:
                if axis == 1:
                    dmed = dmed.reshape(len(dmed),1)
                    dmad = dmad.reshape(len(dmad),1)
                else:
                    dmed = dmed.reshape(1,len(dmed))
                    dmad = dmad.reshape(1,len(dmad))

        if np.ma.isarray(data):
            return dmed, dmad
        else:
            return np.array(dmed), np.array(dmad)

    @abstractmethod
    def execute_process(self):
        """Metaclass abstract method to be implemented for each specific
        process.
        """
        raise NotImplementedError()
 
