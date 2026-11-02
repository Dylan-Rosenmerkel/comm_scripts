""":mod:`pysimdamicm.analysis.utils.plotlib`
    
    XXX Description of the moduele XXX
    
    Module of the main methods for the analysis of a CCD image      
    
.. moduleauthor:: Nuria Castello-Mor
"""

### plot style
from pysimdamicm.utils.plotLib import GetROOTPalette

from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt
from matplotlib import rc
from cycler import cycler
import numpy as np
from random import randint
from scipy.stats import norm as gaussian_fit

def update_mpl_style():
    rc('text', usetex=True)
    rc('font', family="sans-serif")
    rc('xtick',direction='in')
    rc('ytick',direction='in')
    rc('legend', loc='best', frameon=False,markerscale=0.8)
    rc('axes',titlepad=20)
#    rc('lines', markeredgecolor='#043bd1')
    rc('lines', markerfacecolor='None' )
    rc('font',**{'family':'DejaVu Sans', 'serif':['Times'], 'size':12})
#    root_palette_color_list = GetROOTPalette(dtype='list')
#    rc('axes', prop_cycle=cycler('color',root_palette_color_list))

    rc('figure', max_open_warning = 0)
    return
        
class ABCFigure(metaclass=ABCMeta):
    def __init__(self,index,htitle,xlabel,ylabel,random=False,Ncolors=10,labelsize=14,fontsize=14,**kwargs):
        self.__axis_legend__    = {0: 'rows', 1: 'columns'}
        ### set default style for plots
        if random:
            random_color_list = ['#{:06x}'.format(randint(0, 256**3)) for i in range(int(Ncolors))]
            self.color_cycle = cycler('color',random_color_list)
        else:
            root_palette_color_list = GetROOTPalette(dtype='list')
            self.palette = GetROOTPalette(dtype='palette')
            self.color_cycle = cycler('color',root_palette_color_list[::10])

        update_mpl_style()
        rc('axes', prop_cycle=self.color_cycle)

        #### default options for plotting
        self.alpha = 0.8
        self.marker = 'o'
        self.linestyle ='None'
        self.markerfacecolor = 'None'
        self.markeredgecolor = '#043bd1'
        self.color = '#043bd1'
        self.linewidth = 1
        self.markersize = 2

        ### Set figure
        self.figure = plt.figure(index,figsize=(8,8))
        self.figure_id = index

        self.fontsize = fontsize
        self.title  = htitle
        self.xlabel = xlabel
        self.ylabel = ylabel

        plt.title(htitle)

    @property
    def xlabel(self):
        return self.__xlabel
    @xlabel.setter
    def xlabel(self,val):
        self.__xlabel = val
        self.cd()
        plt.xlabel(val)

    @property
    def ylabel(self):
        return self.__ylabel
    @ylabel.setter
    def ylabel(self,val):
        self.__ylabel = val
        self.cd()
        plt.ylabel(val)

    @property
    def title(self):
        return self.__title
    @title.setter
    def title(self,val):
        self.__title = val
        self.cd()
        plt.title(val)

    @property
    def ylog(self):
        return self.__ylog
    @ylog.setter
    def ylog(self,val):
        self.cd()
        if val:
            plt.yscale('log')
        else:
            plt.yscale('linear')

    @property
    def xlog(self):
        return self.__xlog
    @ylog.setter
    def xlog(self,val):
        self.cd()
        if val:
            plt.xscale('log')
        else:
            plt.xscale('linear')

    @property
    def xlim(self):
        return self.__xlim
    @ylog.setter
    def xlim(self,val):
        self.cd()
        plt.xlim(val[0],val[1])

    @property
    def ylim(self):
        return self.__ylim
    @ylog.setter
    def ylim(self,val):
        self.cd()
        plt.ylim(val[0],val[1])

    @property
    def labelsize(self):
        return self.__labelsize
    @labelsize.setter
    def labelsize(self,val):
        self.__labelsize = val
        rc('xtick', labelsize=val)
        rc('ytick', labelsize=val)

    @property
    def fontsize(self):
        return self.__fontsize
    @fontsize.setter
    def fontsize(self,val):
        self.__fontsize = val
        rc('font',**{'family':'DejaVu Sans', 'serif':['Times'], 'size':val})

    def cd(self):
        plt.figure( self.figure.number )

    def DrawLegend(self):
        plt.legend(loc=0)

    def SaveAs(self,output,fig_dpi=100):
        self.cd()
        plt.savefig(output, dpi=fig_dpi)

    def GetRandomHEXColorList(self,n):
        return ['#{:06x}'.format(randint(0, 256**3)) for i in range(int(n))]

    def resize(self,val):
        self.cd()
        self.figure.set_size_inches(val)


    @abstractmethod
    def Draw(self):
        """Metaclass abstract method to be implemented for each specific
        process.
        """
        raise NotImplementedError()


##############################################################################################

class PCDPlot(ABCFigure):
    def __init__(self,index,htitle='Pixel Charge Distribution',xlabel='pixel charge [ADU]',ylabel='counts',
            random=True,histtype='step',**kwargs):
        super().__init__(index,htitle,xlabel,ylabel,random,**kwargs)

        self.histtype = histtype
        self.ylog = True
        #plt.suptitle(htitle)
        #plt.tight_layout(rect=[0.01, 0.01, 1, 0.9], pad=3)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def Draw(self,image,region=None,normed=False,Nbins=None,legend=None,dofit=False):
        
        self.cd()
        
        ### image can be masked array or a simple array
        if np.ma.is_masked(image):
            data = image.compressed()
        else:
            data = image.ravel()
        
        ### number of bins
        Nbins = Nbins or int(np.sqrt(data.size))
       
        ### plotting the histogram
        _=plt.hist( data,bins=Nbins, range=region, density=normed,
                histtype=self.histtype, alpha=self.alpha, label=legend)
        ### fit data to exponential
        if dofit:
            print( " Fitting to a gaussian " )
            mu,std = gaussian_fit.fit(data)
            if region is None:
                region = (min(data),max(data))
            x_fit = np.linspace(region, Nbins)
            _ = plt.plot(x_fit,gaussian_fit.pdf(x_fit),'k', linewidth=2, color='red', 
                    label="fit: $\mu={}$, $\sigma={}$".format(mu,std))
            _ = plt.legend()

            return mu,std
        return


class PedestalPlot(ABCFigure):
    def __init__(self,index,htitle=None, xlabel='column',ylabel='pedestal',
            random=False,**kwargs):
        if htitle is None:
            htitle = '{} vs {}'.format(ylabel,xlabel)
        super().__init__(index,htitle,xlabel,ylabel,random,**kwargs)
        
        self.fig_size = (14,5)

    def Draw(self,pedestal_list, axis=[0,1], legend=None, mean_line=True, std_line=True,
            mad_line=False):
        self.cd()

        subplot = False
        if len(axis)>1:
            self.figure.set_size_inches( self.fig_size )
            subplot = True
        
        kopt = {'markeredgecolor':self.markeredgecolor,
                'markerfacecolor':self.markerfacecolor,
                'marker':self.marker,
                'linestyle':self.linestyle,
                'markersize':self.markersize,
                'linestyle':self.linestyle,
                'linewidth':self.linewidth
                }
        for ped,ax in zip(pedestal_list,axis):
            if subplot:
                plt.subplot(1,len(axis),ax+1)

            _ = plt.plot( ped, **kopt, label=legend )

            self.xlabel = "{} number".format(self.__axis_legend__[ax])
            ped_mean = np.mean(ped)
            if mean_line:
                plt.axhline(ped_mean, color='#5b5f69', linestyle='dashed', linewidth=1, label='mean', alpha=0.8)
                plt.legend()
            if std_line:
                ped_std = np.std(ped)
                plt.axhline(ped_mean+ped_std , color='#f27907', linestyle='dotted', linewidth=1, label='std', alpha=0.8)
                plt.axhline(ped_mean-ped_std , color='#f27907', linestyle='dotted', linewidth=1, alpha=0.8)
                plt.legend()
            if mad_line:
                ped_mad = np.median(abs(ped - np.median(ped)))
                plt.axhline(ped_mean+ped_mad, color='#07ab51', linestyle=(0,(3,10,1,10,1,10)), linewidth=1, label='MAD', alpha=0.8)
                plt.axhline(ped_mean-ped_mad, color='#07ab51', linestyle=(0,(3,10,1,10,1,10)), linewidth=1, alpha=0.8)
                plt.legend()


class ImagePlot(ABCFigure):
    def __init__(self,index,htitle=None, xlabel='columns',ylabel='rows',
            vmin=None,vmax=None,nrows=1,ncols=1,**kwargs):
        super().__init__(index,htitle,xlabel,ylabel,**kwargs)
        
        self.vmin = vmin
        self.vmax = vmax
        self.nrows = nrows
        self.ncols = ncols
        self.nplots = ncols*nrows
        
        plt.suptitle( htitle )

        self.done = False

    def Draw(self,image, subtitle=['raw image'], fontsize=12,histequ=False):
        self.cd()
        
        if self.done:
            ### to prevent replotting in the same image (not allowed for this class)
            return

        #if self.nplots==1:
        #    image = [image]

        subtitle_flag=False
        if len(image) == len(subtitle):
            subtitle_flag = True

        for index, img in enumerate(image):
            plt.subplot(self.nrows,self.ncols, index+1)
            if subtitle_flag:
                plt.gca().set_title(subtitle[index],fontsize=fontsize, pad=3)
            if histequ:
                #img[np.where(img>np.median(img))] = np.median(img)
                #img = equalize_hist(img)
                img,cdf = self.equalize_histogram( img )

            ip = plt.imshow( img, cmap=self.palette, aspect='auto', vmin=self.vmin, vmax=self.vmax )
            plt.colorbar()

        if self.title is not None:
            try:
                plt.tight_layout(rect=[0.01, 0.01, 1, 0.9], pad=3)
            except TypeError:
                pass
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

        ### to prevent replotting in the same image (not allowed for this class)
        self.done = True

    def equalize_histogram(self,im,nbr_bins=None):
        """
        """
        nbr_bins = nbr_bins or int(np.sqrt(im.size))*5
        #get image histogram
        imhist,bins = np.histogram(im.flatten(),nbr_bins,density=True)
        cdf = imhist.cumsum() #cumulative distribution function
        cdf = cdf / cdf[-1] #normalize
        
        #use linear interpolation of cdf to find new pixel values
        im2 = np.interp(im.flatten(),bins[:-1],cdf)
        
        return im2.reshape(im.shape), cdf

   
 

class PCDSingleSkipsPlot(ABCFigure):
    def __init__(self,index,htitle='Single Skips Pixel Charge Distribution',xlabel='single skip charge [ADU]',ylabel='n. skips',
            random=True,histtype='step',**kwargs):
        super().__init__(index,htitle,xlabel,ylabel,random,**kwargs)

        self.histtype = histtype
        self.alpha = 0.4
        self.reshape = (14,5)
        self.ylog = True

    def Draw(self,image,Nskips,axis_to_skip=1,normed=False,Nbins=None,legend=None):
        self.cd()
        
        ### allowing for masked images
        func_to_flatten = getattr(np, 'ravel')
        if np.ma.is_masked(image):
            func_to_flatten = getattr(np, 'compressed')
        ### reshape to flat the array into single images
        nrows,ncols = image.shape
        image_reshape = np.reshape(image,(nrows,int(ncols/Nskips),Nskips))
        data_list = [func_to_flatten(image_reshape[:,:,i]) for i in range(Nskips)]
        
        Nbins = Nbins or int(np.sqrt(len(data_list[0])))
        ### plotting the list of histograms
        #cmap = GetRandomHEXColorList(Nskips)
        cmap = plt.cm.get_cmap(plt.cm.viridis,int(Nskips))
        _=plt.hist(data_list,bins=Nbins,density=normed,
                histtype=self.histtype,alpha=self.alpha, color=cmap.colors)
 


class XYPlot(ABCFigure):
    def __init__(self,index,htitle,xlabel,ylabel,nplots=1,**kwargs):
        super().__init__(index,htitle,xlabel,ylabel,**kwargs)
        
        self.nplots = nplots
        if nplots>1:
            plt.suptitle(htitle)

    def Draw(self,XL,YL=None,subtitles=[],nrows=1,ncols=1, ylog=False,xlog=False,
            fontsize=12,Xerr=None,Yerr=None,xlabel=None,ylabel=None):
        self.cd()
        
        if self.nplots==1:
            XL = [XL]
            if not YL is None:
                YL = [YL]
        
        subtitle_flag=False
        if len(XL) == len(subtitles):
            subtitle_flag = True

        for ind, X in enumerate(XL):
            if self.nplots>1:
                plt.subplot(nrows,ncols,ind+1)
            if subtitle_flag:
                plt.gca().set_title(subtitles[ind],fontsize=fontsize, pad=5)
            if not YL is None:
                xerr = Xerr[ind] if Xerr is not None else None
                yerr = Yerr[ind] if Yerr is not None else None
                plt.errorbar( X,YL[ind],xerr=xerr,yerr=yerr, marker=self.marker, linestyle=self.linestyle,
                        alpha=self.alpha, markeredgecolor=self.markeredgecolor, 
                        markerfacecolor=self.markerfacecolor )
            else:
                plt.plot( X, marker=self.marker, linestyle=self.linestyle, alpha=self.alpha,
                        markeredgecolor=self.markeredgecolor, markerfacecolor=self.markerfacecolor )

            if type(xlabel) is list:
                plt.xlabel(xlabel[ind])
            else:
                plt.xlabel(self.xlabel)
            if type(ylabel) is list:
                plt.ylabel(ylabel[ind])
            else:
                plt.ylabel(self.ylabel)

            if ylog:
                plt.yscale('log')
            if xlog:
                plt.xscale('log')
        
        if len(subtitles)>1 or self.nplots>1:
            #plt.tight_layout(rect=[0.001, 0.009, 0.99, 0.9], pad=2)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
        return

    def pdf(self,x,func,param,N=500,label=None,color='black'):
        """Add a line following function 'func'
        """
        x = np.linspace(min(x),max(x),N)

        self.cd()
        plt.plot(x, func(x,param), '--', color=color, label=label, alpha=self.alpha)
        if label is not None:
            self.DrawLegend()
        
        return

    def append(self,x,y,pdf=None,rand_color=False,marker='o',s=8,label=None):
        if self.nplots>1:
            return

        if rand_color is not None:
            color=color_line=self.GetRandomHEXColorList(1)[0]
        else:
            color=self.markeredgecolor
            color_line='black'

        self.cd()
        if pdf is not None:
            if len(pdf)>3:
                label=pdf[3]
            else:
                label=None
            self.pdf( pdf[0], pdf[1], pdf[2],label=label,color=color)
        sp = plt.scatter(x,y,marker=marker,s=s,alpha=self.alpha, color=color)

        if label is not None:
            self.DrawLegend()
        return




