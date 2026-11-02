#!/usr/bin/env python
""":mod:`plots` -- pre-defined plots for debug mode, ROOT styles
================================================================
    
    .. moduleauthor:: Nuria Castello-Mor <castello@ifca.unican.es> 

"""

from pysimdamicm.data.elements import ELEMENTS
from pysimdamicm.utils.root_plot_styles import get_sed_style

from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from cycler import cycler

import numpy as np
import pandas as pd

import ROOT
_PRINT = True


######################################################################################
# SOME DRAWING PROPERTIES
######################################################################################
#particle_colors_cmap = plt.get_cmap("tab20")
particle_colors_cmap = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
        (1.0,0.4980392156862745, 0.054901960784313725),
        (0.2235294117647059, 0.23137254901960785, 0.4745098039215686),
        (0.3215686274509804, 0.32941176470588235, 0.6392156862745098),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
        ]
particle_colors = lambda pdg: particle_colors_cmap[abs(pdg)%len(particle_colors_cmap)]

particle_markers_list = ['P','v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'o', 'd','D','+','4']
particle_markers = lambda pdg: particle_markers_list[abs(pdg)%len(particle_markers_list)]


######################################################################################
#   FUNCTIONS FOR STYLE
######################################################################################
def SetMatplotStyle(labelsize=12,fontsize=16, IsInteractive=True):
    from matplotlib import rc

    rc('xtick', labelsize=labelsize) 
    rc('ytick', labelsize=labelsize) 
    rc('font',**{'family':'DejaVu Sans', 'serif':['Times'], 'size':fontsize})
    rc('text', usetex=True)
    rc('xtick',direction='in')
    rc('ytick',direction='in')
    rc('legend', loc='best', frameon=False,markerscale=0.8)
    
    root_palette_color_list = GetROOTPalette(dtype='list')
    rc('axes', prop_cycle=cycler('color',root_palette_color_list[::10]) )

    return

def GetROOTPalette(ID=ROOT.kBird, dtype='palette'):
    from matplotlib.colors import LinearSegmentedColormap    
    ROOT.gStyle.SetPalette(ID)
    
    hexPalette = []
    for i in range(ROOT.gStyle.GetNumberOfColors()):
        ci=ROOT.gStyle.GetColorPalette(i)
        hexPalette.append(ROOT.gROOT.GetColor(ci).AsHexString())

    if dtype in ['palette']:
        cm = LinearSegmentedColormap.from_list(str(ID),hexPalette)
        return cm

    return hexPalette

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_list_of_colors(Ncolors,root_palette_id=ROOT.kBird, dtype="hex"):
    RTC = ROOT.TColor
        
    hex_colors = GetROOTPalette(root_palette_id, 'hex')

    hex_ind = range(0,len(hex_colors),int(len(hex_colors)/Ncolors))
    root_colors = []

    if dtype=="hex":
        for ind in hex_ind:
            root_colors.append(hex_colors[ind])
        return root_colors
    
    for ind in hex_ind:
        root_colors.append(RTC().GetColor(hex_colors[ind]))
    
    return root_colors

######################################################################################
#
#   PLOT CLASS for the SPECTRAL BACKGROUND PLOT
#
######################################################################################
colors=['Red','Orange','Magenta','Azure','Cyan','Green','Blue','Violet','Yellow','Spring','Teal','Blue','Violet','Pink']
palette = [51,113]

color_wheel = { 
    'Red'    : ( 632-3,[0+3,1+3,2+3,3+3,4,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]),
    'Orange' : ( 800+1,[0-7,1-7,2-7,3-7,4-7,5-7,6-7,7-7,8-7,9-7,10-7,-1-7,-2-7,-3-7,-4-7,-5-7,-6-7,-7-7,-8-7,-9-7]),
    'Magenta': ( 616-6,[0,1,2,3,4,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]),
    'Azure'  : ( 860-3,[0+3,1+3,2+3,3+3,4+3,5+3,6+3,7+3,8+3,9+3,10+3,-1+3,-2+3,-3+3,-4+3,-5+3,-6+3,-7+3,-8+3,-9+3]),
    'Cyan'   : ( 432-2,[0-3,1-3,2-3,3-3,4-3,-1-3,-2-3,-3-3,-4-3,-5-3,-6-3,-7-3,-8-3,-9-3,-10-3]),
    'Green'  : ( 416-5,[0,1,2,3,4,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]),
    'Yellow' : ( 400-5,[0,1,2,3,4,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]),
    'Spring' : ( 820+2,[0,1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4,-5,-6,-7,-8,-9]),
    'Teal'   : ( 840+4,[0,1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4,-5,-6,-7,-8,-9]),
    'Blue'   : ( 600-8,[0,1,2,3,4,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]), 
    'Violet' : ( 880+5,[0,1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4,-5,-6,-7,-8,-9]),
    'Pink'   : ( 900+3,[0,1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4,-5,-6,-7,-8,-9]),
    }


######################################################################################
#
#   SETUP BACKGORUND COMPARISON PLOT
#
######################################################################################


def SetupsComparisonPlot( epsout, hist_total_list, 
        add_total=False,
        Emin=0.0, Emax=10.0, Nbins=66, 
        ylim=(1e-4,10),
        header="",
        lcolor=[ROOT.kAzure-3,ROOT.kTeal+4,ROOT.kOrange+8,ROOT.kOrange-6,ROOT.kViolet-7,ROOT.kAzure+1]):
    
    # set style 
    sedstyle=get_sed_style()
    sedstyle.cd()
    
    canv = ROOT.TCanvas("comparison")

    # add legend
    l_space=0.06
    Nhist=len(hist_total_list.keys())
    n_lines=Nhist+1
    y_max=0.99
    y_min=y_max-n_lines*l_space
    if y_min<0.1:
        y_min=0.15
    leg = ROOT.TLegend(0.82,y_min,0.99,y_max)

    ### Initialize an histogram for the total background (if set)
    if add_total:
        htotal = ROOT.TH1F("htotal","",Nbins,Emin,Emax)
    
    ### For each setup plot its SED
    kopt="HIST"
    dru_colors = {}
    for i,key in enumerate(hist_total_list.keys()):
        try:
            c=lcolor[i]
        except IndexError:
            c+=10

        hist_total_list[key].SetLineColor(c)
        hist_total_list[key].SetLineStyle(i+1)
        hist_total_list[key].Draw(kopt)
        leg.AddEntry(hist_total_list[key],key.split(":")[-1],"L")

        kopt="HISTsame"

        if add_total:
            htotal.Add(hist_total_list[key])
        
        dru_colors[key] = ROOT.gROOT.GetColor(c).AsHexString()

    if add_total:
        htotal.Draw("HISTsame")

    leg.SetHeader(header)
    leg.Draw()

    canv.SetLogy()
    canv.Update()
    canv.Modified()

    _=canv.SaveAs(epsout)
    
    return dru_colors


class SpecBckPlot(object):
    def __init__(self, saveas, xlabel="E_{cluster} [keV]", ylabel="clusters [/kg/day/keV]",
            line=True ):
        """

        Input
        ======
            saveas      output file name

        Optional
        ========
            xlabel      label for the x-coordenate
            ylabel      label for the y-coordenate
            line        marker or lines?

        """
        
        self.xlabel = xlabel
        self.ylabel = ylabel

        if line:
            self.line = True
            self.marker = False
        else:
            self.marker = True
            self.line = False

        self.saveas = saveas  

    def Append(self, contaminants, total=False ):
        """
        Append into histos all the histograms from contaminats, as well as, its label for the legend

        inputs
        =======
        contaminants    dictionary where the key is the legend for the histogram contained as a
                        value of this keyword, i.e. {'isotope 238U': TH1F }

        """
        if not hasattr(self,"histos"):
            self.histos = []
            self.legend = []
            self.Nhist = 0

        print( "  Adding contaminants: " )
        for key in contaminants.keys():
            try:
                A = int(key.split("a")[0])
                Z = int(key.split("a")[-1].split("z")[0])
                element = ELEMENTS[Z]
                leg_label = r'^{'+str(A)+'}'+element.symbol
                if element.symbol == "Pa" and str(A)=="234":
                    leg_label=r'^{'+str(A)+'m}'+element.symbol
                print(" \t\t ", key,  element.symbol, element.name )

            except ValueError:
                print(" \t\t ", key )
                leg_label = key
            
            self.histos.append( contaminants[key] )
            self.legend.append( leg_label )
            
        # total number of histograms 
        self.Nhist = len(self.histos)

    def GetTotalBkgSpec(self, lofhistos=None, Nbin=None, Emin=None, Emax=None):
        """
        Append into histos all the histograms from contaminats, as well as, its label for the legend

        inputs
        =======
        contaminants    dictionary where the key is the legend for the histogram contained as a
                        value of this keyword, i.e. {'isotope 238U': TH1F }

        """
        
        # lofhistos
        if lofhistos is None:
            lofhistos = self.histos

        # set TH1F for the TOTAL BKG
        if Nbin is None:
            Nbin = lofhistos[0].GetNbinsX()
            Emin = lofhistos[0].GetXaxis().GetXmin()
            Emax = lofhistos[0].GetXaxis().GetXmax()
            
            
        self.htotal = ROOT.TH1F( "htotal","", int(Nbin), Emin, Emax )
        self.htotal.SetDirectory(0)
        self.htotal.SetLineWidth(3)
        self.htotal.SetLineColor(923)
        self.htotal.SetLineStyle(1)
                
        for h in lofhistos:
            self.htotal.Add( h )

        
    def Fit(self, Xmin=None, Xmax=None, fitfunc="pol1", legend=None, precision=3):
        """
        Each histogram of the attribute list <histos> is fitted to the function <fitfunc> 
        in the x range (Xmin,Xmax). The results of the fits are given with a precission 
        of <precision> digits
        
        Create also two attributes:
            integral_6keV  = {'fit':[], 'data':[]}
            integral_2keV  = {'fit':[], 'data':[]} 

        where the integral over the energy ranve 0-6 keV and 0-2keV is sotred. For each 
        histogram two values of the integral are computed, one using the fitted function and 
        the other by using the data from the histogram


        inputs
        ======
            Xmin        min value of the Energy range to define the fit TH1F fit function
            Xmax        max value of the Energy range to define the fit function
            fitfunc     function to use to fit the histogram over the energy range (Xmin-Xmax)
            legend
            precision   number of digits to include the fitted parameter (y_intersection) to the legend
        """
        
        self.fitfunc = []
        self.integral_6keV = {'fit':[],'data':[]}
        self.integral_2keV = {'fit':[],'data':[]}
        self.fit_p0 = []

        if not Xmin == None:
            xmin = Xmin
        else:
            xmin = self.Emin

        if not Xmax == None:
            xmax = Xmax
        else:
            xmax = self.Emax
        if not legend == None:
            self.legend_fit = []

        for k,hist in enumerate(self.histos):

            _ = hist.Fit( "fitfunc", fitfunc )
            self.fitfunc.append( hist.GetFunction( fitfunc ) )
            
            self.fit_p0.append( self.fitfunc[k].GetParameter(0) )

            self.integral_6keV['fit'].append( self.fitfunc[-1].Integral(0.0,6.0) )
            self.integral_2keV['fit'].append( self.fitfunc[-1].Integral(0.0,2.0) )
            
            #### from data
            self.integral_6keV['data'].append( hist.Integral(hist.FindBin(0.0),hist.FindBin(6.0),"width" ))
            self.integral_2keV['data'].append( hist.Integral(hist.FindBin(0.0),hist.FindBin(2.0),"width" ))

            if not legend == None:
                self.legend_fit.append( legend[k] + " (#approx {0} dru)".format(
                    round(self.integral_2keV['data'][-1],precision)) )

            if _PRINT:
                msm = """
                Fit {0}
                    at E=0keV       {1}
                    slope           {2}

                    integral       fit:{3}, data:{4} (0-6keV, with dE={5})
                """.format(k, self.fitfunc[-1].GetParameter(0), self.fitfunc[-1].GetParameter(1),
                        self.integral_6keV['fit'][-1],self.integral_6keV['data'][-1], self.dE)

                print(msm)

    def Draw(self, ylim=None, legend=True, theader=None, 
            addfit=False, addtotal=True, 
            formats=[".eps", ".root",".png"], squared=False, stat_off=True):
        """
        Plot in a single canvas all the histograms stored at <histos> and will be
        stored in the formats specified for <formats>
        """

        # set style 
        sedstyle=get_sed_style()
        sedstyle.cd()
        
        # set colors, markers, ... for all histograms into self.histos
        self.SetStyle(ylim)
        self.SetColorFromLegend(self.legend)

        # add legend
        c = ROOT.TCanvas("c")
        l_space=0.06
        n_lines=int((self.Nhist/2)+1)
        y_max=0.99
        y_min=y_max-n_lines*l_space
        if y_min<0.1:
            y_min=0.15
        leg = ROOT.TLegend(0.82,y_min,0.99,0.95)
        leg.SetNColumns(2)

        # update text for legend if is the case
        if addfit:
            self.legend = self.legend_fit
        
        print(theader,"____________________________________________")
        if not theader is None:
            leg.SetHeader(theader)

        # plot all histograms 
        _DOpt = "HISTO"
        dopt = "HISTO"
        for k, hist in enumerate( self.histos ):
            if self.legend[k].count("all")>0:
                ktotal = k
                continue
            hist.Draw( dopt )
            leg.AddEntry( hist, self.legend[k], "L")
            dopt = _DOpt+"same"
            
        if addtotal:
             self.histos[ktotal].SetLineColor(923)
             self.histos[ktotal].SetLineStyle(1)             
             self.histos[ktotal].Draw(_DOpt+"same")
             leg.AddEntry( self.histos[ktotal], "total", "L")

        if legend:
            leg.Draw()

        c.SetLogy()
        c.Update()
        c.Modified()

        for outformatplot in formats:
            outfile = self.saveas.split(".")[0]
            _ = c.SaveAs( outfile+outformatplot )

    def SetStyle( self, ylim=None, mlsize=2):
        """
        Set style of line/markers for each of the histograms at <histos>
        """

        k = 0
        kk = 0
        kseed = 2
        for seed in range(self.Nhist):
            if seed >= len(colors):
                k = 0
                kk +=1

            color = color_wheel[colors[k]][0] + color_wheel[colors[k]][1][kk]
            k+=1

            if self.line:
                self.histos[seed].SetLineColor( color )
                if kseed+seed > 10:
                    kseed = 2-seed
                self.histos[seed].SetLineStyle( kseed+seed )
                self.histos[seed].SetLineWidth( mlsize )
            else:
                self.histos[seed].SetMarkerColor( color )
                self.histos[seed].SetMarkerStyle( 20+seed )
                self.histos[seed].SetMarkerSize( mlsize )

            if ylim is not None:
                self.histos[seed].SetAxisRange( ylim[0], ylim[1], "Y" )
            self.histos[seed].SetXTitle( self.xlabel )
            self.histos[seed].SetYTitle( self.ylabel )

            #print("colors ....", color_wheel[colors[k]][0] , color_wheel[colors[k]][1][kk] )

    def SetColorFromLegend( self, legend=None ):
        """
        Set style of line/markers for each of the histograms at <histos>
        """

        for k,key in enumerate(legend):
            try:
                try:
                    A = int(key.split("}")[0].split("{")[1])
                except ValueError:
                    A = int(key.split("m}")[0].split("{")[1])
                Zsymbol = key.split("}")[1]
                element = ELEMENTS[Zsymbol]
                color = element.isotopes[A].color
            except IndexError:
                pass
          
            self.histos[k].SetLineColor( color )
            self.histos[k].SetMarkerColor( color )


######################################################################################
#
#   Pre-Defined plots for the DEBUG mode in simpydam.bin.simlCCDimg
#
#####################################################################################
def scatter( X,Y, E=None, markStyle=None, markSize=None, markColor=None, minSize=90.0,
        legend=False, leglabel=None, fig_title=None, grid=False ):
    """

    NN dimensional plot

    inputs
    ======
        
        X               list of arrays 
        Y               list of arrays with the same shape as X
        markStyle       list of styles (str) for each ndarray on X
        markSize        list of sizes (int) for each ndarray on X
        markColor       list of colors (str) for each ndarray on X
        minSize         minimum size for the  first ndarray, the following marker will have minSize+n*0.4
        legend          set if legend has to be plotted
        leglabel        list of legend (format latext text, str) for each ndarray on X

    """
    fig = plt.figure(fig_title)
    if grid:
        from matplotlib.ticker import MultipleLocator
        spacing = 1
        minorLocator = MultipleLocator(spacing)
        ax=fig.subplots(1,1)
    else:
        ax=fig.subplots(1,1)

    if not type(X) is list:
        X = list(X)
        Y = list(Y)

    #plt.style.use('ggplot')
    colors_flag = False
    if type(E) is list:
        colors_flag = True

    Nplots = len(X)
    if not type(markStyle) == list:
        markStyle = [".","s","o","D","^"] 
        markSize  = [minSize+3.0,minSize+5.0,minSize+7.0,minSize+9.0,minSize+10.0]
        markColor = ["#000000","#FF4500","#3A5FCD","#8B008B","#228B22"]

        if Nplots > 5:
            for k in range(Nplots-5):
                markStyle.append( list(MarkerStyle.markers.keys())[6+k] )
                markSize.append( minSize + 12.0 + k*0.4 )
                markColor.append( rand(3) )

    k = 0
    for x,y in zip(X,Y):
        if colors_flag:
            markColor[k] = E[k]
            markSize[k] = 80.

        if legend:
            _p = ax.scatter(x,y, marker=markStyle[k], edgecolors=markColor[k], s=markSize[k], facecolors='none', label=leglabel[k])
            fig.legend()
        else:
            _p = ax.scatter(x,y, edgecolors=markColor[k], marker=markStyle[k], facecolors='none', cmap=GetROOTPaletteAsHex() )
        k+=1

    plt.xlabel( " X ", fontsize=20)
    plt.ylabel( " Y " , fontsize=20)
    Y = np.array(Y)
    X = np.array(X)
    if grid:
        ax.yaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.grid(which='minor')

    if colors_flag:
        plt.colorbar(_p)

    plt.show()
    del(fig)

    return

def plotImages(eventid,ccdid,img_pixel_noise_e,img_dark_current_e,img_adu,X):
    """

    plot a list of images on the DEBUGIN mode of simdamicm.bin.simulCCDimg

    """


    #plt.style.use('classic')
    plt.rcParams.update({'font.size': 20})

    fig = plt.figure("Cluster Reconstruction event={0} && ccd={1}".format(eventid,ccdid),
            figsize=(25,9))

    plt.subplots_adjust(wspace=0.40)

    plt.subplot(2,2,1,aspect='equal')
    dummy = plt.imshow( img_pixel_noise_e, origin='lower' )
    plt.colorbar(dummy,fraction=0.046, pad=0.04)
    plt.ylabel(" pixel noise [e/pixel]")

    plt.subplot(2,2,2,aspect='equal')
    dummy = plt.imshow( img_dark_current_e, origin='lower' )
    plt.colorbar(dummy,fraction=0.046, pad=0.04)
    plt.ylabel(" dark current [e/pixel/exposure]")

    plt.subplot(2,2,3,aspect='equal')
    dummy = plt.imshow( img_adu, origin='lower'  )
    plt.colorbar(dummy,fraction=0.046, pad=0.04)
    plt.ylabel(" Edep [ediff,noise,dc,saturation] ")

    plt.subplot(2,2,4,aspect='equal',adjustable='box-forced')
    Xmin,Xmax = X[0].min()-10, X[0].max()+10
    Ymin,Ymax = X[1].min()-10, X[1].max()+10
    dummy = plt.imshow(img_adu[ Ymin:Ymax, Xmin:Xmax ], origin='lower' )
    plt.colorbar(dummy,fraction=0.046, pad=0.04)
    plt.ylabel(" ZOOM IN " )

    plt.show()

    print(" Debug Info")
    print(" Simulated pixel noise and dark current images, min and max values: ")
    print(" PIXEL NOISE {0}--{1} e/pixel".format(img_pixel_noise_e.min(),img_pixel_noise_e.max()) )
    print(" DARK CURRENT {0}--{1} e/pixel/exposure".format(img_dark_current_e.min(),img_dark_current_e.max()) )
    print(" ")

    del(fig)

    return

######################################################################################
#
#
#
######################################################################################

#from matplotlib import rc
#rc('xtick', labelsize=12) 
#rc('ytick', labelsize=12) 
#rc('font',**{'family':'DejaVu Sans','serif':['Times'],'size':18})
#rc('text', usetex=True)

def create_tripad_plot_sns( hcolz, xlabel, ylabel, saveas, draw_opt=None, is_tree=True,
        is_profile=False, 
        fig_size=(10,10), 
        img_size=(0.6,0.6), hist_fract_size=0.32, 
        img_left_bottom=(0.1,0.1), d_btw_subplots=0.01,
        fig_title="",fig_title_pos=(0.6,0.94),
        fill_hist=False,
        set_hist_ylog=(True,True),
        show=False,
        vmin=1.0,
        fig_dpi=300,
        cmap='plasma',
        **HOpt):
    """Prepare the canvas with three TPads and create inside a 2-dim map plot in 
    the main pad (padup), and its X (paddown) and y-projectios (padl) or profiles.

       ___   ________________  
      | p | |                | 
      | a | |                | 
      | d | |     padup      | 
      | l | |                | 
      |___| |________________| 
             ________________
            |    paddown     |
            |________________|


    Parameters
    ----------
    hcolz   :   TH2F or TTree
        If TTree, <draw_opt> is mandatory
    is_tree : bool
        Whenever the <hcolz> is a ROOT.TTree object        
    xlabel  : str
        The label text for x-axis
    ylabel  : str
        The label text for y-axis
    saveas  : str
        The name and ouput format to save the tripad figure
    draw_opt : string
        String with fields to plot as is given in the ROOT.TH2F.Draw function, for instance
        'posz:posx>>h(100,-200,200,100,30,120)'
    vmin : float
        set to mask some values on the 2D map plot (for instance, those pixels without entries)
    is_profile  : bool
        Set to plot the profiles instead of projections of x and y (padl and paddown, respectively)
    fig_size    : (float,float)
        The size (width and height) of the figure in inches
    img_size    : (float,float)
        Normalized size (width and height) of the 2D map plot (padup)
    hist_fract_size : float
        Fraction of the width or height (for padl or paddown, respectively) for the projections (or
        profiles) plots
    img_left_bottom : (float,float)
        Left and bottom 'empty' side of the subplots of the figure 
    d_btw_subplots : float
        Distance between the sub-plots
    fig_title : str
        Title text for the figure
    fig_title_pos : (float,float)
        Normalized position (x and y) for <fig_title>, default (0.6,0.94)
    fill_hist   : bool
        Set whether to fill the bar of the histogram, unset by default
    set_hist_ylog : (bool,bool)
        Set whether to log the x and y-axes, by default set 
    fig_dpi : int
        DPI to sotre the eps or png figure
    show : bool
        Set whether to show the figure, before save it
    **HOpt : dict
        Any valid option for matplotlib.pylab.bar or matplotlib.pylab.barh
    """

    SetMatplotStyle()
    
    c = ROOT.TCanvas()

    if is_tree:
        ### Draw to get the histogram
        _ = hcolz.Draw(draw_opt,"","COLZ")
        hname=draw_opt.split(">>")[-1].split("(")[0]
        histCOLZ = ROOT.gDirectory.Get(hname)
    
    ### 2D map plot
    Nx = histCOLZ.GetNbinsX()
    Ny = histCOLZ.GetNbinsY()
    img = np.zeros(shape=(Nx,Ny))
    for x in range(1,Nx+1):
        for y in range(1,Ny+1):
            img[x-1][y-1] = histCOLZ.GetBinContent(y,x)

    #### X and Y distribution (projections or profiles)
    if not is_profile:
        hpx=histCOLZ.ProjectionX
        hpy=histCOLZ.ProjectionY
    else:
        hpx=histCOLZ.ProfileX
        hpy=histCOLZ.ProfileY
    
    ### Get bin center and frequencies for the histogram on the X-axis
    x_freq=np.zeros(shape=Nx)
    x_bin=np.zeros(shape=Nx)
    x_bin_w=np.zeros(shape=Nx)
    xprof=hpx('{}_pfx_{}'.format(histCOLZ.GetName(),hash(histCOLZ)))
    for i in range(1,Nx+1):
        x_freq[i-1]=xprof.GetBinContent(i)
        #x_bin[i-1]=xprof.GetXaxis().GetBinCenter(i)
        x_bin[i-1]=xprof.GetXaxis().GetBinLowEdge(i)
        x_bin_w[i-1]=xprof.GetXaxis().GetBinWidth(i)
    
    ### Get bin center and frequencie for the histogram on the y-axis
    y_freq=np.zeros(shape=Ny)
    y_bin=np.zeros(shape=Ny)
    y_bin_w=np.zeros(shape=Ny)
    yprof=hpy('{}_pfy_{}'.format(histCOLZ.GetName(),hash(histCOLZ)))
    for i in range(1,Ny+1):
        y_freq[i-1]=yprof.GetBinContent(i)
        #y_bin[i-1]=yprof.GetXaxis().GetBinCenter(i)
        y_bin[i-1]=yprof.GetXaxis().GetBinLowEdge(i)
        y_bin_w[i-1]=yprof.GetXaxis().GetBinWidth(i)

    ### Get the limits for the X and Y histogram plots
    x_min, x_max = min(x_bin), max(x_bin)
    x_freq_min, x_freq_max = min(x_freq), max(x_freq)
    y_max, y_min = max(y_bin), min(y_bin)
    y_freq_max, y_freq_min = max(y_freq), min(y_freq)

    #### Axes definitions
    nullfmt = plt.NullFormatter()
    
    #### define a box [left_space,bottom_space, width, height] for the three subplots, 
    ####    as well as, the colorbar
    img_left   = img_left_bottom[0]+hist_fract_size*img_size[0]+d_btw_subplots
    img_bottom = img_left_bottom[1]+hist_fract_size*img_size[1]+d_btw_subplots
    bar_size = 0.02

    rect_img   = [img_left,img_bottom,img_size[0],img_size[1]]
    rect_histx = [img_left+0.01/2.,img_left_bottom[1],img_size[0]-bar_size/2.0,hist_fract_size*img_size[1]]
    rect_histy = [img_left_bottom[0],img_bottom,hist_fract_size*img_size[1],img_size[0]]
    rect_colorbar = [img_left+img_size[0]+0.01,img_bottom,bar_size,img_size[1]]

    #### DRAWING
    fig = plt.figure(figsize=fig_size)
    fig.clear()
    fig.clf()

    # if any, plot title
    if len(fig_title)>0:
        fig.suptitle(fig_title,x=fig_title_pos[0],y=fig_title_pos[1], fontsize=15)
    # sub-plot for the 2D map plot (remove ticks of x and y)
    # axImg   = fig.add_subplot(222,position=rect_img)
    axImg = plt.subplot(2,2,2,position=rect_img)
    axImg.tick_params(axis='both',direction='in', labelleft=False, labelbottom=False)
    # sub-plot for the X projection or profile
    axHistX = plt.subplot(2,2,4,position=rect_histx)
    axHistX.tick_params(axis='both',direction='in')
    # sub-plot for the Y projection or profile
    axHistY = plt.subplot(2,2,1,position=rect_histy)
    axHistY.tick_params(axis='both', direction='in')

    # 2D map plot
    #if not vmin is None:
    #    img_masked = np.ma.masked_where(img <= vmin, img)
    #    img=img_masked
    mimg = np.ma.masked_array( img, mask=img<1e-1 )
    im=axImg.imshow(mimg,origin='lower',aspect='equal', cmap=GetROOTPalette(dtype='palette'))#plt.get_cmap(cmap)) #,norm=mpl_colors.LogNorm())
    ### Plot histogram as bar or points, for projections or profile, respectively
    if not is_profile:
        # Draw histogram x
        #   assuming equally binned
        h1=axHistX.bar(x_bin,x_freq,width=x_bin_w,fill=fill_hist,align='edge',**HOpt)
        # Draw histogram y
        #   assuming equally binned
        h2=axHistY.barh(y_bin,y_freq,height=y_bin_w,fill=fill_hist,align='edge',**HOpt)
    else:
        ### plot as white the bars, and plot the error for each bin
        h1=axHistX.bar(x_bin,x_freq,color='white',edgecolor='white',yerr=np.sqrt(x_freq),
                xerr=(x_bin[1]-x_bin[0])/2.,fill=False,**HOpt)
        h2=axHistY.barh(y_bin,y_freq,color='white',edgecolor='white',xerr=np.sqrt(y_freq),
                yerr=(y_bin[1]-y_bin[0])/2.,fill=False,**HOpt)

    # text labels for x and y
    axHistX.set_xlabel(xlabel)
    axHistY.set_ylabel(ylabel)
    # fix the size of the histograms to be the same as the 2D map plot
    axHistX.set_xlim(x_min,x_max+x_bin_w[-1])
    #if not is_profile:
    #    axHistX.set_ylim(0,x_freq_max*1.2)
    axHistY.set_ylim(y_min,y_max+y_bin_w[-1])
    #if not is_profile:
    #    axHistY.set_xlim(0,y_freq_max*1.2)

    # log (if needed)
    if set_hist_ylog[0]:
        axHistX.set_yscale('log')
    if set_hist_ylog[1]:
        axHistY.set_xscale('log')

    # plot the colorbar 
    fig.colorbar(im,cax=fig.add_axes(rect_colorbar))

    if show:
        plt.show()

    plt.savefig(saveas, dpi=fig_dpi)
    

    return
    
