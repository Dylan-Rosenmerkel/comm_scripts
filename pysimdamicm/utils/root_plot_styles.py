#!/usr/bin/env python
""":mod:`plots` -- pre-defined plots for debug mode, ROOT styles
=================================================================

    
    .. moduleauthor:: Nuria Castello-Mor <castello@ifca.unican.es> 

"""

import ROOT

####################################################################################################
#
#   ROOT PLOT STYLEs
#
####################################################################################################
""":mod:`plotstyles` -- ROOT plot styles
=====================================

.. module:: plotstyles
   :platform: Unix
      :synopsis: define ROOT plot styles .
      .. moduleauthor:: Jordi Duarte-Campderros <jorge.duarte.campderros@cern.ch>
"""

def damicmStyle(canvas_w=800, rateHW=0.75, fontnumber=13, fontpres=2):
    
    import ROOT

    mStyle = get_sifca_style()


    # --------------------------------
    #   LEGEND
    # --------------------------------
    mStyle.SetLegendBorderSize(0)
    mStyle.SetLegendFont(42)
    mStyle.SetLegendFillColor(0)

    # --------------------------------
    #   CANVAS
    # --------------------------------
    mStyle.SetCanvasBorderMode(0)
    mStyle.SetCanvasBorderSize(3)
    mStyle.SetCanvasColor(0)
    if canvas_w is not None:
        mStyle.SetCanvasDefW(int(canvas_w))
        if rateHW is not None:
            mStyle.SetCanvasDefH(int(canvas_w*rateHW))

    # --------------------------------
    #   PAD
    # --------------------------------
    mStyle.SetPadBorderMode(0)
    mStyle.SetPadBorderSize(2)
    mStyle.SetPadColor(0)
    mStyle.SetPadTopMargin(0.10)
    mStyle.SetPadRightMargin(0.16)
    mStyle.SetPadBottomMargin(0.16)
    mStyle.SetPadLeftMargin(0.135)
    
    mStyle.SetPadTickX(1)
    mStyle.SetPadTickY(1)

    # --------------------------------
    #   FRAME
    # --------------------------------
    mStyle.SetFrameFillColor(0)
    mStyle.SetFrameBorderMode(0)
    mStyle.SetFrameFillStyle(0)
    mStyle.SetFrameLineColor(1)
    mStyle.SetFrameLineWidth(2)
    mStyle.SetFrameBorderSize(1)

    # --------------------------------
    #   HIST
    # --------------------------------
    mStyle.SetHistFillColor(0)
    mStyle.SetHistFillStyle(1)
    mStyle.SetHistLineColor(ROOT.kBlack)
    mStyle.SetLineWidth(2)
    mStyle.SetLineStyle(0)
    
    # --------------------------------
    #   FUNC
    # --------------------------------
    mStyle.SetFuncWidth(2)

    # --------------------------------
    #   Title
    # --------------------------------
    mStyle.SetTitleFillColor(0)
    mStyle.SetTitleBorderSize(0)

    mStyle.SetTitleFont(42,"pad")
    mStyle.SetTitleFontSize(0.045)
    mStyle.SetTitleSize(0.045)
    mStyle.SetTitleSize(0.045,"x")
    mStyle.SetTitleSize(0.045,"y")
    mStyle.SetTitleSize(0.045,"z")

    mStyle.SetTitleOffset(1.2,"x")
    mStyle.SetTitleOffset(1.5,"y")
    mStyle.SetTitleOffset(1.2,"z")

    mStyle.SetTitleAlign(12)
    mStyle.SetTitleX(0.01)
    mStyle.SetTitleY(0.93)

    #mStyle.SetTitleYOffset(0.6)
    #mStyle.SetTitleXOffset(0.5)

    # --------------------------------
    #   Axis / others
    # --------------------------------
    mStyle.SetStatBorderSize(0)
    
    mStyle.SetTextFont(42)
    mStyle.SetTextSize(0.045)
    mStyle.SetLabelFont(42,"x")
    mStyle.SetLabelFont(42,"y")
    mStyle.SetLabelFont(42,"z")
    mStyle.SetLabelSize(0.045,"x")
    mStyle.SetLabelSize(0.045,"y")
    mStyle.SetLabelSize(0.045,"z")
    mStyle.SetLabelFont(42)
    mStyle.SetStatFont(42)
    mStyle.SetStatFontSize(0.04)
    mStyle.SetHatchesLineWidth(2)
    mStyle.SetLineColor(1)

    #----------------------------------------------------------------------------
    # OptFits
    #----------------------------------------------------------------------------
    mStyle.SetLegendFillColor(0)
    mStyle.SetLegendTextSize(0.025)
    mStyle.SetStatBorderSize(0)
    mStyle.SetStatColor(0)
    mStyle.SetStatFont(42)
    mStyle.SetStatFontSize(0.025)
    mStyle.SetStatX(0.79)
    mStyle.SetStatY(0.89)


    #mStyle.SetNdivisions(10, "X")
    #mStyle.SetNdivisions(10, "Y")

    mStyle.SetOptStat(0)

    mStyle.SetPalette(57) 
    ROOT.gROOT.ForceStyle()

    return mStyle



#def get_sifca_style(squared=False,stat_off=True,fit_stat=False):
#    """Return a ROOT.gStyle to be used for the SIFCA group
#    """
#    sifca = damicmStyle()
#    return sifca
def squaredStyle(): 
    """.. function:: squaredStyle() -> ROOT.gStyle
    
    Return a ROOT.gStyle based in the Latinos' CMS group
    """
    import ROOT
    
    ROOT.GloStyle = ROOT.gStyle
    
    squaredStyle = ROOT.TStyle("squaredStyle", "squaredStyle")
    ROOT.gStyle = squaredStyle
    
    #----------------------------------------------------------------------------
    # Font Legend?
    #----------------------------------------------------------------------------
    squaredStyle.SetTextFont(132)
    squaredStyle.SetTextSize(0.025)
    
    #----------------------------------------------------------------------------
    # Canvas
    #----------------------------------------------------------------------------
    squaredStyle.SetCanvasBorderMode(  0)
    squaredStyle.SetCanvasBorderSize( 10)
    squaredStyle.SetCanvasColor     (  0)
    squaredStyle.SetCanvasDefH      (600)
    squaredStyle.SetCanvasDefW      (550)
    squaredStyle.SetCanvasDefX      ( 10)
    squaredStyle.SetCanvasDefY      ( 10)

    #----------------------------------------------------------------------------
    # Pad
    #----------------------------------------------------------------------------
    squaredStyle.SetPadBorderMode  (   0)
    squaredStyle.SetPadBorderSize  (  10)
    squaredStyle.SetPadColor       (   0)
    squaredStyle.SetPadBottomMargin(0.16)
    squaredStyle.SetPadTopMargin   (0.08)
    squaredStyle.SetPadLeftMargin  (0.16)
    squaredStyle.SetPadRightMargin (0.16)

    #----------------------------------------------------------------------------
    # Frame
    #----------------------------------------------------------------------------
    squaredStyle.SetFrameFillStyle ( 0)
    squaredStyle.SetFrameFillColor ( 0)
    squaredStyle.SetFrameLineColor ( 1)
    squaredStyle.SetFrameLineStyle ( 0)
    squaredStyle.SetFrameLineWidth ( 2)
    squaredStyle.SetFrameBorderMode( 0)
    squaredStyle.SetFrameBorderSize(10)

    #----------------------------------------------------------------------------
    # Hist
    #----------------------------------------------------------------------------
    squaredStyle.SetHistFillColor(0)
    squaredStyle.SetHistFillStyle(1)
    squaredStyle.SetHistLineColor(1)
    squaredStyle.SetHistLineStyle(0)
    squaredStyle.SetHistLineWidth(1)

    #----------------------------------------------------------------------------
    # Axis
    #----------------------------------------------------------------------------
    squaredStyle.SetLabelFont  (   132, "xyz")
    #squaredStyle.SetLabelOffset(0.015, "xyz")
    squaredStyle.SetLabelSize  (0.050, "xyz")
    squaredStyle.SetNdivisions (  505, "xyz")
    squaredStyle.SetTitleFont  (   132, "xyz")
    squaredStyle.SetTitleSize  (0.050, "xyz")

    #  squaredStyle.SetNdivisions ( -503, "y")
    squaredStyle.SetTitleOffset(  1.4,   "x")
    squaredStyle.SetTitleOffset(  1.4,   "y")
    squaredStyle.SetTitleOffset(  1.4,   "z")
    squaredStyle.SetPadTickX   (           1)  # Tick marks on the opposite side of the frame
    squaredStyle.SetPadTickY   (           1)  # Tick marks on the opposite side of the frame

    #----------------------------------------------------------------------------
    # Title
    #----------------------------------------------------------------------------
    squaredStyle.SetTitleBorderSize(    0)
    squaredStyle.SetTitleFillColor (   10)
    squaredStyle.SetTitleAlign     (   23)
    squaredStyle.SetTitleFontSize  (0.045)
    squaredStyle.SetTitleX         (0.560)
    squaredStyle.SetTitleOffset    (  1.0)
    squaredStyle.SetTitleY         (0.99)
    squaredStyle.SetTitleFont(132, "")

    squaredStyle.SetPalette(53)
    #----------------------------------------------------------------------------
    # Stat
    #----------------------------------------------------------------------------
    #squaredStyle.SetOptStat       (1110)
    #squaredStyle.SetStatBorderSize(   0)
    #squaredStyle.SetStatColor     (  10)
    #squaredStyle.SetStatFont      (  132)
    #squaredStyle.SetStatX         (0.94)
    #squaredStyle.SetStatY         (0.91)
    return squaredStyle


def atlasStyle():
    """.. function:: AtlasStyle() -> ROOT.gStyle
    
    Return a ROOT.gStyle based in ATLAS Style
    which is based on style from BaBar
    It is actually a wrapper to the AtlasStyle 
    macro (should be located in the system FIXE HOW)
    """
    import ROOT
    import AtlasStyle

    AtlasStyle.ROOT.SetAtlasStyle()

    atlasStyle= ROOT.gStyle

    atlasStyle.SetCanvasDefH      (600)
    atlasStyle.SetCanvasDefW      (700)
    #atlasStyle.SetCanvasDefX      ( 10)
    #atlasStyle.SetCanvasDefY      ( 10)

    atlasStyle.SetPalette(53)

    return atlasStyle

def njStyle():
    """.. function:: njStyle() -> ROOT.gStyle
    
    Return a ROOT.gStyle based in Castello-Mor and
    Duarte-Campderros mixed styles.
    """
    import ROOT
    ROOT.GloStyle = ROOT.gStyle
    #jl TStyle
    njStyle = ROOT.TStyle('jlStyle', "N-J Style");

    njStyle.SetTextSize(0.025)
    #set the background color to white
    njStyle.SetFillColor(10);
    njStyle.SetFrameFillColor(10);
    njStyle.SetCanvasColor(10);
    njStyle.SetPadColor(10);
    njStyle.SetTitleFillColor(0);
    njStyle.SetStatColor(10);


    #----------------------------------------------------------------------------
    # Legend
    #----------------------------------------------------------------------------
    njStyle.SetTextFont(132)
    njStyle.SetTextSize(0.045)

    njStyle.SetLegendBorderSize(0)
    njStyle.SetLegendFillColor(0)
    njStyle.SetLegendTextSize(0.025)


    #----------------------------------------------------------------------------
    # Canvas
    #----------------------------------------------------------------------------
    njStyle.SetCanvasBorderMode(  0)
    njStyle.SetCanvasBorderSize( 10)
    njStyle.SetCanvasColor     (  0)
    njStyle.SetCanvasDefH      (450)
    njStyle.SetCanvasDefW      (600)
    njStyle.SetCanvasDefX      ( 10)
    njStyle.SetCanvasDefY      ( 10)

    #----------------------------------------------------------------------------
    # Pad
    #----------------------------------------------------------------------------
    njStyle.SetPadBorderMode  (   0)
    njStyle.SetPadBorderSize  (   8)
    njStyle.SetPadColor       (   0)
    njStyle.SetPadBottomMargin(0.15)
    njStyle.SetPadTopMargin   (0.08)
    njStyle.SetPadLeftMargin  (0.10)
    njStyle.SetPadRightMargin (0.10)


    #use the primary color palette
    njStyle.SetPalette(53);

    #set the default line color for a histogram to be black
    njStyle.SetHistLineColor(ROOT.kBlack);

    #set the default line color for a fit function to be red
    njStyle.SetFuncColor(ROOT.kRed);

    #make the axis labels black
    njStyle.SetLabelColor(ROOT.kBlack,"xyz");

    #set the default title color to be black
    njStyle.SetTitleColor(ROOT.kBlack);

    #set the margins
    #njStyle.SetPadBottomMargin(0.15);
    #njStyle.SetPadTopMargin(0.08);
    #njStyle.SetPadRightMargin(0.10);
    #njStyle.SetPadLeftMargin(0.15);

    #set axis label and title text sizes
    njStyle.SetLabelFont(132,"x");
    njStyle.SetLabelFont(132,"y");
    njStyle.SetLabelFont(132,"z");
    njStyle.SetLabelSize(0.05,"x");
    njStyle.SetLabelSize(0.05,"y");
    njStyle.SetLabelSize(0.05,"z");
    njStyle.SetLabelOffset(0.015,"x");
    njStyle.SetLabelOffset(0.015,"y");
    njStyle.SetLabelOffset(0.015,"z");
    njStyle.SetTitleFont(132,"x");
    njStyle.SetTitleFont(132,"y");
    njStyle.SetTitleFont(132,"z");
    njStyle.SetTitleSize(0.04,"x");
    njStyle.SetTitleSize(0.04,"y");
    njStyle.SetTitleSize(0.04,"z");
    njStyle.SetTitleXOffset(1.6);
    njStyle.SetTitleYOffset(1.2);
    njStyle.SetTitleOffset(1.1,"z");
    njStyle.SetStatFont(132);
    njStyle.SetStatFontSize(0.08);
    njStyle.SetTitleBorderSize(0);
    njStyle.SetStatBorderSize(0);
    njStyle.SetTextFont(132);

    #set line widths
    njStyle.SetFrameLineWidth(2);
    njStyle.SetFuncWidth(2);
    njStyle.SetHistLineWidth(2);
    #set the number of divisions to show
    njStyle.SetNdivisions(506, "xy");

    #turn off xy grids
    njStyle.SetPadGridX(0);
    njStyle.SetPadGridY(0);

    #set the tick mark style
    njStyle.SetPadTickX(1);
    njStyle.SetPadTickY(1);

    #turn off stats
    njStyle.SetOptStat(1);
    njStyle.SetOptFit(1);

    #marker settings
    njStyle.SetMarkerStyle(20);
    njStyle.SetMarkerSize(0.8);
    njStyle.SetLineWidth(3);

    #done
    #njStyle.cd();
    #ROOT.gROOT.ForceStyle();
    #ROOT.gStyle.ls();
    return njStyle

def get_sed_style(canvas_W=1300,rateHW=0.65,
        fontnumber=13,fontpres=2
        ):
    """Return a ROOT.gStyle to be used for the SIFCA group
    """
    import ROOT

    ROOT.GloStyle = ROOT.gStyle

    sedStyle = ROOT.TStyle("sedStyle", "sedStyle")
    ROOT.gStyle = sedStyle

    #----------------------------------------------------------------------------
    # Legend
    #----------------------------------------------------------------------------

    textfont=int(10*fontnumber+fontpres)
    sedStyle.SetTextFont(textfont)
    sedStyle.SetTextSize(0.045)
    
    sedStyle.SetLegendBorderSize(0)
    sedStyle.SetLegendFillColor(0)
    sedStyle.SetLegendTextSize(0.025)

    #----------------------------------------------------------------------------
    # Canvas
    #----------------------------------------------------------------------------
    sedStyle.SetCanvasBorderMode(0)
    sedStyle.SetCanvasBorderSize(2)
    sedStyle.SetCanvasColor(0)
    sedStyle.SetCanvasDefH(int(rateHW*canvas_W))
    sedStyle.SetCanvasDefW(int(canvas_W))
    #sifcaStyle.SetCanvasDefX( 10)
    #sifcaStyle.SetCanvasDefY( 10)
    #
    ##----------------------------------------------------------------------------
    ## Pad
    ##----------------------------------------------------------------------------
    sedStyle.SetPadBorderMode(0)
    sedStyle.SetPadBorderSize(2)
    sedStyle.SetPadColor(0)
    sedStyle.SetPadBottomMargin(0.14)
    sedStyle.SetPadTopMargin(0.08)
    sedStyle.SetPadLeftMargin(0.10)
    sedStyle.SetPadRightMargin(0.22)

    sedStyle.SetPadTickX(1)
    sedStyle.SetPadTickY(1)
    #----------------------------------------------------------------------------
    # Frame
    #----------------------------------------------------------------------------
    sedStyle.SetFrameFillStyle(0)
    sedStyle.SetFrameFillColor(0)
    sedStyle.SetFrameLineColor(1)
    sedStyle.SetFrameLineStyle(0)
    sedStyle.SetFrameLineWidth(1)
    sedStyle.SetFrameBorderMode(0)
    sedStyle.SetFrameBorderSize(1)

    #----------------------------------------------------------------------------
    # Hist
    #----------------------------------------------------------------------------
    sedStyle.SetHistFillColor(0)
    sedStyle.SetHistFillStyle(1)
    sedStyle.SetHistLineColor(1)
    sedStyle.SetHistLineStyle(0)
    sedStyle.SetHistLineWidth(2)

    #----------------------------------------------------------------------------
    # Func
    #----------------------------------------------------------------------------
    sedStyle.SetFuncWidth(2)

    #----------------------------------------------------------------------------
    # Title
    #----------------------------------------------------------------------------
    sedStyle.SetTitleBorderSize( 0)
    sedStyle.SetTitleFillColor( 0)
    sedStyle.SetTitleX(0.03)
    sedStyle.SetTitleFont(textfont)
    sedStyle.SetTitleSize(0.045)
    sedStyle.SetPalette(57)
    #----------------------------------------------------------------------------
    # Stat
    #----------------------------------------------------------------------------
    sedStyle.SetStatBorderSize(0)
    sedStyle.SetStatColor(0)
    sedStyle.SetOptStat(0)
    #----------------------------------------------------------------------------
    # Axis
    #----------------------------------------------------------------------------
    #sifcaStyle.SetPadTickX   (           1)  # Tick marks on the opposite side of the frame
    #sifcaStyle.SetPadTickY   (           1)  # Tick marks on the opposite side of the frame
    sedStyle.SetTitleFont(textfont,"x")
    sedStyle.SetTitleFont(textfont,"y")
    sedStyle.SetTitleFont(textfont,"z")
    sedStyle.SetTitleSize(0.045,"x")
    sedStyle.SetTitleSize(0.045,"y")
    sedStyle.SetTitleSize(0.045,"z")

    sedStyle.SetTitleOffset(1.20,"x")
    sedStyle.SetTitleOffset(1.20,"y")
    sedStyle.SetTitleOffset(1.20,"z")
    sedStyle.SetLabelFont(textfont,"x")
    sedStyle.SetLabelFont(textfont,"y")
    sedStyle.SetLabelFont(textfont,"z")
    sedStyle.SetLabelSize(0.045,"x")
    sedStyle.SetLabelSize(0.045,"y")
    sedStyle.SetLabelSize(0.045,"z")

    # ---------------------------------------
    # Extra
    # ---------------------------------------    
    sedStyle.SetNumberContours(99)

    #marker settings
    sedStyle.SetMarkerStyle(20);
    sedStyle.SetMarkerSize(0.8);
    sedStyle.SetLineWidth(2);

    return sedStyle

def get_moskita_style(squared=False,stat_off=False,fit_stat=True):
    """Return a ROOT.gStyle to be used for the SIFCA group
    """
    import ROOT

    ROOT.GloStyle = ROOT.gStyle

    moskitaStyle = ROOT.TStyle("moskitaStyle", "moskitaStyle")
    ROOT.gStyle = moskitaStyle


    #----------------------------------------------------------------------------
    # Legend
    #----------------------------------------------------------------------------
    moskitaStyle.SetTextFont(132)
    moskitaStyle.SetTextSize(0.045)

    moskitaStyle.SetLegendBorderSize(0)
    moskitaStyle.SetLegendFillColor(0)
    moskitaStyle.SetLegendTextSize(0.025)

    #----------------------------------------------------------------------------
    # Canvas
    #----------------------------------------------------------------------------
    moskitaStyle.SetCanvasBorderMode(0)
    moskitaStyle.SetCanvasBorderSize(10)
    moskitaStyle.SetCanvasColor(0)
    moskitaStyle.SetCanvasDefH(850)
    moskitaStyle.SetCanvasDefW(1130)

    #
    ##----------------------------------------------------------------------------
    ## Pad
    ##----------------------------------------------------------------------------
    moskitaStyle.SetPadBorderMode  (   0)
    moskitaStyle.SetPadBorderSize  (   1)
    moskitaStyle.SetPadColor       (   0)
    moskitaStyle.SetPadBottomMargin(0.14)
    moskitaStyle.SetPadTopMargin   (0.10)
    moskitaStyle.SetPadLeftMargin  (0.14)
    moskitaStyle.SetPadRightMargin (0.11)
    if squared:
        moskitaStyle.SetCanvasDefH(1000)
        moskitaStyle.SetCanvasDefW(1024)
        moskitaStyle.SetPadRightMargin (0.18)

    moskitaStyle.SetPadTickX(1)
    moskitaStyle.SetPadTickY(1)
    #----------------------------------------------------------------------------
    # Frame
    #----------------------------------------------------------------------------
    moskitaStyle.SetFrameFillStyle ( 0)
    moskitaStyle.SetFrameFillColor ( 0)
    moskitaStyle.SetFrameLineColor ( 1)
    moskitaStyle.SetFrameLineStyle ( 0)
    moskitaStyle.SetFrameLineWidth ( 1)
    moskitaStyle.SetFrameBorderMode( 0)

    #----------------------------------------------------------------------------
    # Hist
    #----------------------------------------------------------------------------
    moskitaStyle.SetHistFillColor(0)
    moskitaStyle.SetHistFillStyle(1)
    moskitaStyle.SetHistLineColor(1)
    moskitaStyle.SetHistLineStyle(0)
    moskitaStyle.SetHistLineWidth(2)

    #----------------------------------------------------------------------------
    # Func
    #----------------------------------------------------------------------------
    moskitaStyle.SetFuncWidth(2)

    #----------------------------------------------------------------------------
    # Title
    #----------------------------------------------------------------------------
    moskitaStyle.SetTitleBorderSize(0)
    moskitaStyle.SetTitleFillColor(0)
    if squared:
        moskitaStyle.SetTitleX(0.16)
    else:
        moskitaStyle.SetTitleX(0.13)
    #moskitaStyle.SetTitleAlign(132)
    moskitaStyle.SetTitleFont(132)
    moskitaStyle.SetTitleSize(0.055)
    moskitaStyle.SetPalette(57)

    #----------------------------------------------------------------------------
    # Stat
    #----------------------------------------------------------------------------
    moskitaStyle.SetStatColor(0)
    if stat_off:
        moskitaStyle.SetOptStat(0)
    #----------------------------------------------------------------------------
    # OptFits
    #----------------------------------------------------------------------------
    moskitaStyle.SetLegendFillColor(0)
    moskitaStyle.SetLegendTextSize(0.025)
    if fit_stat:
        moskitaStyle.SetOptFit(1)
        moskitaStyle.SetStatColor(0)
        moskitaStyle.SetStatFont(132)
        moskitaStyle.SetStatFontSize(0.025)

    #----------------------------------------------------------------------------
    # Axis
    #----------------------------------------------------------------------------
    #moskitaStyle.SetPadTickX   (           1)  # Tick marks on the opposite side of the frame
    #moskitaStyle.SetPadTickY   (           1)  # Tick marks on the opposite side of the frame
    moskitaStyle.SetTitleFont(132, "x")
    moskitaStyle.SetTitleFont(132, "y")
    moskitaStyle.SetTitleFont(132, "z")

    if not squared:
        moskitaStyle.SetTitleOffset(0.65,"y")
        moskitaStyle.SetTitleSize(0.085,"x")
        moskitaStyle.SetTitleSize(0.085,"y")
        moskitaStyle.SetTitleSize(0.085,"z")
        moskitaStyle.SetLabelSize(0.075,"x")
        moskitaStyle.SetLabelSize(0.075,"y")
        moskitaStyle.SetLabelSize(0.075,"z")
    else:
        moskitaStyle.SetTitleY(0.99)
        moskitaStyle.SetTitleOffset(1.10,"x")
        moskitaStyle.SetTitleOffset(1.2,"y")
        moskitaStyle.SetTitleOffset(1.2,"z")
        moskitaStyle.SetTitleSize(0.065,"x")
        moskitaStyle.SetTitleSize(0.065,"y")
        moskitaStyle.SetTitleSize(0.065,"z")
        moskitaStyle.SetLabelSize(0.055,"x")
        moskitaStyle.SetLabelSize(0.055,"y")
        moskitaStyle.SetLabelSize(0.055,"z")


    moskitaStyle.SetLabelFont(132, "x")
    moskitaStyle.SetLabelFont(132, "y")
    moskitaStyle.SetLabelFont(132, "z")

    # ---------------------------------------
    # Extra
    # ---------------------------------------    
    moskitaStyle.SetNumberContours(99)


    #marker settings
    moskitaStyle.SetMarkerStyle(21);
    moskitaStyle.SetMarkerSize(0.8);
    moskitaStyle.SetLineWidth(2);


    moskitaStyle.SetOptTitle(0)

    moskitaStyle.cd()
    ROOT.gROOT.ForceStyle()

    return moskitaStyle




def get_sifca_style(squared=False,stat_off=False,fit_stat=True):
    """Return a ROOT.gStyle to be used for the SIFCA group
    """
    import ROOT

    ROOT.GloStyle = ROOT.gStyle

    sifcaStyle = ROOT.TStyle("sifcaStyle", "sifcaStyle")
    ROOT.gStyle = sifcaStyle


    #----------------------------------------------------------------------------
    # Legend
    #----------------------------------------------------------------------------
    sifcaStyle.SetTextFont(132)
    sifcaStyle.SetTextSize(0.065)

    sifcaStyle.SetLegendBorderSize(0)
    sifcaStyle.SetLegendFillColor(0)
    sifcaStyle.SetLegendTextSize(0.025)

    #----------------------------------------------------------------------------
    # Canvas
    #----------------------------------------------------------------------------
    sifcaStyle.SetCanvasBorderMode(0)
    sifcaStyle.SetCanvasBorderSize(10)
    sifcaStyle.SetCanvasColor(0)
    sifcaStyle.SetCanvasDefH(850)
    sifcaStyle.SetCanvasDefW(1130)
    #
    ##----------------------------------------------------------------------------
    ## Pad
    ##----------------------------------------------------------------------------
    sifcaStyle.SetPadBorderMode  (   0)
    sifcaStyle.SetPadBorderSize  (   1)
    sifcaStyle.SetPadColor       (   0)
    sifcaStyle.SetPadBottomMargin(0.14)
    sifcaStyle.SetPadTopMargin   (0.10)
    sifcaStyle.SetPadLeftMargin  (0.14)
    sifcaStyle.SetPadRightMargin (0.05)
    if squared:
        sifcaStyle.SetCanvasDefH(1000)
        sifcaStyle.SetCanvasDefW(1024)
        sifcaStyle.SetPadRightMargin (0.18)

    sifcaStyle.SetPadTickX(1)
    sifcaStyle.SetPadTickY(1)
    #----------------------------------------------------------------------------
    # Frame
    #----------------------------------------------------------------------------
    sifcaStyle.SetFrameFillStyle ( 0)
    sifcaStyle.SetFrameFillColor ( 0)
    sifcaStyle.SetFrameLineColor ( 1)
    sifcaStyle.SetFrameLineStyle ( 0)
    sifcaStyle.SetFrameLineWidth ( 1)
    sifcaStyle.SetFrameBorderMode( 0)
#    sifcaStyle.SetFrameBorderSize( 1)

    #----------------------------------------------------------------------------
    # Hist
    #----------------------------------------------------------------------------
    sifcaStyle.SetHistFillColor(0)
    sifcaStyle.SetHistFillStyle(1)
    sifcaStyle.SetHistLineColor(1)
    sifcaStyle.SetHistLineStyle(0)
    sifcaStyle.SetHistLineWidth(2)

    #----------------------------------------------------------------------------
    # Func
    #----------------------------------------------------------------------------
    sifcaStyle.SetFuncWidth(2)

    #----------------------------------------------------------------------------
    # Title
    #----------------------------------------------------------------------------
    sifcaStyle.SetTitleBorderSize(0)
    sifcaStyle.SetTitleFillColor(0)
    if squared:
        sifcaStyle.SetTitleX(0.16)
    else:
        sifcaStyle.SetTitleX(0.13)
    #sifcaStyle.SetTitleAlign(12)
    sifcaStyle.SetTitleFont(132)
    sifcaStyle.SetTitleSize(0.035)
    sifcaStyle.SetPalette(57)
    #----------------------------------------------------------------------------
    # Stat
    #----------------------------------------------------------------------------
    sifcaStyle.SetStatColor(0)
    if stat_off:
        sifcaStyle.SetOptStat(0)
    #----------------------------------------------------------------------------
    # OptFits
    #----------------------------------------------------------------------------
#    sifcaStyle.SetLegendBorderSize(1)
    sifcaStyle.SetLegendFillColor(0)
    sifcaStyle.SetLegendTextSize(0.025)
    if fit_stat:
        sifcaStyle.SetOptFit(1)
#        sifcaStyle.SetStatBorderSize(1)
        sifcaStyle.SetStatColor(0)
        sifcaStyle.SetStatFont(132)
        sifcaStyle.SetStatFontSize(0.025)

    #----------------------------------------------------------------------------
    # Axis
    #----------------------------------------------------------------------------
    #sifcaStyle.SetPadTickX   (           1)  # Tick marks on the opposite side of the frame
    #sifcaStyle.SetPadTickY   (           1)  # Tick marks on the opposite side of the frame
    sifcaStyle.SetTitleFont(132, "x")
    sifcaStyle.SetTitleFont(132, "y")
    sifcaStyle.SetTitleFont(132, "z")
    sifcaStyle.SetTitleSize(0.045,"x")
    sifcaStyle.SetTitleSize(0.045,"y")
    sifcaStyle.SetTitleSize(0.045,"z")

    sifcaStyle.SetTitleOffset(1.20,"x")
    sifcaStyle.SetTitleOffset(1.20,"y")
    sifcaStyle.SetTitleOffset(1.20,"z")
    if squared:
        sifcaStyle.SetTitleOffset(1.4,"x")
        sifcaStyle.SetTitleOffset(1.6,"y")
        sifcaStyle.SetTitleOffset(1.4,"z")
    sifcaStyle.SetLabelFont(132, "x")
    sifcaStyle.SetLabelFont(132, "y")
    sifcaStyle.SetLabelFont(132, "z")
    sifcaStyle.SetLabelSize(0.045,"x")
    sifcaStyle.SetLabelSize(0.045,"y")
    sifcaStyle.SetLabelSize(0.045,"z")

    # ---------------------------------------
    # Extra
    # ---------------------------------------    
    sifcaStyle.SetNumberContours(99)


    #marker settings
    sifcaStyle.SetMarkerStyle(20);
    sifcaStyle.SetMarkerSize(0.8);
    sifcaStyle.SetLineWidth(1);


    sifcaStyle.SetOptTitle(0)

    sifcaStyle.cd()
    ROOT.gROOT.ForceStyle()

    return sifcaStyle


def setpalette(name="rainbow", ncontours=99):
    """.. function::setpalette()
    
    Set a color palette from a given RGB list
    stops, red, green and blue should all be lists 
    of the same length 
    see set_decent_colors for an example"""
    from ROOT import TColor,gStyle
    from array import array

    if name == "gray" or name == "grayscale":
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [1.00, 0.84, 0.61, 0.34, 0.00]
        green = [1.00, 0.84, 0.61, 0.34, 0.00]
        blue  = [1.00, 0.84, 0.61, 0.34, 0.00]
    elif name == 'darkbody':
        stops = [0.00, 0.25, 0.50, 0.75, 1.00]
        red   = [0.00, 0.50, 1.00, 1.00, 1.00]
        green = [0.00, 0.00, 0.55, 1.00, 1.00]
        blue  = [0.00, 0.00, 0.00, 0.00, 1.00]
    elif name == 'inv_darkbody':
        stops = [0.00, 0.25, 0.50, 0.75, 1.00]
        red   = [1.00, 1.00, 1.00, 0.50, 0.00]
        green = [1.00, 1.00, 0.55, 0.00, 0.00]
        blue  = [1.00, 0.00, 0.00, 0.00, 0.00]
    elif name == 'deepsea':
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [0.00, 0.09, 0.18, 0.09, 0.00]
        green = [0.01, 0.02, 0.39, 0.68, 0.97]
        blue  = [0.17, 0.39, 0.62, 0.79, 0.97]
    elif name == 'forest':
        stops = [0.00, 0.25, 0.50, 0.75, 1.00]
        red   = [0.93, 0.70, 0.40, 0.17, 0.00]
        green = [0.97, 0.89, 0.76, 0.64, 0.43]
        blue  = [0.98, 0.89, 0.64, 0.37, 0.17]
    else:
        # default palette, looks cool
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [0.00, 0.00, 0.87, 1.00, 0.51]
        green = [0.00, 0.81, 1.00, 0.20, 0.00]
        blue  = [0.51, 1.00, 0.12, 0.00, 0.00]


    s = array('d', stops)
    r = array('d', red)
    g = array('d', green)
    b = array('d', blue)
    
    npoints = len(s)
    TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    gStyle.SetNumberContours(ncontours)


