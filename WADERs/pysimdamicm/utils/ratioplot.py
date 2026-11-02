import ROOT


def get_ratio_plot_frames(c):
    """Prepare the canvas with two TPads suitable to create inside
    a usual ratio plot. In order to use it, don't forget to use
    TPad.cd to the targeted pad before draw

       ________________
      |                |
      |                |
      |     padup      |
      |                |
      |________________|
       ________________
      |    paddown     |
      |________________|

    Parameters
    ----------
    c: ROOT.TCanvas()
        the canvas where the pads are included
    Return:
    padup: ROOT.TPad
        the upper pad
    paddown: ROOT.TPad
        the downer pad
    """

    # The pad to place the main plot
    padup = ROOT.TPad("padup_{0}".format(hash(c)),"padup",0,0.30,1,1)
    padup.SetBottomMargin(0.01)
    padup.Draw()
    padup.cd()
    # the pad to place the ratio plot
    c.cd()
    paddown = ROOT.TPad("paddown_{0}".format(hash(c)),"paddown",0,0.03,1,0.28)
    paddown.SetTopMargin(0.01)
    paddown.SetBottomMargin(0.43) # 0.3 -- otherwise the PDF plots cut the Xtitle
    paddown.Draw()
    paddown.cd()
    c.cd()
    
    return padup,paddown

def create_ratio_plot(href,h,is_fitfunc=True,add_uncertenties=False,ylog=True,display=True,
        ylabel='Events',xlabel='Pixel Charge [e^{-}]'):
    """
    
    href : model
    h    : data

    """

    c = ROOT.TCanvas('c','',1300,900)
    pu,pd= get_ratio_plot_frames(c)
    
    if is_fitfunc:
        fitfunc = href
        href = create_fitfunc_histogram(href,h)

    # -- Regular plots with the histos in the upper pad
    # ---- need to get the maximum y first
    y1 = max(href.GetMaximum(), h.GetMaximum())*1.5
    pu.cd()
    frame = pu.DrawFrame(href.GetBinLowEdge(1),0.0,href.GetXaxis().GetBinUpEdge(href.GetNbinsX()),y1)
    # -- setting the titles, extracted from the histos
    frame.GetYaxis().SetTitle(href.GetYaxis().GetTitle())
    # -- not using the x-title, as the error ratio will be used only
    frame.GetXaxis().SetLabelOffset(999)
    # and draw the histos
    h.GetYaxis().SetTitle(ylabel)
    h.Draw()
    h.Draw(" PE SAME")
    # model from fitfunc
    href.Draw("P SAME")
    #if is_fitfunc:
    #    fitfunc.Draw("L SAME")
    
    # -- Done, regular plots
    if ylog:
        pu.SetLogy()

    c.cd()

    ratio = create_residuals(href,h,xlabel)

    # Draw a line in 1 to visualize the ideal case (href-h)/h=0
    line = ROOT.TLine(ratio.GetXaxis().GetXmin(),0.0,ratio.GetXaxis().GetXmax(),0.0)
    line.SetLineColor(ROOT.kGray)
    line.SetLineStyle(8)
    line.SetLineWidth(2)
    
    if add_uncertenties:
        errors = create_errors_band(hr)

    pd.cd()
    ratio.Draw("PESAME")
    line.Draw("SAME")
    if add_uncertenties:
        __container = (pu,pd,frame,errors,ratio,line)
    else:
        __container = (pu,pd,frame,ratio,line)

    c.cd()
    if display:
        c.Update()
        c.Draw()
        input("press enter ... ")

    return c,__container



def create_fitfunc_histogram(fitfunc, h):

    #hfunc = h.Clone("hfitfunc")
    #hfunc.SetName("hfitfunc")
    #hfunc.Reset()
    hfunc = ROOT.TH1F("hfitfunc","hfitfunc",h.GetNbinsX(),h.GetBinLowEdge(1),h.GetBinLowEdge(h.GetNbinsX()+1))
    hfunc.SetLineColor(ROOT.kRed)
    hfunc.SetMarkerColor(ROOT.kRed)
    hfunc.SetMarkerStyle(21)
    hfunc.SetMarkerSize(1.0)

    for i in range(1,h.GetNbinsX()+1):
        xi = h.GetBinCenter(i)
        yi = fitfunc.Eval(xi)

        hfunc.SetBinContent(i,yi) 
    
    return hfunc


def create_errors_band(hr):
    """
    hr : create_residuals
    """

    errors = ROOT.TH1F()
    hr.Copy(errors)

    for i in range(1,hr.GetNbinsX()+1):
        errors.SetBinContent(i,0)
        try:
            print(i,hr.GetBinError(i)/hr.GetBinContent(i))
            errors.SetBinError(i,1-hr.GetBinError(i)/hr.GetBinContent(i))
        except ZeroDivisionError:
            pass

    # set attributes
    errors.SetMaximum(1.4)
    errors.SetMinimum(0.6)
    errors.SetMarkerColor(1)
    errors.SetMarkerStyle(20)
    errors.SetMarkerSize(0)
    errors.SetFillColor(2)
    errors.SetLineColor(2)
    #errors.SetFillStyle(3345)
    # titles margins, sizes,...
    errors.GetXaxis().SetTitleOffset(1.0)
    errors.GetXaxis().SetTitleSize(0.18)
    errors.GetXaxis().SetLabelSize(0.15)
    errors.GetXaxis().SetTitle(hr.GetXaxis().GetTitle())
    errors.GetYaxis().SetNdivisions(205)
#    errors.GetYaxis().SetTitle(errors_ytitle)
    errors.GetYaxis().SetTitleSize(0.14)
    errors.GetYaxis().SetTitleOffset(0.4)
    errors.GetYaxis().SetLabelSize(0.14)
    
    return errors

def create_residuals(hM,hD,xlabel=""):
    """
    hM : model
    hD : data

    residual = (hD - hM)/hD
    """
    #h3 = hD.Clone("h3")
    h3 = ROOT.TH1F()
    hD.Copy(h3)

    h3.SetLineColor(ROOT.kBlack)
    h3.SetMarkerStyle(21)
    h3.SetTitle("")
    #h3.SetMinimum(0.8)
    #h3.SetMaximum(1.35)
    h3.Sumw2()
    h3.SetStats(0)
    h3.Add(hM,-1)
    h3.Divide(hD)
    h3.Scale(100.0)

    # Adjust y-axis settings
    y = h3.GetYaxis()
    y.SetTitle("(Data-Model)/Data [x100]")
    y.SetNdivisions(505)
    y.SetTitleSize(16)
    y.SetTitleFont(43)
    y.SetTitleOffset(1.55)
    y.SetLabelFont(43)
    y.SetLabelSize(17)
  
    # Adjust x-axis settings
    x = h3.GetXaxis()
    x.SetTitle(xlabel)
    x.SetTitleSize(20)
    x.SetTitleFont(43)
    x.SetTitleOffset(4.0)
    x.SetLabelFont(43)
    x.SetLabelSize(20)
    
    return h3



def create_ratio(h1, h2):
    
    #h3 = h1.Clone("h3")
    h3 = ROOT.TH1F()
    h1.Copy()

    h3.SetLineColor(ROOT.kBlack)
    h3.SetMarkerStyle(21)
    h3.SetTitle("")
    h3.SetMinimum(0.8)
    h3.SetMaximum(1.35)
    # Set up plot for markers and errors
    h3.Sumw2()
    h3.SetStats(0)
    h3.Divide(h2)
  
    # Adjust y-axis settings
    y = h3.GetYaxis()
    y.SetTitle("ratio h1/h2 ")
    y.SetNdivisions(505)
    y.SetTitleSize(20)
    y.SetTitleFont(43)
    y.SetTitleOffset(1.55)
    y.SetLabelFont(43)
    y.SetLabelSize(15)
  
    # Adjust x-axis settings
    x = h3.GetXaxis()
    x.SetTitleSize(20)
    x.SetTitleFont(43)
    x.SetTitleOffset(4.0)
    x.SetLabelFont(43)
    x.SetLabelSize(15)

    return h3

