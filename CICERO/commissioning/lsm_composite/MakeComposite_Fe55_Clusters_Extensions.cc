// copied from LSM_Module_Testing_Analysis/src/MakeComposite_Fe55_Clusters_Extensions.cc (commit af3787b + uncommitted changes) by utils/sync_from_lsm_analysis.py: edit it there, not here
// Diego Venegas-Vargas
// DAMIC-M Collaboration
// Per-extension Fe-55 (and Am241) cluster analysis for LSM module testing.
//
// For each CCD extension (1-4): builds composite "front-side"/"back-side" cluster
// images (split by sigma_xy, the diffusion-broadened cluster width, which encodes
// interaction depth), fits the Fe-55 Kalpha/Kbeta energy peaks, computes CTI
// (charge-transfer inefficiency) fractions relative to the cluster center, and
// produces per-extension and cross-extension comparison plots + summary logs.
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits>

#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"
#include "TBox.h"
#include "TLatex.h"
#include "TLine.h"
#include "TF1.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TString.h"
#include "TMath.h"
#include "TPad.h"
#include "TImage.h"
#include "TGraphErrors.h"
#include "TAxis.h"


// --- For cling vector dictionaries ---
#ifdef __CLING__
#pragma link C++ class std::vector<int>+;
#pragma link C++ class std::vector<float>+;
#pragma link C++ class std::vector<std::vector<int> >+;
#pragma link C++ class std::vector<std::vector<float> >+;
#endif

// -------------------------- Config ---------------------------------
int    NBINS_COMP_X   = 401;     // composite cluster grid bins per axis
int    NBINS_COMP_Y   = 401;     // composite cluster grid bins per axis
double RANGE_COMP   = 4.0;     // +/- range for composite cluster histograms(pixels)
int    BINS_PER_PIX = 25;      // sampling density for pixel maps
double E_PER_e    = 3.74;     // eV->e- conversion from eV to electrons

double ENE_MIN_FE55      = 4.5;     // energy window min for Fe events in [keV]
double ENE_MAX_FE55      = 7.0;   // energy window max for Fe events in [keV]

double ENE_MIN_AM241 = 10;     // energy window min for Am241 events in [keV]
double ENE_MAX_AM241 = 30;   // energy window max for Am241 events in [keV]
double ENE_AM241_RESOLUTION = 0.5;   // energy resolution for Am241 events in [keV]
int    NBINS_ENERGY_AM241 = (ENE_MAX_AM241-ENE_MIN_AM241)/ENE_AM241_RESOLUTION;     // number of bins for energy histogram of Am241 events


double SIGMA_FRONT_CUT = 1;  // front/back cut in sigma_xy [pixels]
double    SINGLE_e_RES_e = 0.2; // single-e resolution [e-]
double   eNUM_MIN       = 0.5;       // min number of electrons per pixel
double   eNUM_MAX       = 1900.5;      // max number of electrons per pixel
int   E_PIXEL_BIN_NUM = (eNUM_MAX-eNUM_MIN)/SINGLE_e_RES_e; // binning for pixel energy histograms
int min_e_number_display =500; // minimum number of electrons to display in pixel energy histograms
int max_e_number_display =515; // maximum number of electrons to display in pixel energy histograms

// -------------------- Histogram binning (per-extension plots) --------------------

// Fe-55-windowed energy histograms: front/back energy spectra, max-pixel-energy
// spectrum, and the energy axis of the energy-vs-sigma_xy 2D plot.
int NBINS_ENERGY_FE55 = 50;              // bins over [ENE_MIN_FE55, ENE_MAX_FE55]

// Wide-range energy axis (0 to ENE_MAX_FE55) used by the energy/max-pixel-energy vs
// cluster-size 2D plots, which include all clusters rather than just the Fe-55 window.
int    NBINS_ENERGY_WIDE = 140;          // bins over [ENERGY_WIDE_MIN, ENE_MAX_FE55]
double ENERGY_WIDE_MIN   = 0.0;          // [keV]

// Cluster-size histogram binning (cluster size distribution, and the cluster-size axis
// of the energy/max-pixel-energy vs cluster-size 2D plots). Half-integer edges put bin
// centers on integer pixel counts.
int    NBINS_CLUSTER_SIZE = 100;         // bins over [CLUSTER_SIZE_MIN, CLUSTER_SIZE_MAX]
double CLUSTER_SIZE_MIN   = 0.5;         // [pixels]
double CLUSTER_SIZE_MAX   = 100.5;       // [pixels] (100 pixels max cluster size + 0.5)

// sigma_x / sigma_y / sigma_xy distribution binning (hSigmax, hSigmay, hSigmaxy)
int    NBINS_SIGMA_DIST = 200;           // bins over [SIGMA_DIST_MIN, SIGMA_DIST_MAX]
double SIGMA_DIST_MIN   = 0.0;           // [pixels]
double SIGMA_DIST_MAX   = 8.0;           // [pixels]

// sigma_xy axis shared by the "energy vs sigma_xy" and "posx vs sigma_xy" 2D plots
int    NBINS_SIGMA_XY_AXIS = 50;         // bins over [SIGMA_XY_AXIS_MIN, SIGMA_XY_AXIS_MAX]
double SIGMA_XY_AXIS_MIN   = 0.0;        // [pixels]
double SIGMA_XY_AXIS_MAX   = 2.0;        // [pixels]

// hPosX/hPosY placeholder binning: constructed with a dummy axis range before the real
// per-image range is applied via SetLimits() once nColumns/nRows are known below; only
// the bin count actually survives that call, but both edges are named for clarity.
int    NBINS_POS_PLACEHOLDER = 20;
double POS_PLACEHOLDER_MIN   = 0.0;      // dummy, overwritten by SetLimits()
double POS_PLACEHOLDER_MAX   = 1.0;      // dummy, overwritten by SetLimits()

// hPosXY coarsening: one bin per this many columns/rows of the sensor image, with a
// floor of MIN_POS_XY_BINS so a small image doesn't request zero bins.
int POS_XY_COARSEN_FACTOR = 100;
int MIN_POS_XY_BINS       = 1;

// Column/row index of the sensor image's lower-left corner (images are 0-indexed),
// used as the real axis minimum for hPosX/hPosY/hPosXY/hposx_sigmaxy.
double IMAGE_ORIGIN = 0.0;

// Fe-55 Kalpha/Kbeta double-Gaussian fit: window and seed for the Kalpha fit.
// The Kbeta window is then placed relative to the fitted Kalpha peak (see
// ProcessOneExtension, front- and back-side fits, for the two slightly
// different placement/validation strategies).
const double KALPHA_FIT_LO     = 5.3;   // Kalpha fit window low edge [keV]
const double KALPHA_FIT_HI     = 6.1;   // Kalpha fit window high edge [keV]
const double KALPHA_SEED_MU    = 5.9;   // Fe-55 Kalpha line [keV]
const double KALPHA_SEED_SIGMA = 0.10;  // [keV]
const double FIT_SIGMA_MIN     = 0.03;  // Gaussian sigma parameter limit [keV]
const double FIT_SIGMA_MAX     = 0.40;  // Gaussian sigma parameter limit [keV]

// --------------------- Small helpers --------------------------------

static inline void EnsureDir(const TString& path){
  gSystem->mkdir(path, kTRUE);
}

static inline TString BaseLeaf(const char* path){
  TString s(path);
  int slash = s.Last('/');
  if (slash >= 0) s = s(slash+1, s.Length()-slash-1);
  int dot = s.Last('.');
  if (dot >= 0) s = s(0, dot);
  return s;
}

static inline int ParseExtNum(const TString& p){
  // find "_ext<digit>"
  int idx = p.Index("_ext");
  if (idx < 0 || idx+4 >= p.Length()) return -1;
  char c = p[idx+4];
  if (c<'1' || c>'4') return -1;
  return c - '0';
}

// Save canvas as PNG
static inline void SaveCanvas(TCanvas* c, const TString& basePath){
  if (!c) return;
  c->Modified();
  c->Update();
  c->SaveAs(basePath + ".png");
}

// Deletes the canvas unless `interactive` is set, in which case it is left open for
// the caller (used for the Am241 canvas when MakeComposite_Fe55_Clusters_Extensions's
// interactiveAm241 argument is true, so it can be inspected/zoomed in the ROOT GUI).
static inline void ReleaseCanvasForInteractive(TCanvas* c, bool interactive){
  if (c && !interactive) delete c;
}

// Build a TH2F from pixel lists, with half-integer edges (bin centers at ints)
static TH2F* VectorsToTH2F(const std::vector<int>& x,
                           const std::vector<int>& y,
                           const std::vector<float>& val,
                           const char* name="h2")
{
  if (x.size()!=y.size() || y.size()!=val.size() || x.empty()){
    // Return a tiny empty hist to avoid null checks
    return new TH2F(name, name, 1, -0.5, 0.5, 1, -0.5, 0.5);
  }
  int min_x = *std::min_element(x.begin(), x.end());
  int max_x = *std::max_element(x.begin(), x.end());
  int min_y = *std::min_element(y.begin(), y.end());
  int max_y = *std::max_element(y.begin(), y.end());

  TH2F* h2 = new TH2F(name, name,
                      max_x - min_x + 1, min_x - 0.5, max_x + 0.5,
                      max_y - min_y + 1, min_y - 0.5, max_y + 0.5);
  for (size_t i=0;i<x.size();++i) h2->Fill(x[i], y[i], val[i]);
  return h2;
}

// Rectangle integral without changing axis ranges (robust)
static double SumRect(TH2D* h, double xlo, double xhi, double ylo, double yhi){
  auto ax = h->GetXaxis();
  auto ay = h->GetYaxis();
  const double eps = 1e-9;
  int x1 = ax->FindBin(xlo + eps), x2 = ax->FindBin(xhi - eps);
  int y1 = ay->FindBin(ylo + eps), y2 = ay->FindBin(yhi - eps);
  return h->Integral(x1, x2, y1, y2);
}

// ProjectX over a Y slab [ylo,yhi]
static TH1D* ProjXRect(TH2D* h, const char* name, double ylo, double yhi){
  auto ay = h->GetYaxis();
  int y1 = ay->FindBin(ylo + 1e-9);
  int y2 = ay->FindBin(yhi - 1e-9);
  return h->ProjectionX(name, y1, y2);
}

// ProjectY over an X slab [xlo,xhi]
static TH1D* ProjYRect(TH2D* h, const char* name, double xlo, double xhi){
  auto ax = h->GetXaxis();
  int x1 = ax->FindBin(xlo + 1e-9);
  int x2 = ax->FindBin(xhi - 1e-9);
  return h->ProjectionY(name, x1, x2);
}

// Accumulates one cluster's pixel-energy map (hview, an oversampled per-pixel image
// from VectorsToTH2F) into a composite histogram, re-centered on the cluster's fitted
// position (x,y). Dividing by BINS_PER_PIX^2 converts the oversampled bin content back
// to per-pixel units before adding into the composite.
static void AccumulateComposite(TH2D* composite, TH2F* hview, double x, double y){
  for (int ix=1; ix<=composite->GetNbinsX(); ++ix){
    for (int iy=1; iy<=composite->GetNbinsY(); ++iy){
      const double gx = x + composite->GetXaxis()->GetBinLowEdge(ix);
      const double gy = y + composite->GetYaxis()->GetBinLowEdge(iy);
      const double bc = hview->GetBinContent(hview->FindBin(gx, gy)) / (BINS_PER_PIX*BINS_PER_PIX);
      composite->SetBinContent(ix, iy, composite->GetBinContent(ix,iy) + bc);
    }
  }
}

// Draws the front-composite heatmap (log-z, +/-4 pixel range) with the five CTI-region
// outline boxes (center/above/below/right/left) overlaid, plus a title, on the current
// pad. If zmax>0 the z-axis is fixed to [1e-6,zmax] (used to give every extension in a
// comparison grid the same color scale); otherwise the histogram's own auto-range is used.
// Returns the cloned+styled histogram (caller owns it, as with the original inline code).
static TH2D* DrawFrontCompositeHeatmap(TH2D* hFront, double rightMargin, double boxLineWidth,
                                        double zmax, const TString& title){
  gPad->SetRightMargin(rightMargin); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12); gPad->SetLogz();
  TH2D* h = (TH2D*)hFront->Clone();
  h->GetXaxis()->SetRangeUser(-4,4); h->GetYaxis()->SetRangeUser(-4,4);
  if (zmax > 0){ h->SetMinimum(1e-6); h->SetMaximum(zmax); }
  h->SetTitle("");
  h->GetXaxis()->SetTitle("#Delta x [pix]");
  h->GetYaxis()->SetTitle("#Delta y [pix]");
  h->Draw("COLZ");
  auto drawBox=[&](double x1,double y1,double x2,double y2, Color_t lc){
    TBox* b=new TBox(x1,y1,x2,y2); b->SetLineColor(lc); b->SetLineWidth(boxLineWidth); b->SetFillStyle(0); b->Draw("same"); };
  drawBox(-0.5,-0.5, 0.5, 0.5, kBlack);   // center
  drawBox(-0.5, 1.5, 0.5, 2.5, kBlue+1);  // above
  drawBox(-0.5,-1.5, 0.5,-0.5, kRed+1);   // below
  drawBox( 1.5,-0.5, 2.5, 0.5, kGreen+2); // right
  drawBox(-1.5,-0.5,-0.5, 0.5, kMagenta); // left
  TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
  lab.DrawLatex(0.50,0.965,title);
  return h;
}

// Draws the CTI "fraction relative to center" bar chart + legend on the current pad
// (Above/Below/Right/Left bars plus a reference line at 1.0 for the center).
static void DrawFrontRegionBars(const TString& frameName, const TString& centerBinLabel,
                                 double fAbove, double fBelow, double fRight, double fLeft,
                                 double centerCharge){
  gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.16); gPad->SetBottomMargin(0.18);
  TH1F* frame = new TH1F(frameName, ";Region;Fraction (w.r.t. center)", 5, 0.5, 5.5);
  frame->GetXaxis()->SetBinLabel(1,"Above");
  frame->GetXaxis()->SetBinLabel(2,"Below");
  frame->GetXaxis()->SetBinLabel(3,"Right");
  frame->GetXaxis()->SetBinLabel(4,"Left");
  frame->GetXaxis()->SetBinLabel(5,centerBinLabel);
  const double yMax = 1.25*std::max(1.0, std::max({fAbove,fBelow,fRight,fLeft}));
  frame->SetMinimum(0.0); frame->SetMaximum(yMax); frame->Draw("AXIS");
  TLine* ref = new TLine(0.5,1.0,5.5,1.0); ref->SetLineStyle(2); ref->SetLineColor(kGray+2); ref->Draw();
  auto drawBar=[&](int bin,double y,Color_t col){
    double x1=frame->GetXaxis()->GetBinLowEdge(bin), x2=frame->GetXaxis()->GetBinUpEdge(bin);
    TBox* b=new TBox(x1,0.0,x2,y); b->SetFillColorAlpha(col,0.35); b->SetLineColor(col); b->SetLineWidth(3); b->Draw("same"); return b; };
  TBox* bAbove  = drawBar(1,fAbove, kBlue+1);
  TBox* bBelow  = drawBar(2,fBelow, kRed+1);
  TBox* bRight  = drawBar(3,fRight, kGreen+2);
  TBox* bLeft   = drawBar(4,fLeft,  kMagenta);
  TBox* bCenter = drawBar(5,1.0,    kGray+1);

  auto leg = new TLegend(0.18, 0.58, 0.55, 0.88);
  leg->SetBorderSize(0); leg->SetFillStyle(0); leg->SetTextFont(42); leg->SetTextSize(0.035);
  leg->AddEntry(bCenter, Form("Center charge = %.0f", centerCharge), "f");
  leg->AddEntry(bAbove,  Form("Above  = %.3f", fAbove),  "f");
  leg->AddEntry(bBelow,  Form("Below  = %.3f", fBelow),  "f");
  leg->AddEntry(bRight,  Form("Right  = %.3f", fRight),  "f");
  leg->AddEntry(bLeft,   Form("Left   = %.3f", fLeft ),  "f");
  leg->Draw();
}

// Parameters of one fitted Gaussian peak (amplitude, mean, sigma) plus their fit errors.
struct GaussPeakFit {
    double amp=0, amp_err=0;
    double mu=0,  mu_err=0;
    double sigma=0, sigma_err=0;
};

// Draws `hist` as filled bars with a solid outline on top (the matplotlib-style look
// used for all the cluster-energy spectra in this macro). Assumes hist is already
// styled/binned; only sets the cosmetic bar/outline attributes and draws.
static void DrawBarWithOutline(TH1* hist, Color_t fillColor, double fillAlpha, Color_t lineColor,
                                double barWidth, double barOffset, const TString& outlineName,
                                Color_t outlineColor, int outlineWidth){
  hist->SetFillColorAlpha(fillColor, fillAlpha);
  hist->SetLineColor(lineColor);
  hist->SetBarWidth(barWidth);
  hist->SetBarOffset(barOffset);
  hist->Draw("BAR");
  TH1* outline = (TH1*)hist->Clone(outlineName);
  outline->SetFillStyle(0); outline->SetLineColor(outlineColor); outline->SetLineWidth(outlineWidth);
  outline->Draw("BAR SAME");
}

// Draws a 1D histogram (filled, hatched) on its own canvas with a title label, and saves
// it. Used for the plain per-extension distributions (pixel energy, cluster size, max
// pixel energy) that only differ in color/labels/x-range.
static void DrawAndSave1DHist(TH1* hist, const TString& xTitle, Color_t color,
                               const TString& canvasName, const TString& canvasTitle,
                               const TString& panelLabel, const TString& savePath,
                               double xRangeLo=0, double xRangeHi=-1){
  TCanvas* c = new TCanvas(canvasName, canvasTitle, 900, 600);
  c->cd(); gPad->SetGridx(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);
  hist->SetTitle(TString(";") + xTitle + ";Counts");
  if (xRangeHi > xRangeLo) hist->GetXaxis()->SetRangeUser(xRangeLo, xRangeHi);
  hist->SetLineColor(color);
  hist->SetFillColor(color);
  hist->SetFillStyle(3001);
  hist->Draw("Hist");
  TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
  lab.DrawLatex(0.50, 0.965, panelLabel);
  SaveCanvas(c, savePath);
  delete c;
}

// Draws a 2D histogram as a log-z COLZ plot on its own canvas with a title label, and
// saves it. Used for the plain per-extension 2D distributions (energy/max-pixel-energy
// vs cluster size) that only differ in data/labels.
static void DrawAndSave2DHist(TH2* hist, const TString& xTitle, const TString& yTitle,
                               const TString& canvasName, const TString& canvasTitle,
                               const TString& panelLabel, const TString& savePath){
  TCanvas* c = new TCanvas(canvasName, canvasTitle, 900, 700);
  c->cd(); gPad->SetLogz(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);
  hist->SetTitle("");
  hist->GetXaxis()->SetTitle(xTitle);
  hist->GetYaxis()->SetTitle(yTitle);
  hist->Draw("COLZ");
  TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
  lab.DrawLatex(0.50, 0.965, panelLabel);
  SaveCanvas(c, savePath);
  delete c;
}

// Result of fitting the Fe-55 Kalpha+Kbeta double-Gaussian to a cluster-energy histogram.
struct DoubleGaussFitResult {
  GaussPeakFit kAlpha, kBeta;
  TF1* fAlpha = nullptr;
  TF1* fBeta  = nullptr;   // null if no Kbeta fit was performed or it failed validation
  double chi2 = 0; int ndf = 0; double prob = 0;
};

// Fits the Fe-55 Kalpha peak (window [KALPHA_FIT_LO,KALPHA_FIT_HI]) then the Kbeta peak
// in a fixed-width [mu+2*sigma, +1.0 keV] window placed relative to the fitted Kalpha
// mu/sigma, drawing both on the current pad. `validateFit` controls whether a bad Kalpha
// fit (chi2/sigma/range checks) blocks the Kbeta attempt, and whether the Kbeta fit
// itself is validated and discarded on failure -- used for the back-side fit; the
// front-side fit passes validateFit=false and always keeps whatever it fits.
//
// NOTE: an earlier version of the back-side fit placed Kbeta in a much narrower
// [mu+2*sigma, +3*sigma] window. For a well-resolved (small sigma) Kalpha peak, that
// window undershoots the true ~0.6 keV Kalpha-Kbeta separation and lands in the dip
// between the two peaks rather than on the real Kbeta peak, so the fit reliably failed
// to converge -- confirmed via the "[FIT DIAG]" logging below on real back-side data.
// Both sides now use the same fixed +1.0 keV window, which empirically brackets the
// real Kbeta peak in both cases.
static DoubleGaussFitResult FitAndDrawKAlphaKBeta(TH1F* hist, const TString& fAlphaName,
                                                   const TString& fBetaName, bool validateFit,
                                                   const char* emptyHistWarning){
  DoubleGaussFitResult out;
  const double kAlphaLo = KALPHA_FIT_LO, kAlphaHi = KALPHA_FIT_HI;
  out.fAlpha = new TF1(fAlphaName, "gaus", kAlphaLo, kAlphaHi);
  double kBetaLo = 6.0, kBetaHi = 6.8; // defaults used for the chi2 window if Kbeta isn't fit

  if (hist->GetEntries() > 0) {
    out.fAlpha->SetParameters(hist->GetMaximum(), KALPHA_SEED_MU, KALPHA_SEED_SIGMA);
    out.fAlpha->SetParLimits(2, FIT_SIGMA_MIN, FIT_SIGMA_MAX);
    const int fitStatus = hist->Fit(out.fAlpha, "RQ0S");
    const double mu = out.fAlpha->GetParameter(1);
    const double sigma = out.fAlpha->GetParameter(2);
    const bool kAlphaOk = !validateFit ||
        (fitStatus == 0 && sigma > 0.01 && sigma < 1.0 && mu >= ENE_MIN_FE55 && mu <= ENE_MAX_FE55);

    // Diagnostic: report exactly which validation criterion rejected the Kalpha fit
    // (only relevant when validateFit=true, i.e. currently the back-side path).
    if (validateFit && !kAlphaOk) {
      std::cerr << "[FIT DIAG] " << fAlphaName << ": Kalpha fit REJECTED -- "
                << "status=" << fitStatus << " mu=" << mu << " sigma=" << sigma
                << " | failed:";
      if (fitStatus != 0) std::cerr << " fitStatus!=0";
      if (!(sigma > 0.01)) std::cerr << " sigma<=0.01";
      if (!(sigma < 1.0)) std::cerr << " sigma>=1.0";
      if (!(mu >= ENE_MIN_FE55)) std::cerr << " mu<ENE_MIN_FE55(" << ENE_MIN_FE55 << ")";
      if (!(mu <= ENE_MAX_FE55)) std::cerr << " mu>ENE_MAX_FE55(" << ENE_MAX_FE55 << ")";
      std::cerr << std::endl;
    }

    if (kAlphaOk) {
      kBetaLo = std::max(ENE_MIN_FE55, mu + 2*sigma);
      kBetaHi = std::min(ENE_MAX_FE55, kBetaLo + 1.0);
      if (kBetaHi <= kBetaLo) kBetaHi = std::min(ENE_MAX_FE55, kBetaLo + 0.5);

      out.fBeta = new TF1(fBetaName, "gaus", kBetaLo, kBetaHi);
      out.fBeta->SetParameters(hist->GetMaximum() / 3.0, (kBetaLo + kBetaHi) / 2.0, 0.12);
      out.fBeta->SetParLimits(2, FIT_SIGMA_MIN, FIT_SIGMA_MAX);
      const int betaStatus = hist->Fit(out.fBeta, "RQ0S");
      if (validateFit) {
        const double sigmaB = out.fBeta->GetParameter(2);
        const bool kBetaOk = (betaStatus == 0 && sigmaB > 0.01 && sigmaB < 1.0);
        if (!kBetaOk) {
          // Diagnostic: report exactly which criterion rejected the Kbeta fit.
          std::cerr << "[FIT DIAG] " << fBetaName << ": Kbeta fit REJECTED -- "
                    << "status=" << betaStatus << " sigma=" << sigmaB
                    << " (window=[" << kBetaLo << "," << kBetaHi << "]) | failed:";
          if (betaStatus != 0) std::cerr << " fitStatus!=0";
          if (!(sigmaB > 0.01)) std::cerr << " sigma<=0.01";
          if (!(sigmaB < 1.0)) std::cerr << " sigma>=1.0";
          std::cerr << std::endl;
          delete out.fBeta; out.fBeta = nullptr;
        }
      }
    }
  }

  if (out.fBeta) {
    out.fAlpha->SetLineColor(kRed+1);  out.fAlpha->SetLineWidth(3);
    out.fBeta->SetLineColor(kBlue+1);  out.fBeta->SetLineWidth(3);
    out.fAlpha->Draw("SAME"); out.fBeta->Draw("SAME");
    out.kAlpha.mu = out.fAlpha->GetParameter(1); out.kAlpha.mu_err = out.fAlpha->GetParError(1);
    out.kAlpha.sigma = out.fAlpha->GetParameter(2); out.kAlpha.sigma_err = out.fAlpha->GetParError(2);
    out.kAlpha.amp = out.fAlpha->GetParameter(0); out.kAlpha.amp_err = out.fAlpha->GetParError(0);
    out.kBeta.mu = out.fBeta->GetParameter(1); out.kBeta.mu_err = out.fBeta->GetParError(1);
    out.kBeta.sigma = out.fBeta->GetParameter(2); out.kBeta.sigma_err = out.fBeta->GetParError(2);
    out.kBeta.amp = out.fBeta->GetParameter(0); out.kBeta.amp_err = out.fBeta->GetParError(0);
  } else {
    if (hist->GetEntries() == 0) std::cerr << emptyHistWarning << std::endl;
    out.kAlpha.mu = KALPHA_SEED_MU; out.kAlpha.sigma = 0.12;
    out.kBeta.mu  = 6.4;            out.kBeta.sigma  = 0.12;
  }

  // Chi2 over the union of the Kalpha/Kbeta fit windows (using the actual attempted
  // window even when the corresponding fit failed/was skipped).
  auto inAlpha = [&](double x){ return x >= kAlphaLo && x <= kAlphaHi; };
  auto inBeta  = [&](double x){ return x >= kBetaLo  && x <= kBetaHi;  };
  for (int ib = 1; ib <= hist->GetNbinsX(); ++ib) {
    const double x = hist->GetXaxis()->GetBinCenter(ib);
    if (!(inAlpha(x) || inBeta(x))) continue;
    const double y = hist->GetBinContent(ib);
    const double yfit = out.fAlpha->Eval(x) + (out.fBeta ? out.fBeta->Eval(x) : 0.0);
    const double var = y + 1e-6;
    out.chi2 += (y - yfit) * (y - yfit) / var;
    ++out.ndf;
  }
  out.ndf = std::max(1, out.ndf - 6);
  out.prob = TMath::Prob(out.chi2, out.ndf);
  return out;
}

// Draws the "front+back overlaid" cluster-energy panel (two stacked bar histograms with
// their fitted Kalpha/Kbeta Gaussian components and two per-side legends) on the current
// pad. Used both for a single extension's own canvas and, reusing the same stored fit
// results, inside the 4-extension comparison grid -- `objId` (extension number, or loop
// index) seeds unique ROOT object names so the two call sites (and repeated calls across
// extensions) don't collide. `ymaxPanel` is computed differently by each caller (see call
// sites) and passed in rather than recomputed here.
static void DrawFrontBackEnergyPanel(TH1F* hEnergyFront, TH1F* hEnergyBack,
                                      const GaussPeakFit& frontKAlpha, const GaussPeakFit& frontKBeta,
                                      const GaussPeakFit& backKAlpha,  const GaussPeakFit& backKBeta,
                                      double ymaxPanel, int objId, const TString& title)
{
  TH1F* h_front=(TH1F*)hEnergyFront->Clone();
  h_front->SetTitle(";Energy [keV];Counts");
  h_front->SetFillColorAlpha(kRed+1,0.5);
  h_front->SetLineColor(kRed+2);
  h_front->SetBarWidth(0.95);
  h_front->SetBarOffset(0.05);
  h_front->SetMaximum(1.15 * ymaxPanel);
  h_front->GetYaxis()->SetNoExponent(true);   // avoid "x10^3" label
  h_front->GetYaxis()->SetMaxDigits(3);
  h_front->Draw("BAR");

  auto frontOutline = (TH1F*)hEnergyFront->Clone(Form("hE_outline_front_fb_%d",objId));
  frontOutline->SetFillStyle(0); frontOutline->SetLineColor(kBlack);
  frontOutline->SetLineWidth(1);
  frontOutline->Draw("BAR SAME");

  TH1F* h_back=(TH1F*)hEnergyBack->Clone();
  h_back->SetTitle(";Energy [keV];Counts");
  h_back->SetFillColorAlpha(kBlue+1,0.5);
  h_back->SetLineColor(kBlue+2);
  h_back->SetBarWidth(0.95);
  h_back->SetBarOffset(0.05);
  h_back->SetMaximum(1.15 * ymaxPanel);
  h_back->GetYaxis()->SetNoExponent(true);
  h_back->GetYaxis()->SetMaxDigits(3);
  h_back->Draw("BAR SAME");

  auto backOutline = (TH1F*)hEnergyBack->Clone(Form("hE_outline_back_fb_%d",objId));
  backOutline->SetFillStyle(0); backOutline->SetLineColor(kBlack);
  backOutline->SetLineWidth(1);
  backOutline->Draw("BAR SAME");

  // Reconstruct the fitted Gaussian curves from the stored parameters (the TF1 objects
  // from the original fits are gone by the time this is called from the comparison grid).
  TF1* g1=new TF1(Form("cEg1_%d",objId),"gaus", frontKAlpha.mu-4*frontKAlpha.sigma, frontKAlpha.mu+4*frontKAlpha.sigma);
  TF1* g2=new TF1(Form("cEg2_%d",objId),"gaus", frontKBeta.mu-4*frontKBeta.sigma, frontKBeta.mu+4*frontKBeta.sigma);
  TF1* g3=new TF1(Form("cEg3_%d",objId),"gaus", backKAlpha.mu-4*backKAlpha.sigma, backKAlpha.mu+4*backKAlpha.sigma);
  TF1* g4=new TF1(Form("cEg4_%d",objId),"gaus", backKBeta.mu-4*backKBeta.sigma, backKBeta.mu+4*backKBeta.sigma);

  g1->SetParameters(frontKAlpha.amp, frontKAlpha.mu, std::max(0.03, frontKAlpha.sigma));
  g2->SetParameters(frontKBeta.amp, frontKBeta.mu, std::max(0.03, frontKBeta.sigma));
  g1->SetNpx(600); g2->SetNpx(600);
  g3->SetParameters(backKAlpha.amp, backKAlpha.mu, std::max(0.03, backKAlpha.sigma));
  g4->SetParameters(backKBeta.amp, backKBeta.mu, std::max(0.03, backKBeta.sigma));
  g3->SetNpx(600); g4->SetNpx(600);

  g1->SetLineColor(kRed+1);  g1->SetLineWidth(4); g1->SetLineStyle(1); g1->Draw("SAME");
  g2->SetLineColor(kRed-7);  g2->SetLineWidth(4); g2->SetLineStyle(1); g2->Draw("SAME");
  g3->SetLineColor(kBlue+1); g3->SetLineWidth(2); g3->SetLineStyle(1); g3->Draw("SAME");
  g4->SetLineColor(kBlue-7); g4->SetLineWidth(2); g4->SetLineStyle(1); g4->Draw("SAME");

  auto leg_front = new TLegend(0.145, 0.65, 0.385, 0.89);
  leg_front->SetBorderSize(1); leg_front->SetFillStyle(1001); leg_front->SetTextFont(42); leg_front->SetTextSize(0.030);
  leg_front->AddEntry(h_front, "#sigma_{xy}<1 (Front)", "f");
  leg_front->AddEntry(g1, Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", frontKAlpha.mu, frontKAlpha.mu_err, frontKAlpha.sigma, frontKAlpha.sigma_err), "l");
  leg_front->AddEntry(g2, Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", frontKBeta.mu, frontKBeta.mu_err, frontKBeta.sigma, frontKBeta.sigma_err), "l");

  auto leg_back = new TLegend(0.64, 0.65, 0.88, 0.89);
  leg_back->SetBorderSize(1); leg_back->SetFillStyle(1001); leg_back->SetTextFont(42); leg_back->SetTextSize(0.030);
  leg_back->AddEntry(h_back, "#sigma_{xy}>1 (Back)", "f");
  leg_back->AddEntry(g3, Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", backKAlpha.mu, backKAlpha.mu_err, backKAlpha.sigma, backKAlpha.sigma_err), "l");
  leg_back->AddEntry(g4, Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", backKBeta.mu, backKBeta.mu_err, backKBeta.sigma, backKBeta.sigma_err), "l");

  leg_front->Draw();
  leg_back->Draw();

  TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
  lab.DrawLatex(0.50,0.965,title);
}

// -------------------- results container per extension ----------------
struct ExtResults {
    // bookkeeping
    int    nFiles = 0;
    Long64_t totalClusters = 0;
    int    nAcceptedEnergy = 0;  // number passing energy window (for bookkeeping)
    int    nAcceptedFront  = 0;
    int    nAcceptedBack   = 0;

    // composites & spectra
    // hFront/hBack: Fe-55-cut composite CTI maps (see DrawFrontCompositeHeatmap).
    // hEnergy_front/hEnergy_back: Fe-55-cut front/back cluster-energy spectra.
    // henergy_sigmaxy: Fe-55-cut energy vs sigma_xy.
    // hEnergy: wide (0-80 keV) Am241 spectrum, deliberately NOT Fe-55-cut.
    // hPixelEnergy/hClusterSize/hMaxPixelEnergy/henergy_clustersize/
    // hmaxpixelenergy_clustersize: filled for every cluster regardless of energy
    // (NOT Fe-55-cut) -- see the fill loop in ProcessOneExtension.
    TH2D*  hFront  = nullptr;
    TH2D*  hBack   = nullptr;
    TH1F*  hEnergy = nullptr;
    TH1F*  hEnergy_front = nullptr;
    TH1F*  hEnergy_back = nullptr;
    TH2F*  henergy_sigmaxy = nullptr;
    TH1F*  hPixelEnergy = nullptr;
    TH1I*  hClusterSize = nullptr;
    TH1F*  hMaxPixelEnergy = nullptr;
    TH2F*  henergy_clustersize = nullptr;
    TH2F*  hmaxpixelenergy_clustersize = nullptr;

    // projections (normalized copies for compare tiles)
    TH1D*  projX   = nullptr;    // back: ProjectionX over y ∈ [-4,-3]
    TH1D*  projY   = nullptr;    // back: ProjectionY over x ∈ [-4,-3]
    double projY_int_e = 0.0;    // "hleft" integral [e-]
    double projX_int_e = 0.0;    // "hbelow" integral [e-]
    double projY_mean=0, projY_rms=0, projY_skew=0;
    double projX_mean=0, projX_rms=0, projX_skew=0;

    // integrals (e-)
    double frontIntegral_e = 0.0;
    double backIntegral_e  = 0.0;

    // CTI fractions (front)
    double centerCharge = 0.0;
    double fAbove=0.0, fBelow=0.0, fRight=0.0, fLeft=0.0;

    // Energy fit, front-side events (sigma_xy < SIGMA_FRONT_CUT): two independent
    // Gaussians fit in separate windows around the Fe-55 Kalpha (~5.9 keV) and
    // Kbeta (~6.4 keV) lines. No fit-quality validation is applied here (unlike
    // the back-side fit below) -- kept as-is rather than unified, see ProcessOneExtension.
    GaussPeakFit frontKAlpha, frontKBeta;
    double frontChi2=0; int frontNdf=0; double frontFitProb=0;

    // Energy fit, back-side events (sigma_xy > SIGMA_FRONT_CUT): same two-Gaussian
    // scheme and window placement as the front-side fit, but both the Kalpha and Kbeta
    // fits are validated (chi2/sigma sanity checks), falling back to seed defaults for
    // both peaks if either one fails -- see FitAndDrawKAlphaKBeta.
    GaussPeakFit backKAlpha, backKBeta;
    double backChi2=0; int backNdf=0; double backFitProb=0;

    // from info tree (per-extension calibration: gain, single-e resolution, DC)
    double gainext=0, gainext_err=0;
    double sigext=0, sigext_err=0;
    double dcext=0, dcext_err=0;
};

// Draws a per-extension TGraphErrors for one info-tree calibration quantity (gain,
// single-e resolution, or dark current) on the current pad, skipping extensions with
// no processed files. `xAxisTitleOverride`, when non-empty, replaces the x-axis title
// set implicitly by the ";Extension;<yTitle>" title string (only the gain panel does
// this in the original code -- preserved as an explicit opt-in rather than applied to
// all three, since the other two panels were never given that override).
static void DrawCalibrationGraph(const ExtResults R[5], double ExtResults::*value,
                                  double ExtResults::*err, Color_t color,
                                  const TString& yTitle, const TString& title,
                                  const TString& xAxisTitleOverride = ""){
  gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);
  double x[4], y[4], ex[4]={0,0,0,0}, ey[4];
  int n = 0;
  for (int e = 1; e <= 4; ++e) {
    if (R[e].nFiles == 0) continue;
    x[n] = e; y[n] = R[e].*value; ey[n] = R[e].*err; n++;
  }
  if (n == 0) return;
  TGraphErrors* gr = new TGraphErrors(n, x, y, ex, ey);
  gr->SetTitle(TString(";Extension;") + yTitle);
  gr->SetMarkerStyle(20); gr->SetMarkerSize(1.2); gr->SetMarkerColor(color);
  gr->Draw("AP");
  gr->GetXaxis()->SetNdivisions(4);
  gr->GetXaxis()->SetLimits(0.5, 4.5);
  if (!xAxisTitleOverride.IsNull()) gr->GetXaxis()->SetTitle(xAxisTitleOverride);
  TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.04);
  lab.DrawLatex(0.5, 0.92, title);
}

// --------------------- per-extension processing ----------------------
// Reads the clusterRec tree(s) for one CCD extension, fills the composite/spectrum
// histograms, fits the Fe-55 Kalpha/Kbeta peaks (front- and back-side separately),
// and saves the per-extension diagnostic canvases under outDir/ext<extNum>/.
static ExtResults ProcessOneExtension(const std::vector<TString>& files,
                                      int extNum,
                                      int image_number,
                                      const TString& outDir,
                                      const char* moduleUsed,
                                      bool interactiveAm241)
{
  ExtResults R;
  R.nFiles = (int)files.size();
  if (R.nFiles==0) return R;

  // Prepare output folder for this extension
  TString extDir = Form("%s/ext%d", outDir.Data(), extNum);
  EnsureDir(extDir);

  // Build a chain with only those files
  TChain chain("clustersRec");
  for (auto& f : files) chain.Add(f);

  // Read info tree from first file (gain, single-e resolution, DC per extension).
  // Branches are named "ext<N>_<quantity>[_err]"; missing branches are left at 0.
  {
    TFile* finfo = TFile::Open(files[0]);
    if (finfo && !finfo->IsZombie()) {
      TTree* tinfo = (TTree*)finfo->Get("info");
      if (tinfo && tinfo->GetEntries() > 0) {
        TString extPrefix = Form("ext%d_", extNum);
        double gainVal=0, gainErr=0, sigVal=0, sigErr=0, dcVal=0, dcErr=0;
        struct CalibVar { const char* quantity; double* value; double* err; };
        const CalibVar vars[] = {
          {"gainext", &gainVal, &gainErr},
          {"sigext",  &sigVal,  &sigErr},
          {"dcext",   &dcVal,   &dcErr},
        };
        for (const auto& v : vars) {
          TString br    = extPrefix + v.quantity;
          TString brErr = br + "_err";
          if (tinfo->FindBranch(br.Data()))    tinfo->SetBranchAddress(br.Data(),    v.value);
          if (tinfo->FindBranch(brErr.Data())) tinfo->SetBranchAddress(brErr.Data(), v.err);
        }
        tinfo->GetEntry(0);
        R.gainext = gainVal; R.gainext_err = gainErr;
        R.sigext  = sigVal;  R.sigext_err  = sigErr;
        R.dcext   = dcVal;   R.dcext_err   = dcErr;
      }
      finfo->Close();
    }
    if (finfo) delete finfo;
  }

  // Branches read from the clustersRec tree, one entry (vector) per cluster:
  //   pixels_x/pixels_y/pixels_E : per-pixel column/row/energy[keV] within each cluster
  //   PosX/PosY                  : cluster mean position [pixels]
  //   wSTD_XY/fSTD_X/fSTD_Y      : cluster width (rmsxy/rmsx/rmsy) [pixels]
  //   Energy                     : cluster total energy [keV]
  //   Qmax                       : cluster's single brightest pixel energy [keV]
  //   Npix                       : cluster size [pixels]
  // NOTE: rmsxy (wSTD_XY) is bound here but never read below -- sigma_xy is instead
  // recomputed from rmsx/rmsy (sqrt(0.5*(sx^2+sy^2))) rather than taken from the tree.
  std::vector<std::vector<int>>   *pixels_x=nullptr, *pixels_y=nullptr;
  std::vector<std::vector<float>> *pixels_E=nullptr;
  std::vector<float> *PosX=nullptr, *PosY=nullptr;
  std::vector<float> *rmsxy=nullptr, *rmsx=nullptr, *rmsy=nullptr;
  std::vector<float> *Energy=nullptr;
  std::vector<float> *Qmax=nullptr;
  std::vector<int> *Npix=nullptr;

  chain.SetBranchAddress("pixels_x", &pixels_x);
  chain.SetBranchAddress("pixels_y", &pixels_y);
  chain.SetBranchAddress("pixels_E", &pixels_E);
  chain.SetBranchAddress("PosX",     &PosX);
  chain.SetBranchAddress("PosY",     &PosY);
  chain.SetBranchAddress("wSTD_XY",  &rmsxy);
  chain.SetBranchAddress("fSTD_X",   &rmsx);
  chain.SetBranchAddress("fSTD_Y",   &rmsy);
  chain.SetBranchAddress("Energy",   &Energy);
  chain.SetBranchAddress("Qmax",     &Qmax);
  chain.SetBranchAddress("Npix",     &Npix);

  // Sensor image geometry (columns x rows) per readout/module configuration, identified
  // by image_number -- these dimensions are convention, not derived from anything else
  // in this file, so confirm them against the module config before adding a new case.
  // NOTE: image_number 6/7 additionally narrow the composite-cluster grid on one axis
  // (NBINS_COMP_X/Y, a *global*, so this mutation persists for every extension/image
  // processed afterward in the same run -- see also the front/back cut-variable
  // selection below, which pairs 6 with 9 and 7 with 10 but only 6/7 appear here).
  int nColumns=0, nRows=0;
  if (image_number==4){ nColumns=6400; nRows=1600; }
  else if (image_number==5){ nColumns=1280; nRows=480; }
  else if (image_number==6){ nColumns=640;  nRows=1600; NBINS_COMP_Y=41;}
  else if (image_number==7){ nColumns=640;  nRows=1600; NBINS_COMP_X=41;}
  else if (image_number==8){ nColumns=3500;  nRows=500; }
  else if (image_number==11){ nColumns=6300;  nRows=12; }
  else { nColumns=640; nRows=640; }

  // Composite cluster histograms for CTI studies
  R.hFront  = new TH2D(Form("hFront_ext%d",extNum),  "Composite Cluster Front Events",NBINS_COMP_X, -RANGE_COMP, RANGE_COMP,NBINS_COMP_Y, -RANGE_COMP, RANGE_COMP);
  R.hBack   = new TH2D(Form("hBack_ext%d",extNum),   "Composite Cluster Back Events",NBINS_COMP_X, -RANGE_COMP, RANGE_COMP,NBINS_COMP_Y, -RANGE_COMP, RANGE_COMP);
  
  // Cluster energy distribution (in keV)
  R.hEnergy = new TH1F(Form("hEnergy_ext%d",extNum), "hClusterE",NBINS_ENERGY_AM241, ENE_MIN_AM241, ENE_MAX_AM241);
  R.hEnergy_front = new TH1F(Form("hEnergy_ext_front%d",extNum), "hClusterE",NBINS_ENERGY_FE55, ENE_MIN_FE55, ENE_MAX_FE55);
  R.hEnergy_back = new TH1F(Form("hEnergy_ext_back%d",extNum), "hClusterE",NBINS_ENERGY_FE55, ENE_MIN_FE55, ENE_MAX_FE55);

  // Cluster energy vs sigma_xy
  R.henergy_sigmaxy= new TH2F(Form("hE_sigmaxy_ext%d",extNum),"Energy vs #sigma_{xy}",
                              NBINS_ENERGY_FE55, ENE_MIN_FE55, ENE_MAX_FE55,
                              NBINS_SIGMA_XY_AXIS, SIGMA_XY_AXIS_MIN, SIGMA_XY_AXIS_MAX);

  // Cluster size distribution (in pixels)
  R.hClusterSize = new TH1I(Form("hClusterSize_ext%d",extNum), "hClusterSize",
                            NBINS_CLUSTER_SIZE, CLUSTER_SIZE_MIN, CLUSTER_SIZE_MAX);

  // Max pixel energy distribution (in keV)
  R.hMaxPixelEnergy = new TH1F(Form("hMaxPixelE_ext%d",extNum), "hMaxPixelE",
                               NBINS_ENERGY_FE55, ENE_MIN_FE55, ENE_MAX_FE55);

  // Pixel energy distribution (in electrons)
  R.hPixelEnergy = new TH1F(Form("hPixelE_ext%d",extNum), "hPixelE",E_PIXEL_BIN_NUM, eNUM_MIN, eNUM_MAX); // electrons

  // Cluster energy vs cluster size
  R.henergy_clustersize = new TH2F(Form("hE_clustersize_ext%d",extNum),"Energy vs Cluster Size",
                                   NBINS_ENERGY_WIDE, ENERGY_WIDE_MIN, ENE_MAX_FE55,
                                   NBINS_CLUSTER_SIZE, CLUSTER_SIZE_MIN, CLUSTER_SIZE_MAX);

  R.hmaxpixelenergy_clustersize = new TH2F(Form("hMaxPixelE_clustersize_ext%d",extNum),"Max Pixel Energy vs Cluster Size",
                                           NBINS_ENERGY_WIDE, ENERGY_WIDE_MIN, ENE_MAX_FE55,
                                           NBINS_CLUSTER_SIZE, CLUSTER_SIZE_MIN, CLUSTER_SIZE_MAX);

  // Additional distributions (saved later)
  TH1F* hPosX = new TH1F(Form("hPosX_ext%d",extNum), "PosX",
                         NBINS_POS_PLACEHOLDER, POS_PLACEHOLDER_MIN, POS_PLACEHOLDER_MAX);   // range updated below
  TH1F* hPosY = new TH1F(Form("hPosY_ext%d",extNum), "PosY",
                         NBINS_POS_PLACEHOLDER, POS_PLACEHOLDER_MIN, POS_PLACEHOLDER_MAX);
  TH1F* hSigmaxy = new TH1F(Form("hSigmaxy_ext%d",extNum), "Sigmaxy", NBINS_SIGMA_DIST, SIGMA_DIST_MIN, SIGMA_DIST_MAX);
  TH1F* hSigmax  = new TH1F(Form("hSigmax_ext%d", extNum),  "Sigmax",  NBINS_SIGMA_DIST, SIGMA_DIST_MIN, SIGMA_DIST_MAX);
  TH1F* hSigmay  = new TH1F(Form("hSigmay_ext%d", extNum),  "Sigmay",  NBINS_SIGMA_DIST, SIGMA_DIST_MIN, SIGMA_DIST_MAX);
  TH2F* hposx_sigmaxy = nullptr;
  TH2F* hPosXY = nullptr;
  // finalize axis ranges for the helper hists
  hPosX->GetXaxis()->SetLimits(IMAGE_ORIGIN, nColumns);
  hPosY->GetXaxis()->SetLimits(IMAGE_ORIGIN, nRows);
  hPosXY = new TH2F(Form("hPosXY_ext%d",extNum), "PosXY",
                    std::max(MIN_POS_XY_BINS, nColumns/POS_XY_COARSEN_FACTOR), IMAGE_ORIGIN, nColumns,
                    std::max(MIN_POS_XY_BINS, nRows/POS_XY_COARSEN_FACTOR),    IMAGE_ORIGIN, nRows);
  hposx_sigmaxy = new TH2F(Form("hposx_sigmaxy_ext%d",extNum), "posx_sigmaxy",
                           nColumns, IMAGE_ORIGIN, nColumns,
                           NBINS_SIGMA_XY_AXIS, SIGMA_XY_AXIS_MIN, SIGMA_XY_AXIS_MAX);

  // Loop
  const Long64_t nEntries = chain.GetEntries();
  for (Long64_t i=0;i<nEntries;++i){
    chain.GetEntry(i);
    const size_t nClusters = PosX->size();
    R.totalClusters += nClusters;

    for (size_t n=0;n<nClusters;++n){
      // cluster properties
      const double E = Energy->at(n); // cluster energy in keV
      const double x = PosX->at(n); // cluster mean position x in pixels
      const double y = PosY->at(n); // cluster mean position y in pixels
      const double sx= rmsx->at(n); // cluster sigma x in pixels
      const double sy= rmsy->at(n); // cluster sigma y in pixels
      const double sxy= std::sqrt(0.5*(sx*sx + sy*sy)); // cluster sigma_xy in pixels
      const double npix = Npix->at(n); // cluster size in pixels
      const double qmax= Qmax->at(n); // max pixel energy in keV for this cluster

      // pixel properties
      const auto& px = pixels_x->at(n);
      const auto& py = pixels_y->at(n);
      const auto& pE = pixels_E->at(n);

      // histogram pixel energy in units of electrons instead of keV
      for (auto e : pE) {
        R.hPixelEnergy->Fill(e*1000.0/E_PER_e);
      }

      R.hClusterSize->Fill(npix);
      R.hMaxPixelEnergy->Fill(qmax);
      R.henergy_clustersize->Fill(E, npix);
      R.hmaxpixelenergy_clustersize->Fill(qmax, npix);
      R.hEnergy->Fill(E);

      // book-keeping distributions for events in energy window
      if (E>ENE_MIN_FE55 && E<ENE_MAX_FE55){
        hSigmaxy->Fill(sxy); hSigmax->Fill(sx); hSigmay->Fill(sy);
        hPosX->Fill(x); hPosY->Fill(y);
        hPosXY->Fill(x,y,E);
        hposx_sigmaxy->Fill(x, sxy, E);
        R.henergy_sigmaxy->Fill(E, sxy);
        R.nAcceptedEnergy++;
      }

      // Choose which width measurement discriminates front/back events. Default is the
      // combined sigma_xy; image_number 6/9 use sigma_x alone and 7/10 use sigma_y alone
      // (matching the single-axis composite narrowing applied for 6/7 above), presumably
      // because those configurations only have diffusion information on one axis.
      double sCutVar = sxy;
      if (image_number==6 || image_number==9) sCutVar = sx;
      if (image_number==7 || image_number==10) sCutVar = sy;

      // FRONT selection: shallow-interaction events (small diffusion => small sigma_xy)
      if (E>ENE_MIN_FE55 && E<ENE_MAX_FE55 && sCutVar < SIGMA_FRONT_CUT){
        R.hEnergy_front->Fill(E);

        TH2F* hview = VectorsToTH2F(px, py, pE, "h2_front_tmp");
        AccumulateComposite(R.hFront, hview, x, y);
        delete hview;
        R.nAcceptedFront++;
      }

      // BACK selection: deep-interaction events (large diffusion => large sigma_xy)
      if (E>ENE_MIN_FE55 && E<ENE_MAX_FE55 && sCutVar > SIGMA_FRONT_CUT){
        R.hEnergy_back->Fill(E);

        TH2F* hview = VectorsToTH2F(px, py, pE, "h2_back_tmp");
        AccumulateComposite(R.hBack, hview, x, y);
        delete hview;
        R.nAcceptedBack++;
      }
    } // clusters
  } // entries

  // Normalize composites
  if (R.nAcceptedFront>0) R.hFront->Scale(1.0/R.nAcceptedFront);
  if (R.nAcceptedBack >0) R.hBack ->Scale(1.0/R.nAcceptedBack);

  // Integrals in electrons
  R.frontIntegral_e = R.hFront->Integral()*1000.0/E_PER_e;
  R.backIntegral_e  = R.hBack ->Integral()*1000.0/E_PER_e;

  // --------- FRONT: CTI fractions w.r.t center ----------
  // Regions are one pixel box away from the center box in each direction
  // (not the immediately-adjacent box) -- matches the boxes drawn on the
  // front-composite heatmap in DrawFrontCompositeHeatmap below:
  // center: x,y in [-0.5,0.5]
  // above : x in [-0.5,0.5],   y in [1.5,2.5]
  // below : x in [-0.5,0.5],   y in [-1.5,-0.5]
  // right : x in [1.5,2.5],    y in [-0.5,0.5]
  // left  : x in [-1.5,-0.5],  y in [-0.5,0.5]
  if (R.hFront){
    const double centerSum = SumRect(R.hFront, -0.5, 0.5,  -0.5,  0.5);
    const double aboveSum  = SumRect(R.hFront, -0.5, 0.5, 1.5, 2.5);
    const double belowSum  = SumRect(R.hFront, -0.5, 0.5,  -1.5, -0.5);
    const double rightSum  = SumRect(R.hFront,  1.5, 2.5, -0.5, 0.5);
    const double leftSum   = SumRect(R.hFront, -1.5,-0.5,  -0.5,  0.5);
    R.centerCharge = centerSum;
    if (centerSum>0){
      R.fAbove = aboveSum/centerSum; R.fBelow = belowSum/centerSum;
      R.fRight = rightSum/centerSum; R.fLeft  = leftSum/centerSum;
    }
  }

  // ---------------- BACK: CTI map (Fe-55 cut) + projections & stats ----------------
  if (R.hBack){
    // Projections on the [-4,-3] pixel slab -- far enough from the cluster center (which
    // sits at the origin of this re-centered composite) to sample the CTI trailing tail
    // rather than the cluster's own core charge.
    TH1D* hbelow = ProjXRect(R.hBack, Form("hbelow_ext%d",extNum), -4.0, -3.0);
    TH1D* hleft  = ProjYRect(R.hBack, Form("hleft_ext%d", extNum), -4.0, -3.0);

    // Store stats BEFORE normalization
    R.projY_int_e = hleft ->Integral()*1000.0/E_PER_e;
    R.projX_int_e = hbelow->Integral()*1000.0/E_PER_e;

    R.projY_mean = hleft->GetMean();  R.projY_rms = hleft->GetRMS();  R.projY_skew = hleft->GetSkewness();
    R.projX_mean = hbelow->GetMean(); R.projX_rms = hbelow->GetRMS(); R.projX_skew = hbelow->GetSkewness();

    // Normalize for overlays in compare canvases
    if (hbelow->Integral()>0) hbelow->Scale(1.0/hbelow->Integral());
    if (hleft ->Integral()>0) hleft ->Scale(1.0/hleft ->Integral());
    R.projX = (TH1D*)hbelow->Clone(Form("projX_ext%d",extNum));
    R.projY = (TH1D*)hleft ->Clone(Form("projY_ext%d",extNum));

    // Per-extension canvas: composite + projections
    gStyle->SetOptStat(0);
    TCanvas* cBack = new TCanvas(Form("cBack_ext%d",extNum),"Back composite + projections",1200,550);
    cBack->Divide(2,1);
    // Left: heatmap
    cBack->cd(1);
    gPad->SetRightMargin(0.15); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12); gPad->SetLogz();
    TH2D* hb = (TH2D*)R.hBack->Clone();
    hb->GetXaxis()->SetRangeUser(-4,4); hb->GetYaxis()->SetRangeUser(-4,4);
    hb->SetTitle("");
    hb->GetXaxis()->SetTitle("#Delta x [pix]");
    hb->GetYaxis()->SetTitle("#Delta y [pix]");
    hb->Draw("COLZ");
    // Title
    { TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
      lab.DrawLatex(0.50,0.965,Form("Back-Side CTI Map (Fe-55 cut) - %s - Ext %d", moduleUsed, extNum)); }

    // Right: projections with legend
    cBack->cd(2);
    gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.16); gPad->SetBottomMargin(0.16);
    R.projX->SetTitle(";Pixel offset [pix];Normalized counts");
    R.projX->SetLineColor(kBlack); 
    R.projX->SetLineWidth(3);
    R.projX->SetFillColorAlpha(kBlack, 0.08);
    R.projX->SetMarkerStyle(20);
    R.projX->SetMarkerColor(kBlack);

    R.projY->SetLineColor(kBlue+1);
    R.projY->SetLineWidth(3);
    R.projY->SetFillColorAlpha(kBlue+1, 0.08);
    R.projY->SetMarkerStyle(20);
    R.projY->SetMarkerColor(kBlue+1);

    double ymax = std::max(R.projX->GetMaximum(), R.projY->GetMaximum())*1.15;
    R.projX->SetMaximum(ymax); R.projX->SetMinimum(0.0);
    R.projX->Draw("HIST"); R.projY->Draw("HIST SAME");
    auto legB = new TLegend(0.57,0.72,0.89,0.88);
    legB->AddEntry(R.projX,"ProjX (Y: -4 #rightarrow -3 px)","l");
    legB->AddEntry(R.projY,"ProjY (X: -4 #rightarrow -3 px)","l");
    legB->Draw();
    SaveCanvas(cBack, Form("%s/back_CTI_map_Fe55", extDir.Data()));
    delete cBack;
  }

  // -------------- AM241: energy distribution ----------------
  {
    TCanvas* cAm241 = new TCanvas(Form("cAm241_ext%d",extNum),"Am241 energy distribution",900,600);
    cAm241->cd(); gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);
    R.hEnergy->SetTitle(";Energy [keV];Counts");
    DrawBarWithOutline(R.hEnergy, kBlue+1, 0.08, kBlue+2, 0.95, 0.025,
                       Form("hE_outline_am241_ext%d",extNum), kBlue+1, 2);
    SaveCanvas(cAm241, Form("%s/am241_energy_spectrum", extDir.Data()));
    ReleaseCanvasForInteractive(cAm241, interactiveAm241);
  }

  // -------------- FRONT: CTI map (Fe-55 cut) + region bars ----------------
  {
    gStyle->SetOptStat(0);

    TCanvas* cFront = new TCanvas(Form("cFront_ext%d",extNum),"Front composite + CTI bars",1200,550);
    cFront->Divide(2,1);

    cFront->cd(1);
    DrawFrontCompositeHeatmap(R.hFront, 0.15, 3, -1,
        Form("Front-Side CTI Map (Fe-55 cut) - %s - Ext %d", moduleUsed, extNum));

    cFront->cd(2);
    DrawFrontRegionBars(Form("frameF_ext%d",extNum), "Center=1.0",
                         R.fAbove, R.fBelow, R.fRight, R.fLeft, R.centerCharge);

    SaveCanvas(cFront, Form("%s/front_CTI_map_Fe55", extDir.Data()));
    delete cFront;
  }

  // -------------- ENERGY: double-Gaussian style plot - Front----------------
  {
    TCanvas* cE = new TCanvas(Form("cE_ext%d",extNum),"Cluster Energy (double Gaussian)",900,600);
    cE->cd(); gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);

    R.hEnergy_front->SetTitle(";Energy [keV];Counts");
    DrawBarWithOutline(R.hEnergy_front, kGray+1, 0.6, kGray+2, 0.95, 0.025,
                       Form("hE_outline_front_ext%d",extNum), kBlack, 2);

    const auto fit = FitAndDrawKAlphaKBeta(R.hEnergy_front, Form("g1_ext%d",extNum), Form("g2_ext%d",extNum),
        /*validateFit=*/false,
        Form("Front energy histogram empty for ext %d, using default params", extNum));
    R.frontKAlpha = fit.kAlpha; R.frontKBeta = fit.kBeta;
    R.frontChi2 = fit.chi2; R.frontNdf = fit.ndf; R.frontFitProb = fit.prob;

    // Legend with splitline labels
    auto leg = new TLegend(0.58,0.63,0.89,0.88);
    leg->SetBorderSize(1); leg->SetFillStyle(1001); leg->SetTextFont(42); leg->SetTextSize(0.035);
    leg->AddEntry(fit.fAlpha,Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", R.frontKAlpha.mu,R.frontKAlpha.mu_err,R.frontKAlpha.sigma,R.frontKAlpha.sigma_err),"l");
    leg->AddEntry(fit.fBeta,Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", R.frontKBeta.mu,R.frontKBeta.mu_err,R.frontKBeta.sigma,R.frontKBeta.sigma_err),"l");
    leg->AddEntry(R.hEnergy_front,"Data","f");
    leg->Draw();

    // per-tile title (consistency with compare)
    { TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
      lab.DrawLatex(0.50,0.965,Form("Front-Side Energy Spectrum - %s - Ext %d", moduleUsed, extNum)); }

    SaveCanvas(cE, Form("%s/front_energy_spectrum_Fe55fit", extDir.Data()));
    delete cE;
  }

  // -------------- ENERGY: double-Gaussian style plot - Back ----------------
  {
    // NOTE: this canvas used to share the ROOT name "cE_ext%d" with the front-side
    // canvas above, which ROOT would silently rename/replace -- given a distinct name.
    TCanvas* cE_back = new TCanvas(Form("cE_back_ext%d",extNum),"Cluster Energy (double Gaussian)",900,600);
    cE_back->cd(); gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);

    R.hEnergy_back->SetTitle(";Energy [keV];Counts");
    DrawBarWithOutline(R.hEnergy_back, kGray+1, 0.6, kGray+2, 0.95, 0.025,
                       Form("hE_outline_back_ext%d",extNum), kBlack, 2);

    const auto fit = FitAndDrawKAlphaKBeta(R.hEnergy_back, Form("g3_ext%d",extNum), Form("g4_ext%d",extNum),
        /*validateFit=*/true,
        Form("Back energy histogram empty for ext %d, using default params", extNum));
    R.backKAlpha = fit.kAlpha; R.backKBeta = fit.kBeta;
    R.backChi2 = fit.chi2; R.backNdf = fit.ndf; R.backFitProb = fit.prob;

    // Legend with splitline labels
    auto leg = new TLegend(0.58,0.63,0.89,0.88);
    leg->SetBorderSize(1); leg->SetFillStyle(1001); leg->SetTextFont(42); leg->SetTextSize(0.035);
    leg->AddEntry(fit.fAlpha,Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", R.backKAlpha.mu,R.backKAlpha.mu_err,R.backKAlpha.sigma,R.backKAlpha.sigma_err),"l");
    leg->AddEntry(fit.fBeta,Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", R.backKBeta.mu,R.backKBeta.mu_err,R.backKBeta.sigma,R.backKBeta.sigma_err),"l");
    leg->AddEntry(R.hEnergy_back,"Data","f");
    leg->Draw();

    // per-tile title (consistency with compare)
    { TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
      lab.DrawLatex(0.50,0.965,Form("Back-Side Energy Spectrum - %s - Ext %d", moduleUsed, extNum)); }

    SaveCanvas(cE_back, Form("%s/back_energy_spectrum_Fe55fit", extDir.Data()));
    delete cE_back;
  }

  // -------------- ENERGY: double-Gaussian style plot - Front & Back ----------------
  {
    TCanvas* cE_fb = new TCanvas(Form("cE_fb_ext%d",extNum),
                              "Cluster Energy: Front & Back", 900, 600);
    cE_fb->cd(); gPad->SetGridx(); gPad->SetGridy();
    gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);

    const double ymaxPanel = 1.2 * std::max({
      R.hEnergy_back->GetMaximum(),
      R.frontKAlpha.amp, R.frontKBeta.amp, R.backKAlpha.amp, R.backKBeta.amp,
      R.hEnergy_front->GetMaximum()
    });

    DrawFrontBackEnergyPanel(R.hEnergy_front, R.hEnergy_back,
                              R.frontKAlpha, R.frontKBeta, R.backKAlpha, R.backKBeta,
                              ymaxPanel, extNum,
                              Form("Front & Back Cluster Energy - %s - Ext %d", moduleUsed, extNum));

    SaveCanvas(cE_fb, Form("%s/front_back_energy_spectrum_Fe55fit", extDir.Data()));
    delete cE_fb;
  }

  // Pixel energy distribution
  DrawAndSave1DHist(R.hPixelEnergy, "Electron number [e-]", kBlue,
                     Form("cPixE_ext%d",extNum), "Pixel Energy Distribution",
                     Form("Pixel Energy Distribution - %s - Ext %d", moduleUsed, extNum),
                     Form("%s/pixel_energy_allclusters", extDir.Data()),
                     min_e_number_display, max_e_number_display);

  // Cluster size distribution
  DrawAndSave1DHist(R.hClusterSize, "Cluster Size [pixels]", kGreen+2,
                     Form("cSize_ext%d",extNum), "Cluster Size Distribution",
                     Form("Cluster Size Distribution - %s - Ext %d", moduleUsed, extNum),
                     Form("%s/cluster_size_allclusters", extDir.Data()));

  // Max pixel energy distribution
  DrawAndSave1DHist(R.hMaxPixelEnergy, "Max Pixel Energy [keV]", kBlue+2,
                     Form("cMaxPixE_ext%d",extNum), "Max Pixel Energy Distribution",
                     Form("Max Pixel Energy Distribution - %s - Ext %d", moduleUsed, extNum),
                     Form("%s/max_pixel_energy_allclusters", extDir.Data()));

  // Cluster energy vs cluster size
  DrawAndSave2DHist(R.henergy_clustersize, "Energy [keV]", "Cluster Size [pixels]",
                     Form("cESize_ext%d",extNum), "Cluster Energy vs Cluster Size",
                     Form("Cluster Energy vs Cluster Size - %s - Ext %d", moduleUsed, extNum),
                     Form("%s/energy_vs_cluster_size_allclusters", extDir.Data()));

  // Max pixel energy vs cluster size
  DrawAndSave2DHist(R.hmaxpixelenergy_clustersize, "Max Pixel Energy [keV]", "Cluster Size [pixels]",
                     Form("cMaxPixESize_ext%d",extNum), "Max Pixel Energy vs Cluster Size",
                     Form("Max Pixel Energy vs Cluster Size - %s - Ext %d", moduleUsed, extNum),
                     Form("%s/max_pixel_energy_vs_cluster_size_allclusters", extDir.Data()));

  // ----------------- Extra plots per extension ------------------
  {
    // posX vs sigmaXY
    TCanvas* c1=new TCanvas(Form("c_posx_sigmaxy_ext%d",extNum),"posx vs sigmaxy",800,600);
    c1->cd(); gPad->SetLogz(); hposx_sigmaxy->SetTitle("Mean x vs #sigma_{xy}");
    hposx_sigmaxy->GetXaxis()->SetTitle("Mean x [col]");
    hposx_sigmaxy->GetYaxis()->SetTitle("#sigma_{xy} [pix]");
    hposx_sigmaxy->Draw("COLZ");
    SaveCanvas(c1, Form("%s/posx_vs_sigmaXY_Fe55", extDir.Data()));
    delete c1;

    // energy vs sigmaXY
    TCanvas* c2=new TCanvas(Form("c_E_sigmaxy_ext%d",extNum),"Energy vs sigmaxy",800,600);
    c2->cd(); R.henergy_sigmaxy->SetTitle("Cluster Energy vs #sigma_{xy}");
    R.henergy_sigmaxy->GetXaxis()->SetTitle("Energy [keV]");
    R.henergy_sigmaxy->GetYaxis()->SetTitle("#sigma_{xy} [pix]");
    R.henergy_sigmaxy->Draw("COLZ");
    SaveCanvas(c2, Form("%s/energy_vs_sigmaXY_Fe55", extDir.Data()));
    delete c2;

    // sigma hists
    TCanvas* c3=new TCanvas(Form("c_sigmaxy_ext%d",extNum),"sigmaXY",700,500);
    hSigmaxy->SetTitle("#sigma_{xy} Distribution;#sigma_{xy} [pix];Counts");
    hSigmaxy->SetLineColor(kRed); hSigmaxy->SetFillColor(kRed); hSigmaxy->SetFillStyle(3001);
    hSigmaxy->Draw("hist"); SaveCanvas(c3, Form("%s/sigmaXY_Fe55", extDir.Data()));
    delete c3;

    TCanvas* c4=new TCanvas(Form("c_sigmax_ext%d",extNum),"sigmaX",700,500);
    hSigmax->SetTitle("#sigma_{x} Distribution;#sigma_{x} [pix];Counts");
    hSigmax->SetLineColor(kRed); hSigmax->SetFillColor(kRed); hSigmax->SetFillStyle(3001);
    hSigmax->Draw("hist"); SaveCanvas(c4, Form("%s/sigmaX_Fe55", extDir.Data()));
    delete c4;

    TCanvas* c5=new TCanvas(Form("c_sigmay_ext%d",extNum),"sigmaY",700,500);
    hSigmay->SetTitle("#sigma_{y} Distribution;#sigma_{y} [pix];Counts");
    hSigmay->SetLineColor(kRed); hSigmay->SetFillColor(kRed); hSigmay->SetFillStyle(3001);
    hSigmay->Draw("hist"); SaveCanvas(c5, Form("%s/sigmaY_Fe55", extDir.Data()));
    delete c5;

    // positions
    TCanvas* c6=new TCanvas(Form("c_posXY_ext%d",extNum),"PosXY",800,700);
    hPosXY->SetTitle("Cluster Position Distribution;X [col];Y [row]");
    hPosXY->Draw("BOXCOLZ"); SaveCanvas(c6, Form("%s/posXY_Fe55", extDir.Data()));
    delete c6;

    TCanvas* c7=new TCanvas(Form("c_posX_ext%d",extNum),"PosX",700,500);
    hPosX->SetTitle("Cluster X Position;X [col];Counts");
    hPosX->SetLineColor(kBlue); hPosX->SetFillColor(kBlue); hPosX->SetFillStyle(3001);
    hPosX->Draw("hist"); SaveCanvas(c7, Form("%s/posX_Fe55", extDir.Data()));
    delete c7;

    TCanvas* c8=new TCanvas(Form("c_posY_ext%d",extNum),"PosY",700,500);
    hPosY->SetTitle("Cluster Y Position;Y [row];Counts");
    hPosY->SetLineColor(kBlue); hPosY->SetFillColor(kBlue); hPosY->SetFillStyle(3001);
    hPosY->Draw("hist"); SaveCanvas(c8, Form("%s/posY_Fe55", extDir.Data()));
    delete c8;
  }

  // Per-extension calibration summary (info tree: gain, single-e resolution, DC)
  {
    TCanvas* cCal = new TCanvas(Form("c_calibration_ext%d", extNum), "Calibration (info tree)", 500, 320);
    cCal->SetGrid(0, 0);
    TPaveText* pt = new TPaveText(0.08, 0.15, 0.92, 0.88, "NDC");
    pt->SetTextAlign(12);
    pt->SetTextSize(0.045);
    pt->SetBorderSize(1);
    pt->SetFillColor(kWhite);
    pt->AddText(Form("Extension %d - %s (from info tree)", extNum, moduleUsed));
    pt->AddText("");
    pt->AddText(Form("Gain:        %g #pm %g  [e^{-}/ADU]", R.gainext, R.gainext_err));
    pt->AddText(Form("Single-e #sigma:  %g #pm %g  [e^{-}]", R.sigext, R.sigext_err));
    pt->AddText(Form("Dark current: %g #pm %g  [e^{-}/pix/s]", R.dcext, R.dcext_err));
    pt->Draw();
    SaveCanvas(cCal, Form("%s/calibration_info", extDir.Data()));
    delete cCal;
  }

  return R;
}

// ---------------------- main entry point ----------------------------
// Reads `filelist` (one file path per line, each tagged "_ext<1-4>"), runs
// ProcessOneExtension per extension, then produces the cross-extension 2x2
// comparison canvases and summary logs under outDir/Fe_55_Analysis_Plots/<moduleUsed>/.
void MakeComposite_Fe55_Clusters_Extensions(const char* filelist="test_list.txt",
                                 const int   image_number=4,
                                 const char* moduleUsed="DMXX",
                                 const char* outDir="out",
                                 bool interactiveAm241 = false)
{
  if (interactiveAm241)
    gROOT->SetBatch(kFALSE);

  // Read list and bucket by extension
  std::ifstream fin(filelist);
  if (!fin){
    std::cerr << "[ERROR] Cannot open file list: " << filelist << "\n";
    return;
  }
  std::map<int, std::vector<TString>> byExt; // 1..4
  std::string line;
  while (std::getline(fin, line)){
    if (line.empty()) continue;
    TString p(line.c_str());
    int ext = ParseExtNum(p);
    if (ext>=1 && ext<=4) byExt[ext].push_back(p);
  }
  fin.close();

  // Output root folder: out/<Module>/<leaf of listfile>
  TString leaf = BaseLeaf(filelist);
  TString baseTag = Form("%s/Fe_55_Analysis_Plots/%s/%s", outDir, moduleUsed, leaf.Data());
  EnsureDir(baseTag);
  EnsureDir(baseTag + "/compare");

  // Process each extension, collect results
  ExtResults R[5]; // indices 1..4 used; R[0] unused
  for (int e=1;e<=4;++e){
    if (byExt.count(e)==0 || byExt[e].empty()){
      std::cout << "[INFO] No files for ext" << e << "\n";
      continue;
    }
    std::cout << "[RUN] Processing ext" << e << " with " << byExt[e].size() << " files\n";
    R[e] = ProcessOneExtension(byExt[e], e, image_number, baseTag, moduleUsed, interactiveAm241);
  }

  // ---------------- Comparison 2x2: common scales -------------------
  // Back heatmap z-range
  double zminB=1e-6, zmaxB=0.0;
  double ymaxProj=0.0;
  for (int e=1;e<=4;++e){
    if (!R[e].hBack) continue;
    zmaxB = std::max(zmaxB, R[e].hBack->GetMaximum());
    if (R[e].projX) ymaxProj = std::max({ymaxProj, R[e].projX->GetMaximum(), R[e].projY->GetMaximum()});
  }
  if (zmaxB<=0) zmaxB=1.0;

  // Front heatmap z-range (min is fixed at 1e-6 inside DrawFrontCompositeHeatmap)
  double zmaxF=0.0;
  for (int e=1;e<=4;++e){
    if (!R[e].hFront) continue;
    zmaxF = std::max(zmaxF, R[e].hFront->GetMaximum());
  }
  if (zmaxF<=0) zmaxF=1.0;

  // Cluster Energy y-range
  double ymaxEnergy=0.0;
  for (int e=1;e<=4;++e){
    if (!R[e].hEnergy) continue;
    ymaxEnergy = std::max(ymaxEnergy, R[e].hEnergy_front->GetMaximum());
  }
  if (ymaxEnergy<=0) ymaxEnergy=1.0;

  // ----------------- (1) Back-side CTI map comparison 2×2 -------------------
  {
    TCanvas* c = new TCanvas("compare_back","Back 2x2", 2800, 2200);
    c->Divide(2,2);
    int pad=1;
    for (int e=1; e<=4; ++e){
      if (!R[e].hBack) { ++pad; continue; }
      c->cd(pad++);
      // two subpads per tile
      TPad* left = new TPad(Form("bL_%d",e),"",0.0,0.0,0.57,1.0); left->Draw();
      TPad* right= new TPad(Form("bR_%d",e),"",0.58,0.0,1.0,1.0); right->Draw();

      left->cd(); left->SetRightMargin(0.16); left->SetLeftMargin(0.12); left->SetBottomMargin(0.12); left->SetLogz();
      TH2D* h = (TH2D*)R[e].hBack->Clone();
      h->GetXaxis()->SetRangeUser(-4,4); h->GetYaxis()->SetRangeUser(-4,4);
      h->SetMinimum(zminB); h->SetMaximum(zmaxB);
      h->GetXaxis()->SetTitle("#Delta x [pix]");
      h->GetYaxis()->SetTitle("#Delta y [pix]");
      h->SetTitle("");
      h->Draw("COLZ");
      { TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
        lab.DrawLatex(0.50,0.965,Form("Back-Side CTI Map (Fe-55 cut) - %s - Ext %d", moduleUsed, e)); }

      right->cd(); right->SetGridx(); right->SetGridy(); right->SetLeftMargin(0.16); right->SetBottomMargin(0.16);
      if (R[e].projX && R[e].projY){
        TH1D* px=(TH1D*)R[e].projX->Clone(); TH1D* py=(TH1D*)R[e].projY->Clone();
        px->GetXaxis()->SetTitle("Pixel offset [pix]");
        px->GetYaxis()->SetTitle("Normalized counts");
        px->GetYaxis()->SetTitleOffset(2.3);
        px->SetMaximum(1.15*ymaxProj); px->SetMinimum(0.0);
        px->SetLineColor(kBlack);
        px->SetLineWidth(3);
        px->SetFillColorAlpha(kBlack, 0.08);
        px->SetMarkerStyle(20);
        px->SetMarkerColor(kBlack);
        py->SetLineColor(kBlue+1);
        py->SetLineWidth(3);
        py->SetFillColorAlpha(kBlue+1, 0.08);
        py->SetMarkerStyle(20);
        py->SetMarkerColor(kBlue+1);

        px->Draw("HIST"); py->Draw("HIST SAME");
        auto leg = new TLegend(0.45,0.8,0.89,0.88);
        leg->SetBorderSize(1);
        leg->SetFillStyle(1001);
        leg->SetTextFont(42);
        leg->SetTextSize(0.038);
        leg->AddEntry(px,"ProjX (Y: -4 #rightarrow -3 px)","l");
        leg->AddEntry(py,"ProjY (X: -4 #rightarrow -3 px)","l");
        leg->Draw();
      }
    }
    SaveCanvas(c, Form("%s/compare/compare_back_CTI_map_Fe55_2x2", baseTag.Data()));
    delete c;
  }


  // ----------------- (2) Front-side CTI map comparison 2×2 ------------------
  {
    TCanvas* c = new TCanvas("compare_front","Front 2x2", 2800, 2200);
    c->Divide(2,2);
    int pad=1;
    for (int e=1; e<=4; ++e){
      if (!R[e].hFront) { ++pad; continue; }
      c->cd(pad++);
      TPad* left = new TPad(Form("fL_%d",e),"",0.0,0.0,0.57,1.0); left->Draw();
      TPad* right= new TPad(Form("fR_%d",e),"",0.57,0.0,1.0,1.0); right->Draw();

      left->cd();
      DrawFrontCompositeHeatmap(R[e].hFront, 0.16, 6, zmaxF,
          Form("Front-Side CTI Map (Fe-55 cut) - %s - Ext %d", moduleUsed, e));

      right->cd();
      DrawFrontRegionBars(Form("frameF_cmp_%d",e), "Center",
                           R[e].fAbove, R[e].fBelow, R[e].fRight, R[e].fLeft, R[e].centerCharge);
    }
    SaveCanvas(c, Form("%s/compare/compare_front_CTI_map_Fe55_2x2", baseTag.Data()));
    delete c;
  }

  // ----------------- (3) Energy comparison 2×2 ----------------------
  {
        TCanvas* c = new TCanvas("compare_energy","Energy 2x2", 1400, 1000);
        c->Divide(2,2);
        int pad=1;
        for (int e=1;e<=4;++e){
        if (!R[e].hEnergy_front){ ++pad; continue; }
        c->cd(pad++); gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);

        TH1F* h=(TH1F*)R[e].hEnergy_front->Clone();

        h->SetTitle(";Energy [keV];Counts");
        h->SetFillColorAlpha(kGray+1,0.60); 
        h->SetLineColor(kGray+2); 
        h->SetBarWidth(0.95); 
        h->SetBarOffset(0.025);
        h->SetMaximum(1.15 * ymaxEnergy);
        h->GetYaxis()->SetNoExponent(true);   // avoid "×10^3" label
        h->GetYaxis()->SetMaxDigits(3);
        h->Draw("BAR");
        auto outline=(TH1F*)h->Clone(Form("hEout_cmp_%d",e)); outline->SetFillStyle(0); outline->SetLineColor(kBlack); outline->SetLineWidth(2); 
        outline->Draw("BAR SAME");

        // draw fitted components using stored params
        TF1* g1=new TF1(Form("cEg1_%d",e),"gaus",R[e].frontKAlpha.mu-4*R[e].frontKAlpha.sigma, R[e].frontKAlpha.mu+4*R[e].frontKAlpha.sigma);
        TF1* g2=new TF1(Form("cEg2_%d",e),"gaus",R[e].frontKBeta.mu-4*R[e].frontKBeta.sigma, R[e].frontKBeta.mu+4*R[e].frontKBeta.sigma);

        // fallback if a fit failed
        const double A1 = (R[e].frontKAlpha.amp>0 ? R[e].frontKAlpha.amp : h->GetMaximum());
        const double A2 = (R[e].frontKBeta.amp>0 ? R[e].frontKBeta.amp : 0.3*h->GetMaximum());

        g1->SetParameters(A1, R[e].frontKAlpha.mu, std::max(0.03, R[e].frontKAlpha.sigma));
        g2->SetParameters(A2, R[e].frontKBeta.mu, std::max(0.03, R[e].frontKBeta.sigma));
        g1->SetNpx(600); g2->SetNpx(600);
        g1->SetLineColor(kRed+1);  g1->SetLineWidth(3);
        g2->SetLineColor(kBlue+1); g2->SetLineWidth(3);

        g1->Draw("SAME");
        g2->Draw("SAME");

        auto leg=new TLegend(0.58,0.63,0.89,0.88);
        leg->AddEntry(g1,Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", R[e].frontKAlpha.mu,R[e].frontKAlpha.mu_err,R[e].frontKAlpha.sigma,R[e].frontKAlpha.sigma_err),"l");
        leg->AddEntry(g2,Form("#splitline{#mu = %.3f #pm %.3f keV}{#sigma = %.3f #pm %.3f keV}", R[e].frontKBeta.mu,R[e].frontKBeta.mu_err,R[e].frontKBeta.sigma,R[e].frontKBeta.sigma_err),"l");
        leg->AddEntry(h,"Data","f"); leg->Draw();

        TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
        lab.DrawLatex(0.50,0.965,Form("Front-side Energy Spectrum - %s - Ext %d", moduleUsed, e));
    }
    SaveCanvas(c, Form("%s/compare/compare_front_energy_spectrum_Fe55fit_2x2", baseTag.Data()));
    delete c;
  }

// ----------------- (4) Energy comparison Front and Back 2×2 ----------------------
  {
    TCanvas* c = new TCanvas("compare_energy_fb","Energy 2x2", 1400, 1000);
    c->Divide(2,2);
    int pad=1;
    for (int e=1;e<=4;++e){
      if (!R[e].hEnergy_front){ ++pad; continue; }
      c->cd(pad++); gPad->SetGridx(); gPad->SetGridy(); gPad->SetLeftMargin(0.12); gPad->SetBottomMargin(0.12);

      // Note: ymaxEnergy (cross-extension front-side max, computed above) is used
      // directly here rather than a per-panel max (unlike the per-extension version
      // of this panel in ProcessOneExtension) -- preserved as found.
      DrawFrontBackEnergyPanel(R[e].hEnergy_front, R[e].hEnergy_back,
                                R[e].frontKAlpha, R[e].frontKBeta, R[e].backKAlpha, R[e].backKBeta,
                                ymaxEnergy, e,
                                Form("Front and Back Energy Spectrum - %s - Ext %d", moduleUsed, e));
    }
    SaveCanvas(c, Form("%s/compare/compare_front_back_energy_spectrum_Fe55fit_2x2", baseTag.Data()));
    delete c;
  }
// ----------------- (5) Energy vs. Sigmaxy comparison 2×2 ------------------
  {
    TCanvas* c = new TCanvas("compare_E_sigmaxy","Energy vs SigmaXY 2x2", 2800, 2200);
    c->Divide(2,2);
    int pad=1;
    for (int e=1; e<=4; ++e){
      if (!R[e].henergy_sigmaxy) { ++pad; continue; }
      c->cd(pad++);  
      R[e].henergy_sigmaxy->SetTitle("");
      R[e].henergy_sigmaxy->GetXaxis()->SetTitle("Energy [keV]");
      R[e].henergy_sigmaxy->GetYaxis()->SetTitle("#sigma_{xy} [pix]");
      R[e].henergy_sigmaxy->Draw("COLZ");
      {TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
      lab.DrawLatex(0.50,0.965,Form("Cluster Energy vs #sigma_{xy} - %s - Ext %d", moduleUsed, e));}
    }
    SaveCanvas(c, Form("%s/compare/compare_energy_vs_sigmaXY_Fe55_2x2", baseTag.Data()));
    delete c;
  }

  // ------------------- (6) Pixel Energy Distribution comparison 2x2 --------------------
  {
    TCanvas* c = new TCanvas("compare_pixelE","PixelE 2x2", 2800, 2200);
    c->Divide(2,2);
    gPad->SetLogy(true);
    gPad->SetGridx();
    int pad=1;
    for (int e=1; e<=4; ++e){
      if (!R[e].hPixelEnergy) { ++pad; continue; }
      c->cd(pad++);
      gPad->SetGridx();

      R[e].hPixelEnergy->SetTitle("");
      R[e].hPixelEnergy->GetXaxis()->SetTitle("Electron number [e-]");
      R[e].hPixelEnergy->GetYaxis()->SetTitle("Counts");
      R[e].hPixelEnergy->SetLineColor(kBlue);
      R[e].hPixelEnergy->GetXaxis()->SetRangeUser(min_e_number_display,max_e_number_display);
      R[e].hPixelEnergy->Draw("Hist");
      {TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
      lab.DrawLatex(0.50,0.965,Form("Pixel Energy Distribution - %s - Ext %d", moduleUsed, e));}
    }
    SaveCanvas(c, Form("%s/compare/compare_pixel_energy_allclusters_2x2", baseTag.Data()));
    delete c;
  }


  // ------------------- (7) Calibration (info tree): Gain, single-e resolution, DC per extension ---
  {
    TCanvas* c = new TCanvas("compare_calibration", "Calibration per extension", 1400, 1000);
    c->Divide(1, 3);

    c->cd(1);
    DrawCalibrationGraph(R, &ExtResults::gainext, &ExtResults::gainext_err, kBlue+1,
                          "Gain [e^{-}/ADU]", Form("Gain (info tree) - %s", moduleUsed),
                          "Extension (1-4)");

    c->cd(2);
    DrawCalibrationGraph(R, &ExtResults::sigext, &ExtResults::sigext_err, kRed+1,
                          "Single-e resolution #sigma [e^{-}]",
                          Form("Single-electron resolution (info tree) - %s", moduleUsed));

    c->cd(3);
    DrawCalibrationGraph(R, &ExtResults::dcext, &ExtResults::dcext_err, kGreen+2,
                          "Dark current [e^{-}/pix/s]",
                          Form("Dark current (info tree) - %s", moduleUsed));

    SaveCanvas(c, Form("%s/compare/compare_calibration", baseTag.Data()));
    delete c;
  }

  // ------------------- (8) Am241: energy distribution comparison 2x2 --------------------
  {
    TCanvas* c = new TCanvas("compare_am241","Am241 2x2", 2800, 2200);
    c->Divide(2,2);
    int pad=1;
    for (int e=1; e<=4; ++e){
      if (!R[e].hEnergy) { ++pad; continue; }
      c->cd(pad++);
      R[e].hEnergy->SetTitle("");
      R[e].hEnergy->GetXaxis()->SetTitle("Energy [keV]");
      R[e].hEnergy->GetYaxis()->SetTitle("Counts");
      R[e].hEnergy->Draw("COLZ");
      {TLatex lab; lab.SetNDC(); lab.SetTextAlign(22); lab.SetTextSize(0.05);
      lab.DrawLatex(0.50,0.965,Form("Am241 Energy Distribution - %s - Ext %d", moduleUsed, e));}
    }
    SaveCanvas(c, Form("%s/compare/compare_am241_energy_spectrum_2x2", baseTag.Data()));
    delete c;
  }
  // -------------------- Summary log Energy --------------------
  std::ofstream energy_log(baseTag + "/energy_summary.csv");
  energy_log << "Extension,location,energypeak1,energypeak2\n";
  for (int e=1; e<=4; ++e){
    if (R[e].nFiles==0) continue;
    energy_log 
      << e << ","
      << 0 << ","
      << R[e].frontKAlpha.mu << " +/- " << R[e].frontKAlpha.mu_err << ","
      << R[e].frontKBeta.mu << " +/- " << R[e].frontKBeta.mu_err << "\n";
    energy_log 
      << e << ","
      << 1 << ","
      << R[e].backKAlpha.mu << " +/- " << R[e].backKAlpha.mu_err << ","
      << R[e].backKBeta.mu << " +/- " << R[e].backKBeta.mu_err << "\n";
  }
  energy_log.close();

  std::ofstream CTI_log(baseTag + "/CTI_summary.csv");
  CTI_log << "Extension,location,fraction,Mean,RMS,Skewness,Integral\n";
  for (int e=1; e<=4; ++e){
    if (R[e].nFiles==0) continue;
    CTI_log 
      << e << ","
      << 0 << ","
      << R[e].fRight << ","
      << R[e].projX_mean << ","
      << R[e].projX_rms << ","
      << R[e].projX_skew << ","
      << R[e].projX_int_e << "\n";
    CTI_log 
      << e << ","
      << 1 << ","
      << R[e].fAbove << ","
      << R[e].projY_mean << ","
      << R[e].projY_rms << ","
      << R[e].projY_skew << ","
      << R[e].projY_int_e << "\n";
  }
  CTI_log.close();

  // -------------------- main log --------------------
  std::ofstream main_log(baseTag + "/summary.log");
  main_log << "Summary for " << baseTag << "  (Module: " << moduleUsed << ")\n";
  main_log << "================================================================\n";
  for (int e=1; e<=4; ++e){
    if (R[e].nFiles==0) continue;
    main_log << "----------------------------------------------------------------\n";
    main_log << "Extension: ext" << e << "\n";
    main_log << "Files combined: " << R[e].nFiles << "\n";
    main_log << "Info tree (calibration): gain=" << R[e].gainext << " +/- " << R[e].gainext_err
        << "  single-e sigma=" << R[e].sigext << " +/- " << R[e].sigext_err
        << "  DC=" << R[e].dcext << " +/- " << R[e].dcext_err << "\n";
    main_log << "Total clusters: " << R[e].totalClusters << "\n";
    main_log << "Accepted (energy window): " << R[e].nAcceptedEnergy << "\n";
    main_log << "Accepted (front-side):    " << R[e].nAcceptedFront  << "\n";
    main_log << "Accepted (back-side):     " << R[e].nAcceptedBack   << "\n";
    main_log << "Front integral [e-]: " << R[e].frontIntegral_e << "\n";
    main_log << "Back  integral [e-]: " << R[e].backIntegral_e  << "\n";
    main_log << "CTI fractions for front events (w.r.t. center)\n";
    main_log << "  center charge: " << R[e].centerCharge << "\n";
    main_log << "  above=" << R[e].fAbove
        << " below=" << R[e].fBelow
        << " right=" << R[e].fRight
        << " left="  << R[e].fLeft  << "\n";
    main_log << "Back-side projections (slabs at [-4,-3] px)\n";
    main_log << "  ProjY (X slab) integral [e-]: " << R[e].projY_int_e
        << " | mean=" << R[e].projY_mean
        << " RMS="    << R[e].projY_rms
        << " skew="   << R[e].projY_skew << "\n";
    main_log << "  ProjX (Y slab) integral [e-]: " << R[e].projX_int_e
        << " | mean=" << R[e].projX_mean
        << " RMS="    << R[e].projX_rms
        << " skew="   << R[e].projX_skew << "\n";
    main_log << "Front-side energy fit (two separate Gaussians in per-peak windows)\n";
    main_log << "  Kalpha mu=" << R[e].frontKAlpha.mu << " +/- " << R[e].frontKAlpha.mu_err
        << "  sigma=" << R[e].frontKAlpha.sigma << " +/- " << R[e].frontKAlpha.sigma_err << "\n";
    main_log << "  Kbeta  mu=" << R[e].frontKBeta.mu << " +/- " << R[e].frontKBeta.mu_err
        << "  sigma=" << R[e].frontKBeta.sigma << " +/- " << R[e].frontKBeta.sigma_err << "\n";
    main_log << "  chi2/ndf=" << R[e].frontChi2 << "/" << R[e].frontNdf
        << "  p=" << R[e].frontFitProb << "\n";
  }
  main_log.close();

  std::cout << "[DONE] Outputs under: " << baseTag << "\n";

  if (interactiveAm241) {
    // Do not call TApplication::Run() here: interactive ROOT (TRint) is already inside Run();
    // nesting Run() re-enters recursively and blows the stack. Pump events once so canvases refresh.
    gSystem->ProcessEvents();
    std::cout << "[INTERACTIVE] Am241 canvases (cAm241_ext*) left open; PNGs saved. "
                 "Interact with the plot windows, then type .q to quit ROOT.\n";
  }
}
