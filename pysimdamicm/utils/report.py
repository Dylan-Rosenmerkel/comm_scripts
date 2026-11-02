import os
import subprocess

class LaTex(object):
    
    def __init__(self,doc_title=None,doc_author=None,**kwargs):
        """
        """
        self.n_figures = 0

        self.preamble = [
                r"\documentclass{article}",
                r"\usepackage[marginparwidth=6cm, marginparsep=0.7cm]{geometry}",
                r"\newgeometry{top=20mm,bottom=20mm,right=10mm,left=10mm}",
                "\n"
                r"\usepackage[utf8]{inputenc}",
                "\n",
                r"\author{doc_author}".replace('doc_author',doc_author),
                r"\title{doc_title}".replace('doc_title',doc_title),
                "\n",
                r"\usepackage{natbib}",
                r"\usepackage{rotating}",
                r"\usepackage{graphicx}",
                "\n",
                r"\usepackage{lscape}",
                ]

        if doc_title is None:
            self.preamble.pop(3)
        if doc_author is None:
            self.preamble.pop(2)
        
        ### a list of figures 
        self.figures = []
        ### 
        self.abstract = None

    def add_figure(self,fig_name,fig_caption,fig_ref=None,fig_scale=1.0,short_caption=''):
        """
        """
        
        self.n_figures +=1
        
        includefig = [
                r"\begin{figure}",
                r"\begin{center}",
                r"\includegraphics[width=fig_scale\linewidth]{fig_name}".replace('fig_name',fig_name).replace('fig_scale',str(fig_scale)),
                r"\caption[short_caption]{fig_caption}".replace('fig_caption',fig_caption).replace('short_caption',short_caption),
                r"    \label{fig_ref}".replace('fig_ref', "Fig:{}".format(fig_ref)),
                r"\end{center}",
                r"\end{figure}"
                ]
        if fig_ref is None:
            includefig.pop(4)
        
        if self.n_figures % 10 == 0:
            includefig.append(r"\clearpage")
        
        self.figures.extend(includefig)

    def build_pdf(self,outtexfile):
        """Compile the latex source and write it down to a pdf file

        If latex source is not created, this will call build_tex
        """
        
        if not hasattr(self,'main_latex'):
            self.build_tex(outtexfile)
        
        ### outdirectory
        outdir = os.path.split(outtexfile)[0]
        outdir = '.' if outdir=="" else outdir

        ### build a pdf with the latex source
        proc = subprocess.Popen(['pdflatex', '-output-directory',outdir,outtexfile],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE).communicate()
        proc = subprocess.Popen(['pdflatex', '-output-directory',outdir,outtexfile],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE).communicate()

        print(" PDF has been created: ", outtexfile.replace(".tex",".pdf"))
        return

    def build_tex(self,outtexfile,add_abstract=True,add_list_figs=True):
        """Build the latex source and write it down to a tex file
        """

        self.main_latex = self.preamble
        self.main_latex.extend([
                r"\begin{document}",
                r"\maketitle",
                "\n",
                "\n"
                ])


        if add_abstract:
            self.add_abstract()
        
        if add_list_figs:
            self.main_latex.extend([r"\listoffigures","\n"])
            self.main_latex.append(r"\newpage")
       
        ### append to the main block
        self.main_latex.extend(self.figures)
        
        # add ending document
        self.main_latex.extend([
            "\n",
            r"\end{document}",
            "\n"
            ])

        ### build the latex source and save as a tex file
        # make sure the output file name is *.tex
        ftexfile = open("{}.tex".format(outtexfile.split(".")[0]),"w+")
        ftexfile.writelines("\n".join(self.main_latex))
        ftexfile.close()

        print(" A LaTex file has been created: ", outtexfile)


    def add_figure_list(self):
        """Add a list of all figures at the beginning of the report
        """
        fitems = [
                r"List of figures: "
                r"\begin{enumerate}"
                ]
        for i,fig in enumerate(self.figure_list):
            fitems.append(r"\item "+fig)
        fitems.append(r"\end{enumerate}")
        fitems.append("\n")
        self.main_latex.extend(fitems)
    
    def add_abstract(self):
        if self.abstract is None:
            return
        self.main_latex.extend(self.abstract)


