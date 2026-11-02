#!/usr/bin/env python
""":mod:`ttree` -- Useful functions for pyROOT
==============================================


.. module:: pysimdamicm.utils
    :platform: Unix
    :synopsis: Module gathering a bunch of useful analysis-related functions


.. moduleauthor:: Nuria Castello-Mor <castello@ifca.unican.es> 
"""


from pysimdamicm.utils.root_plot_styles import get_sifca_style

import array
import pandas as pd
import numpy as np

import ROOT
ROOT_VERSION = int(ROOT.gROOT.GetVersion().split('/')[0].replace('.',''))

def prepare_tree_reading(tree):
     
    if ROOT_VERSION<622:
        return tree
    _df = {}
    
    for b in tree.GetListOfBranches():
        bname = b.GetName()
        try:
            if b.GetTypeName().count('vector')==1:
                dtype = 'ROOT.std.{}()'.format(b.GetTypeName().replace('<',"('").replace('>',"')"))
                _df[bname] = eval(dtype)
            elif b.GetTypeName().count('vector')==2:
                if b.GetTypeName().count('float')>0:
                    dtype = "ROOT.std.vector(ROOT.std.vector('float') )()"
                else:
                    dtype = "ROOT.std.vector(ROOT.std.vector('int'))()"
                #dtype = 'ROOT.std.vector(ROOT.std.{})()'.format(b.GetTypeName().replace('<',"('").replace('>',"')"))
                _df[bname] = eval(dtype)

        except AttributeError:
            _df[bname] = array.array( b.GetLeaf(bname).GetTypeName().lower()[0],[0])
        tree.SetBranchAddress(bname,_df[bname])
        
    return _df    


def convert_into_array(bdict):

    if ROOT_VERSION<622:
        _pd = convert_into_pandas_oldROOT(bdict)
        return _pd
    
    _pd = {}
    for k,v in bdict.items():
        try:
            _pd[k] = np.ndarray((len(v),), buffer=np.array(v.data()).astype(float))
        except AttributeError:
            _pd[k] = [v[0]]
        except TypeError:
            _val = []
            dtype = float
            for i in range(len(v)):
                if k in ['pixels_x', 'pixels_y']:
                    _val.append( np.ndarray( (len(v[i]),), buffer=np.array(v[i].data()).astype(dtype)).astype(int))
                else:
                    _val.append( np.ndarray( (len(v[i]),), buffer=np.array(v[i].data()).astype(dtype)))

            _pd[k] = _val

    return _pd


class Bunch(object):
    """Just to load a ROOT file and dump all its content as
    globals (similarly to a CLING root session)

    Example
    -------
    After the initialization, the objects of the ROOT file
    can be accessed as data members of the instance

    >>> d = Bunch("rootfilename")
    """
    def __init__(self,fname):
        """Return a bunched ROOT file with all the contents
        of the ROOT file as data members of the instance

        Parameters
        ----------
        fname: str
            The root file
        """
        st = get_sifca_style(squared=True,stat_off=True)
        st.cd()
        ROOT.gROOT.ForceStyle()
        if isinstance(fname,str):
            self._f = ROOT.TFile.Open(fname)
        else:
            self._f = fname
        for i in self._f.GetListOfKeys():
            try:
                self.__dict__[i.GetName()] = self._f.Get(i.GetName())
            except TypeError:
                continue
            if isinstance(self.__dict__[i.GetName()],ROOT.TDirectoryFile):
                self.__dict__[i.GetName()] = Bunch(self._f.Get(i.GetName()))

