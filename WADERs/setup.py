#!/usr/bin/env python

from os import getcwd
import fileinput

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

__version__ = "8.1.0"
cwd = getcwd()

def replaceAll(ifile,ExpList):
    with open(ifile) as f:
        lines = f.readlines()
    
    for searchExp,replaceExp in ExpList:
        newlines = lines
        for k,line in enumerate(lines):
            if searchExp in line:
                newlines[k] = replaceExp+"\n"

    ofile = ifile.replace('.in','')
    with open(ofile,"w") as nf:
        nf.writelines(newlines)

def get_commit_version():
    """Append to the version number, version, the latest git commit 
    """
    import subprocess
    git_version_command = "git log --pretty=format:'%h' -n 1"

    tag = subprocess.run(git_version_command, shell=True, check=True, stdout=subprocess.PIPE,
            universal_newlines=True, cwd=cwd)

    if tag.returncode==0:
        commit = "log{}".format(tag.stdout.strip())
    else:
        commit = "log0000000"    
    return commit

def update_version():
    """Append commit version to the main __init__ file
    """
    
    init_file="{}/{}".format(cwd,"pysimdamicm/__init__.py.in")
    
    commit = "__commit__ = '{}'".format(get_commit_version())
    version = "__version__ = '{}'".format(__version__)

    print(" Commit GIT log: ", commit)
    print(" Version number: ", __version__)
    
    replaceAll(init_file, [("__commit__",commit),("__version__",version)])
    
    return __version__

###################################################################        
setup(
        name='pysimdamicm',
        version=update_version(),
        description='Python Module to reconstruct and analyze [skipper]-CCD data (both raw and simulated)',
        author='Nuria Castello-Mor',
        author_email='nuria.castello.mor@cern.ch',
        package_data=
        {
            'json': ['*.json'],
            'data': ['*.npz']
            },
        setup_requires=['wheel'],
        packages=
        [
            "pysimdamicm",
            "pysimdamicm.activity",
            "pysimdamicm.data",
            "pysimdamicm.utils",
            "pysimdamicm.io",
            "pysimdamicm.processes",
            "pysimdamicm.dqm",
            "pysimdamicm.setups",
            "pysimdamicm.scripts",
            "pysimdamicm.comm",
            "pysimdamicm.json"
        ],
        include_package_data=True,
        zip_safe=False,
        scripts=
        [
            "bin/psimulCCDimg",
            "bin/panaSKImg",
            "bin/compareMEs",
            "bin/dqmSKImg",
            "bin/drawImageGrid",
            "bin/plot_skip_transient",
            "bin/get_compose_fits_file",
            "bin/dc_analysis",
            "bin/pcd_plot",
            "bin/create_settings_for_ccddrone",
            "bin/cluster_topology",
            "bin/plot_fitsval",
            "bin/plot_parinheader",
            "bin/dqm_moskita",
            "bin/diffheader",
            "bin/find_hot_regions",
            "bin/analysis_hotcolumns_f",
            "bin/get_calibration_constant",
            "bin/cut_fits_image",
            "bin/crosstalk_cli",
            "bin/compare_root_files",
            "bin/gain_vs_charge",
            "bin/run_WADERs"
            ]
        )
