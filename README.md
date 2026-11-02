# WADERS

softWare Analysis for Dark matter ExpeRiments  of Sipper images

# How to install

git clone https://gitlab.in2p3.fr/damicm/pysimdamicm.git

# Branches

There are two main Branches:
* [master] **waders**: the master one, the one with the most stable version (that will be periodically updated, now is linked to version 5.0.0] 

* [development] **waders_dev**: this is main to be only for developers where bugs found in waders should be corrected, tested and if all tests have been passed successfully synchronize then waders will be updated


**Deprecated branches will be removed on September.**

* [deprecated] **extend_to_analysis**: branch mostly containning the major changes for analysis and simulations, but deprecated for the DQM softWare

* [deprecated] **DQM**: branch mostly containning the major improvements for the DQM softWare, but not for the rest. 

also **add_qtest** and **master**

## Simulations and Analysis: use the most stable version 'waders' branch 
This branch contains all changes up to date for simulations, but is always behing branch 'extend_to_analysis' as this one change frequencly along with Compton analysis. However, periodically are merged after several tests and checks are done.

```bash
cd <path_to_>/pysimdamicm
cd pysimdamicm
git checkout waders_LBC
pip3  install -r requirements.txt --user
python3 setup.py install --user
```

## Simulations and Analysis: Installation via venv
This branch contains all changes up to date for simulations, but is always behing branch 'extend_to_analysis' as this one change frequencly along with Compton analysis. However, periodically are merged after several tests and checks are done.

```bash
cd <path_to_>/pysimdamicm
git checkout waders_LBC
python3 -m venv venv_waders
source <path_to_>/pysimdamicm/venv_waders/bin/activate
pip3  install -r requirements.txt
python3 setup.py install
```

# pysimdamicm documentation

There are several places where you can find information about that:


For a more detailed information on simulations, find documentation at [code documentation](http://ncastell.web.cern.ch/ncastell/)
Information on analysis maybe not up to date.

For analysis (panaSKImg), find detailed and more techinical notes at: [compton web](https://gev.uchicago.edu/compton/)

Look into the software school: [1st software school](https://indico.in2p3.fr/event/22866/timetable/?view=nicecompact)


# Bugs? Issues?
Write Nuria Castello-Mor on slack, or open an Issues, if not just send me an email d5nunu@gmail.com
