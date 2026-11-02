"""
===========================================================
Simulation and Data PROCESS (:mod: `pysimdamicm.processes`)
===========================================================
.. module:: pysimdamicm.processes

This module contains several types of processes for simulation, comissioning and analysis

.. module:: pysimdamicm.processes
"""

__all__ = ['detector_response','reconstruction','skipper_analysis','skipper_comissioning']

from . import detector_response
from . import reconstruction
from . import absp
from . import skipper_analysis
from . import skipper_comissioning

