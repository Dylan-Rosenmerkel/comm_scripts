.. _howotinstall:

================================
Requirements and/or dependences
================================

This package requires several python modules:
    * `NumPy=>1.17.3 <https://numpy.org/>`_: fundamental package for scientific computing
      with Python

    * `SciPy>=1.0.1 <https://www.scipy.org/>`_: python-based ecosystem for mathematics,
      science and engineering

    * `root_numpy <http://scikit-hep.org/root_numpy/>`_: module that provides an efficient interface between `ROOT <https://root.cern.ch/>`_ and `NumPy  <https://numpy.org/>`_

    * `matplotlib <https://matplotlib.org/>`_: python 2D plotting library (needed for the debuging mode of psimulCCDimg)

Check if these packages are already installed (with the required version) for python3. You can use
**pip3** to list all installed packages and theri pipe this to **grep** to find the row
for the particular packages. For instance, 

    .. code-block:: shell
       :linenos:

       $ pip freeze |grep numpy
       numpy==1.17.3
       numpydoc==0.9.1
       root-numpy==4.8.0
    

If the version is smaller, install the required package(s). For instance,

    .. code-block:: bash
       :linenos:

       $ pip3 install numpy>=1.17.3


.. note::

   The package `root_numpy` must be install against the same ROOT version 

=========
Download
=========


Download the package via Git:

.. code-block:: bash
   :linenos:

   $ cd <path_to_your_local_repo_dir>
   $ git clone https://gitlab.in2p3.fr/damicm/pysimdamicm.git
   

========
Install
========

The package provides a (Distutils) `setup.py` to build and install it. Just

.. code-block:: bash
   :linenos:

   $ python3 setup.py install --user


The **--user** option is used when yo do not have root privilegies (or you 
do not want to install the package in the global sitepackages directories). The package
will be installed inside of hte user director :code:`$HOME/.local`. You have to modigy the
enviroment variables

.. code-block:: bash
   :linenos:

   % export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib
   % export PATH=${PATH}:${HOME}/.local/bin

in order to use the new scripts and modules.



