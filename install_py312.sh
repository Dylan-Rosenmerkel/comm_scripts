#!/bin/bash

#setwaders
source /opt/root_daf/root_install/bin/thisroot.sh
source /home/ncastell/repos/pysimdamicm/venv_waders/bin/activate

python -m pip install .

cp /home/ncastell/repos/pysimdamicm/pysimdamicm/json/*json /home/ncastell/repos/pysimdamicm/venv_waders/lib/python3.10/site-packages/pysimdamicm/json
cp /home/ncastell/repos/pysimdamicm/pysimdamicm/data/p100* /home/ncastell/repos/pysimdamicm/venv_waders/lib/python3.10/site-packages/pysimdamicm/data

