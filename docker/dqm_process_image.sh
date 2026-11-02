#!/bin/bash
# Launch the processing of the image dqmSKImg
# 
# $0  $1   $2    $3    $4         $5         $6      $7
#    TAG  RUN  CCDID  IMAGEID  FILENAME    MEJSON   JSON
TAG=$1
RUN=$2
CCDID=$3
IMAGEID=$4
FILENAME=$5
MEJSON=$6
JSON=$7
DATA_DIR="/data/fits"

# XXX - Cross-check options?? Probably not needed

# Running 
echo "STARTING TIME     : `date`"
echo "WADERS CONTAINER  : ${HOSTNAME}"
echo "WORKING DIRECTORY : ${PWD}"
echo "LINKED VOLUMES [1]: /data/waders_npz"
echo "LINKED VOLUMES [2]: /data/waders_reports"
echo "LINKED VOLUMES [3]: /data/tmp_transfers"
echo "WADERS EXECUTABLE : `which dqmSKImg`"
echo "RUNNING PARAMETERS: "
echo "  -TAG     : {${TAG}]"
echo "  -RUN     : [${RUN}]"
echo "  -CCDID   : [${CCDID}]"
echo "  -IMAGEID : [${IMAGEID}]"
echo "  -FILENAME: [${FILENAME}]"
echo "  -MEJSON  : [${MEJSON}]"
echo "  -JSON    : [${JSON}]"
echo "  -DATA_DIR: [${DATA_DIR}]"

echo
echo "STARTING RUNNING  : `date`"
dqmSKImg -j ${JSON} --me-json ${MEJSON} --dqm-data-dir ${DATA_DIR} --image ${IMAGEID} --ccd ${CCDID} --tag ${TAG} --run ${RUN} --skip clean -o . ${FILENAME}
echo "END RUNNING       : `date`"

echo
echo "Create ROOT folder for transfer and copy the outputs *.root"
mkdir -p /data/tmp_transfers/roots
mv run*/avgimg/panaSKImg_clustersRec_skip_*root /data/tmp_transfers/roots

echo
echo "Create FITS folder for transfer and copy the outputs *.fits"
mkdir -p /data/tmp_transfers/procfits
mv run*/others/skip_*_waders.fits /data/tmp_transfers/procfits

echo 
runID=`printf  "%07d" ${run}`
echo "Create '/data/waders_reports/dark_current_fit/run${runID}' and transfer the dark current png fits"
mkdir -p /data/waders_reports/dark_current_fit/run${runID}
mv run*/me/dcfit/*png /data/waders_reports/dark_current_fit/run${runID}

echo 
echo "Create '/data/waders_npz/run${runID}' (if necessary) and move pkl file"
# move pkl to the data base npz directory
mkdir -p /data/waders_npz/run${runID}
mv run*/me/mongoDB_document_skip_*_*_${nfile}_run*.pkl /data/waders_npz/run${runID}

echo
echo "JOB FINISHED      : " `date`

