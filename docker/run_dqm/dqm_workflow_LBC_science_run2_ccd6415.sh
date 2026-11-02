#!/bin/bash
# use script
# source dqm_workflow_LBC_science.sh $run $nstart $nend 
# 


#   SCRIPT TO RUN THE DQM ON THE SCIENCE-RUN-1

# FOR THE SCIENCE RUN-1 THE IMAGE NUMBER COINCIDE WITH THE ID OF THE IMAGE FILE
# run starts at number 1, and contain a bunch of 100 consecutive (on-time) image

export TAG='science-2'
export CCDNAME='6415'
export OFFSET=151089

# for some unknown reasson the venv does not set this variable properly
source /pbs/throng/damic/software/waders/pysimdamicm/venv_waders/bin/activate
export PYTHONPATH=/pbs/throng/damic/software/waders/pysimdamicm/venv_waders/lib/python3.6/site-packages:$PYTHONPATH

echo "Running DQM version from "
echo `which dqmSKImg`
export DATA_DIR=/sps/damic/LBCdata/raw/run/2022-05-10-science-ccd6415/
export OUT_DIR=/sps/damic/LBCdata/processed/science-run2/ccd${CCDNAME}

export MEJSON=/sps/damic/ncastell/devDamicm/work/LBC/run_dqm/science-run2/dqmSKImg_configuration.json
export JSON=/sps/damic/ncastell/devDamicm/work/LBC/run_dqm/science-run2/panaSKImg_configuration_6415.json

export run=$1
export nstart=$2
export nend=$3

echo "Reprocessing CCD-${CCDNAME}: run ${run}, with IDs ${nstart}:${nend} for run tag ${TAG} (id offset ${OFFSET})"

export zrun=`printf  "%03d" ${run}`

# Create working directory
mkdir -p ${OUT_DIR}
mkdir -p  /data/waders_reports/roots

for nfile in `seq $nstart $nend`;
do 
    IDIMAGE=$((nfile-OFFSET))

    dqmSKImg -j ${JSON} --me-json ${MEJSON} --dqm-data-dir ${DATA_DIR} --image ${IDIMAGE} --ccd ${CCDNAME} --tag ${TAG} --run ${run} --skip clean -o ${OUT_DIR} skip_2022*_*_${nfile}.fits
    
    mv ${OUT_DIR}/run${zrun}/avgimg/panaSKImg_clustersRec_skip_*_*_${nfile}.root  ${OUT_DIR}/run${zrun}/recon
    mv ${OUT_DIR}/run${zrun}/recon/panaSKImg_clustersRec_skip_*_*_${nfile}.root /data/waders_reports/roots

done

# move outputs to directories
echo "- Move PDF to waders_reports "
mv ${OUT_DIR}/run*/me/report/*pdf /data/waders_reports/

export runID=`printf  "%07d" ${run}`

# move dark current fits
echo "- Transfer waders reports  (png) to waders_reports/dark_current_fit "
mkdir -p /data/waders_reports/dark_current_fit/run${runID}
mv ${OUT_DIR}/run*/me/dcfit/*png /data/waders_reports/dark_current_fit/run${runID}


# move pkl to the data base npz directory
echo "- Transfer PKL files to waders_npz/run${runID}"
mkdir -p /data/waders_npz/run${runID}
mv ${OUT_DIR}/run*/me/mongoDB_document*.pkl   /data/waders_npz/run${runID}


# create the npz document for the given run
echo "- Create the full mongoDB document ... "

mkdir -p pkls
cp /data/waders_npz/run${runID}/mongoDB_document*.pkl  pkls/
create_mongDB_document  'pkls/mongoDB_document_*pkl' -o . --run ${run}

rm -rf pkls

mv mongoDB_document*.npz /data/waders_npz

echo "done!"

