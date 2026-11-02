#!/usr/bin/env python3

import os
from glob import glob
import shutil
import subprocess
import schedule
import time

import logging

CCDNAME='6415'

# previous runs are from science-run-1
OFFSET_RUNID = 6
OFFSET_IDIMAGE = 151089

# FILE TO KEEP CONTROL THE ALREADY PROCESSED IMAGES
REPROCESSED_FITS_FILES='/sps/damic/ncastell/devDamicm/work/LBC/run_dqm/science-run2/reprocessed_file_list_{}.log'.format(CCDNAME)

# raw data location at cca.in2p3.fr for SCIENCE RUN 2  -- CCD ID 6415
DATA_DIR='/data/fits/2022-05-10-science-ccd{}'.format(CCDNAME)

# pattern file name for the images in SCIENCE RUN 2 -- CCD ID 6415
FILE_PATTERN='skip_2022*_*_{}.fits'

# SCRIPPT TO RUN dqm_workflow_LBC_science.sh $run $nstart $nend

def get_run_number_id(image_id):
    """For the science run 2, last run id was 5, so for this run we start with run id 6
    And every 100 images we will change the run id
    """
    # for science run 1
    #run_id = int(image_id)//100 - 3

    run_id = ((image_id-OFFSET_IDIMAGE)//100)+OFFSET_RUNID
    
    return str(run_id)

def do_run():
    """Run DQM over a new raw rata to generate the pkl needed to append the fits file image to the
    mongo DB

    """

    # check if new data files are available
    #  the file REPROCESSED_FITS_FILES contains (in a column format) the image id of all reprocessed
    #  fits files
    # the latest entry corresponds to the lates `nfile` reprocessed (nfile from the dqm_workflow_LBC_science.sh script)
    f = open(REPROCESSED_FITS_FILES,"r")
    nfile = int(f.readlines()[-1].strip())
    f.close()
    
    # next image to be reprocessed is
    next_image_id = nfile+1
    
    # OUTPUT MESSAGES
    log_msm = " Look for run number: {}\n".format(next_image_id)
    log_msm += " Reprocessing run {} ... \n".format(next_image_id)

    # read files in the  RAWDATA folder, and check if there is a newest image
    next_file = glob('{}/{}'.format(DATA_DIR,FILE_PATTERN.format(next_image_id)))

    if not len(next_file)==1:
        return

    # NEW IMAGE ID HAS BEEN FOUND    
    next_file = next_file[0].split('/')[-1]

    log_file = open("logs/process_next_image_{}.txt".format(next_image_id),'a+')
    log_file.writelines(log_msm)
    log_file.writelines(" fits file: {} with id {} will be included with run number {} \n".format(next_file,next_image_id,get_run_number_id(next_image_id)))

    # write the ID of the next_image to the REPROCESSED_FITS_FILES
    f = open(REPROCESSED_FITS_FILES,"a+")
    f.writelines("{}\n".format(next_image_id))
    f.close()
    log_file.writelines(" image id {} added to {} \n".format(next_image_id,REPROCESSED_FITS_FILES))
    
    # new image is available, so run the DQM on the new image
    proc_log_file_name = 'logs/reprocessing_{}_nfile_{}.log'.format(next_file.split('.')[0],next_image_id)
    proc_log_file = open(proc_log_file_name,"a+")
    
    log_file.writelines(" starting DQM \n")
    proc = run_dqm(next_image_id,proc_log_file)
    proc_log_file.close()
    log_file.writelines(" DQM finished. \n")
    log_file.close()

    return


def run_dqm(next_image_id, logfile):
    """RUN THE SCRIPT dqm_workflow_LBC_science_run2_ccd6415.sh which runs dqmSKImg for a given image number
    """
    
    run_number = get_run_number_id(next_image_id)

    bash_command_line = [ './dqm_workflow_LBC_science_run2_ccd6415.sh',
            run_number,
            str(next_image_id),
            str(next_image_id)
            ]
    print("starting process ... ")    
    proc = subprocess.check_call([" ".join(bash_command_line)], stdout=logfile,stderr=logfile, shell=True )
    print("finishing process ....")
    return proc


###############################################################################
###############################################################################
# registering the job
#schedule.every(1).hours.do(job,'do_job')
#schedule.every(1).minutes.do(do_run)

schedule.every(3).seconds.do(do_run)
while True:
    schedule.run_pending()
    # in seconds
    time.sleep(1)

# do_run()

    
