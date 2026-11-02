#!/usr/bin/env python3

import sys

if not '-h' in sys.argv and not '--help'in sys.argv: 
    import os
    from glob import glob
    import shutil
    import subprocess
    import time

    import logging
    
    from astropy.io import fits

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def do_run(DQM_DATA_DIR, DQM_OUT_DIR, JSON_ME, JSON_RECON , run ):
    """ Run a list of process on shell using subprocess.check_call
    """
    
    log_msm = " Look for run number: {}\n".format(run)
    log_msm += "Reprocessing run {} ... \n".format(run)
    logging.debug("Look for files in run {}".format(run))

    lof = glob("{}/Image_comm_*fits".format(DQM_DATA_DIR))

    for infile in lof:
        print("---- processing ", infile)
        #### IF NUMBER OF COLS < PIXELS IN REGISTER, OVERSCAN PIXELS CAN NOT BE DETERMINED BY THE HEADER --- KILL AUTOMATIC REPROCESSING
        header = fits.getheader(infile)
        cols = header['NAXIS1']/float(header['NDCMS']) * header['NSBIN']
        
        if int(cols)<(6144+2): #or int(header['VCKDIRN'])==1:
            continue

        _fname = infile.split("/")[-1]
        # new run is ready
        ###############################################################################################
        ### output log file
        log_file_name = "{}/panaSKImg_processing_run{:03}_{}.log".format(os.getcwd(),int(run),_fname.replace(".fits",""))
        log_file = open(log_file_name,"a+")

        # run DQM for the new run
        proc = run_dqm(run,_fname,DQM_DATA_DIR,JSON_ME,JSON_RECON,DQM_OUT_DIR,logfile=log_file)

        ## run RECON once DQM finished
        #_tf.writelines("    Running Reconstruction \n")
        #logging.debug("Running Reconstruction over run {}".format(new_run))
        #if is_source:
        #    proc = run_recon(new_run,logfile=log_file,isSource="Source")
        #else:
        #    proc = run_recon(new_run,logfile=log_file,isSource="Bkg")

        # close log file
        log_file.close()
        # move log file to the correct directory
        _ = shutil.move(log_file_name,
            '{}/run{:03}/logs/'.format(DQM_OUT_DIR,int(run)))

        # close tmp file
        #_tf.close()


    return

def run_dqm(run_id,file_pattern,DQM_DATA_DIR,JSON_ME,JSON_RECON,DQM_OUT_DIR,logfile=None,skip='Clean'):
    """
    """
    msm = ""
    
    panaSKImg = ['dqmSKImg', file_pattern,
            '--dqm-data-dir {}'.format(DQM_DATA_DIR),
            '-j {}'.format(JSON_RECON),
            '--me-json {}'.format(JSON_ME),
            '--run {}'.format(run_id),
            '--skip {}'.format(skip),
            '-o {} '.format(DQM_OUT_DIR)]

    if logfile is not None:
        proc = subprocess.check_call([" ".join(panaSKImg)],stdout=logfile,
                stderr=logfile, shell=True)

    return proc


def run_recon(run_id,skip='_skip_1',logfile=None,isSource="Source"):
    """
    """

    if int(run_id)>=582 and int(run_id)<=604:
        # runs where calibration looks higher than expected
        oroot = 'bcrpanaSKImg_clustersRec_Image_{}_{:03}.root'.format(isSource,run_id)
    elif int(run_id) <= 390:
        # runs with old configuration
        oroot = 'orpanaSKImg_clustersRec_Image_{}_{:03}.root'.format(isSource,run_id)
    elif int(run_id)>=391 and int(run_id)<=439:
        # testing runs
        oroot = 'tpanaSKImg_clustersRec_Image_{}_{:03}.root'.format(isSource,run_id)
    elif int(run_id)>=809 and int(run_id)<=850:
        # serial register runs
        oroot = 'srpanaSKImg_clustersRec_Image_{}_{:03}.root'.format(isSource,run_id)
    else:
        # runs with new configuration
        oroot = 'nrpanaSKImg_clustersRec_Image_{}_{:03}.root'.format(isSource,run_id)

    #### add command lines to panaSKImg to run reconstruction
    panaSKImg = ['/data/waders/pysimdamicm/venv_compton/bin/panaSKImg',
            '"{}/run{:03}/avgimg/Image*fits"'.format(DQM_OUT_DIR,run_id),
            '-j {}'.format(JSON_RECON),
            '--skip {}'.format(skip),
            '-o {}'.format('"{}/run{:03}/recon"'.format(DQM_OUT_DIR,run_id)),
            '--oroot {}'.format(oroot),
            '--mask {}'.format(RECON_MASK)
            ]

    if logfile is not None:
        proc = subprocess.check_call([" ".join(panaSKImg)],stdout=logfile,
                stderr=logfile, shell=True)

    return proc

 
###############################################################################
###############################################################################
# registering the job
#schedule.every(1).hours.do(job,'do_job')
#schedule.every(1).minutes.do(do_run)
#schedule.every(3).seconds.do(do_run)
#
#while True:
#    schedule.run_pending()
#    # in seconds
#    time.sleep(1)

if __name__ == '__main__': 
    import argparse
    from argparse import Action
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--idir",
            action="store",
            dest="idir",
            help='Input directory ')

    parser.add_argument("--odir",
            action="store",
            dest="odir",
            help='Output directory ')

    parser.add_argument("--me-json",
            action="store",
            dest="me_json",
            help='JSON file for the ME ')

    parser.add_argument("--json",
            action="store",
            dest="json",
            help='JSON file for the data reprocessing ')

    parser.add_argument("--run",
            action="store",
            dest="run",
            help='run number')

    arg = parser.parse_args(args=None if sys.argv[1:] else ['--help']) 

    do_run( arg.idir, arg.odir, arg.me_json, arg.json, arg.run )


