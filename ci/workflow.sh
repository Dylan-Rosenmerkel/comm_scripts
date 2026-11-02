#!/bin/bash

setwaders

# SIMULATIONS
psimulCCDimg g4sims/psimulCCDimg_config_file.json --g4file g4sims/Pb210_1000events_bulk_inCCDSensor_run001.root --g4out g4sims

# DATA
panaSKImg -j data/moduletest_lowtemp_image_8.json data/avg_Image_8_Low_Temp_106_20260504_211939_25.fz --ext 1 -o data

# DQM
dqmSKImg dqm/avg_skips_125_20250211_022254_55.fits.bz2 -j dqm/panaSKImg_config_LBC_ACM_4DQM_PA07_104.json -m dqm/dqmSKImg_expectations_config.json -d . -o dqm -e 2 -r 456 -t TESTING



echo "========================================================================== g4sims"
outg4file=`ls g4sims/out_simulCCDimg_log*Pb210_1000events_bulk_inCCDSensor_run001.root`
compare_root_files outputs/out_simulCCDimg_log9047df1_Pb210_1000events_bulk_inCCDSensor_run001.root ${outg4file}
rm g4sims/out_simulCCDimg_log*root 

echo "========================================================================== data"
compare_root_files outputs/panaSKImg_clustersRec_log9047df1_avg_Image_8_Low_Temp_106_20260504_211939_25_ext1.root data/panaSKImg_clustersRec_avg_Image_8_Low_Temp_106_20260504_211939_25_ext1.root
rm data/panaSKImg_clustersRec_avg_Image_8_Low_Temp_106_20260504_211939_25_ext1.root

echo "========================================================================== dqm"
compare_root_files outputs/panaSKImg_clustersRec_log9047df1_avg_skips_125_20250211_022254_55_ext2.root dqm/panaSKImg_clustersRec_avg_skips_125_20250211_022254_55_ext2.root
rm dqm/panaSKImg_clustersRec_avg_skips_125_20250211_022254_55_ext2.root


echo "DONE."
