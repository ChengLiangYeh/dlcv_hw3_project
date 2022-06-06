# TODO: create shell script for running your DANN model

#!/bin/bash

wget -O feature_extractor_checkpoint_36_mnistm_svhn.pth "https://www.dropbox.com/s/03vr2lvdcm42hub/feature_extractor_checkpoint_36_mnistm_svhn.pth?dl=1"

wget -O feature_extractor_checkpoint_51_svhn_usps.pth "https://www.dropbox.com/s/sf3p7lp108nrl7j/feature_extractor_checkpoint_51_svhn_usps.pth?dl=1"

wget -O feature_extractor_checkpoint_51_usps_mnistm.pth "https://www.dropbox.com/s/uuxmjs6xqubwy9c/feature_extractor_checkpoint_51_usps_mnistm.pth?dl=1"

wget -O label_predictor_checkpoint_36_mnistm_svhn.pth "https://www.dropbox.com/s/tbw7yhl72aej677/label_predictor_checkpoint_36_mnistm_svhn.pth?dl=1"

wget -O label_predictor_checkpoint_51_svhn_usps.pth "https://www.dropbox.com/s/tj1cculdwtrh0an/label_predictor_checkpoint_51_svhn_usps.pth?dl=1"

wget -O label_predictor_checkpoint_51_usps_mnistm.pth "https://www.dropbox.com/s/4jr1cnsduhsod1e/label_predictor_checkpoint_51_usps_mnistm.pth?dl=1"
 
python3 DaNN_eval.py $1 $2 $3 
