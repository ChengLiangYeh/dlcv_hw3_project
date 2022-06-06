# TODO: create shell script for running your improved UDA model
#!/bin/bash

wget -O pixelda_c_epoch=60_mnistm_svhn.pth "https://www.dropbox.com/s/yuyzuzwsdm8by4u/pixelda_c_epoch%3D60_mnistm_svhn.pth?dl=1"

wget -O pixelda_g_epoch=60_mnistm_svhn.pth "https://www.dropbox.com/s/ul0adwe08mt4nq6/pixelda_g_epoch%3D60_mnistm_svhn.pth?dl=1"

wget -O pixelda_c_epoch=129_svhn_usps.pth "https://www.dropbox.com/s/wn5zzfu2owlyzmd/pixelda_c_epoch%3D129_svhn_usps.pth?dl=1"

wget -O pixelda_g_epoch=129_svhn_usps.pth "https://www.dropbox.com/s/ty8euozzfz2z9tz/pixelda_g_epoch%3D129_svhn_usps.pth?dl=1"

wget -O pixelda_c_epoch=459_usps_mnistm.pth "https://www.dropbox.com/s/0ng6ewzh6wu1f39/pixelda_c_epoch%3D459_usps_mnistm.pth?dl=1"

wget -O pixelda_g_epoch=459_usps_mnistm.pth "https://www.dropbox.com/s/73dgtb63ru5k7qa/pixelda_g_epoch%3D459_usps_mnistm.pth?dl=1"
 
python3 PixelDA_eval_new.py $1 $2 $3 
