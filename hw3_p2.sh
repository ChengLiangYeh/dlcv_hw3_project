# TODO: create shell script for running your GAN model

#!/bin/bash

wget -O dcgan_d_epoch=388.pth "https://www.dropbox.com/s/dyb80nogjakqxu2/dcgan_d_epoch%3D388.pth?dl=1"

wget -O dcgan_g_epoch=388.pth "https://www.dropbox.com/s/5m6gab3t6fha6a0/dcgan_g_epoch%3D388.pth?dl=1"
 
python3 GAN_eval.py $1
