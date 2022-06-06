# TODO: create shell script for running your VAE model

#!/bin/bash

wget -O checkpoint_360.pth "https://www.dropbox.com/s/l3r8dcj7bkcdz0c/checkpoint_360.pth?dl=1"
 
python3 VAE_eval2.py $1
