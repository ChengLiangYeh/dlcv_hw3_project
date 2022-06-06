import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import os
import random
import numpy as np
from GAN_model import Generator
from GAN_model import Discriminator
from facedataset import FaceDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import sys
output_image_root = sys.argv[1]
print('$1=',output_image_root) 


# hyperparameters 
z_dim = 100
#save_dir_root = 'logs/'

#load trained model function:
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)

# model
G = Generator(in_dim=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

###
g_checkpoint_path = './dcgan_g_epoch=388.pth'
d_checkpoint_path = './dcgan_d_epoch=388.pth'
load_checkpoint(g_checkpoint_path, G)
load_checkpoint(d_checkpoint_path, D)
###

same_seeds(777)
z_sample = Variable(torch.randn(32, z_dim)).cuda()
G.eval()
f_imgs_sample = (G(z_sample).data + 1) / 2.0
filename = os.path.join(output_image_root) ###
torchvision.utils.save_image(f_imgs_sample, filename, nrow=8)
print(f' | Save some samples to {filename}.')