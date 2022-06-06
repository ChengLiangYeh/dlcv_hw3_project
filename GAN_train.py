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

# hyperparameters 
batch_size = 64
z_dim = 100
lr = 1e-4  
n_epoch = 1000
save_dir_root = 'logs/'

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

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

###
g_checkpoint_path = './dcgan_g_epoch=388.pth'
d_checkpoint_path = './dcgan_d_epoch=388.pth'
load_checkpoint(g_checkpoint_path, G)
load_checkpoint(d_checkpoint_path, D)
###

same_seeds(777)
# dataloader (You might need to edit the dataset path if you use extra dataset.)
augmentation = transforms.Compose([     transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.CenterCrop(64),
                                        #transforms.RandomVerticalFlip(p=0.5),

                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
trainset = FaceDataset(root='../hw3_data/face/train', transform=augmentation)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


#TRAIN
# for logging
z_sample = Variable(torch.randn(40, z_dim)).cuda()

for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()

        bs = imgs.size(0)

        """ Train D """
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)
        '''
        # label
        choicelist = ['notflip1', 'notflip2', 'notflip3', 'notflip4', 'notflip5', 'notflip6', 'notflip7', 'notflip8', 'notflip9', 'flip'] #200個epoch後降低flip機率成20%
        choice = random.choice(choicelist)
        if choice == 'flip':
            r_label = torch.zeros((bs)).cuda()
            f_label = torch.ones((bs)).cuda()
        else:
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()
        '''
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()
        
        # dis
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        
        # compute loss
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = (r_loss + f_loss) / 2

        # update model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        """ train G """
        # leaf
        z = Variable(torch.randn(bs, z_dim)).cuda()
        f_imgs = G(z)

        # dis
        f_logit = D(f_imgs)
        #print(f_logit.shape)
        #print(f_logit)
        
        # compute loss
        loss_G = criterion(f_logit, r_label)

        # update model
        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # log
        print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(save_dir_root, f'Epoch_{epoch+389:03d}.jpg') ###
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')
    # show generated image
    #grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    #plt.figure(figsize=(10,10))
    #plt.imshow(grid_img.permute(1, 2, 0))
    #plt.show()
    G.train()
    store_dir = './pth'
    '''
    if (e+1) % 1 == 0:
        torch.save(G.state_dict(), os.path.join(store_dir + '/dcgan_g_epoch=%s' %(e) + '.pth')) ###
        torch.save(D.state_dict(), os.path.join(store_dir + '/dcgan_d_epoch=%s' %(e) + '.pth')) ###
    '''