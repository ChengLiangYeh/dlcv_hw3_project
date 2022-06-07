import torch
from torch import optim
from VAE_model import VAE
from facedataset import FaceDataset
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import manifold, datasets
 
  
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import sys
output_image_root = sys.argv[1]
print('$1=',output_image_root) #hw3_data/digits/mnistm/test
    
random.seed(17)
same_seeds(17)

z_sample = torch.randn(32, 1024)

model = VAE().cuda()
criterion = nn.MSELoss(reduction='sum')

model.eval()

#load weight
checkpoint_path = './checkpoint_360.pth'
state = torch.load(checkpoint_path)
model.load_state_dict(state)
'''
# 準備 dataloader, model, loss criterion 和 optimizer
augmentation = transforms.Compose([     #transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.CenterCrop(64),
                                        #transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
trainset = FaceDataset(root='../hw3_data/face/test', transform=augmentation)
dataloader = DataLoader(trainset, batch_size=10, shuffle=True)
dataloader_fix = DataLoader(trainset, batch_size=64, shuffle = False)
dataiter = iter(dataloader_fix)
imgs_fix = dataiter.next()
imgs_fix = imgs_fix.cuda()
#print(imgs_fix.shape)
'''
'''
###tSNE###
testset = FaceDataset(root='../hw3_data/face/test', transform=transforms.ToTensor())
testdata_loader = DataLoader(testset, batch_size=2621, shuffle=False) #total test images = 2621
###tSNE###
'''

with torch.no_grad():       
    #sample_dir = output_image_root 
        
    # Save the sampled images
    z = z_sample.cuda()
    z = model.decoder_fc(z)
    z = z.view(32,512,4,4)
    out = model.decoder(z)
    save_image(out, output_image_root)
'''       
    # Save the reconstructed images
    recon, mu, log_var = model(imgs_fix)
    recon = recon[10:20, :, :, :]
    imgs_fix = imgs_fix[10:20, :, :, :]
    x_concat = torch.cat([imgs_fix.view(-1, 3, 64, 64), recon.view(-1, 3, 64, 64)], dim=3)
    save_image(x_concat, os.path.join(sample_dir, 'fig1_3.jpg'))
    for i in range(10):
        recon_mse_loss = criterion(recon[i, :, :, :], imgs_fix[i, :, :, :])
        print(recon_mse_loss) 
'''
'''
    #tSNE
    df = pd.read_csv('../hw3_data/face/test.csv')
    #print(df)
    df = np.array(df)
    #print(df)
    #select = df['Male'][0:5]  
    #print(df[0,8]) #male在第八行
    gender_list = df[0: , 3]
    #print(type(gender_list))
    #print(gender_list.shape)

    for data in testdata_loader:
        img = data
        img = img.cuda()
        x1 = model.encoder(img)
        x2 = model.encoder(img)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = model.encoder_fc1(x1)
        x2 = model.encoder_fc2(x2)
        mu = x1
        log_var = x2
        z = model.reparameterize(mu, log_var)
        print(z.shape)

    z = z.cpu()
    z_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(z)

    #Data Visualization
    z_min, z_max = z_tsne.min(0), z_tsne.max(0)
    z_norm = (z_tsne - z_min) / (z_max - z_min)  #Normalize
    plt.figure(figsize=(8, 8))
    for i in range(z_norm.shape[0]):
        plt.text(z_norm[i, 0], z_norm[i, 1], str(gender_list[i]), color=plt.cm.Set1(gender_list[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./tsne_output.png")
    plt.show()
'''
