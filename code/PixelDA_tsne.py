import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from digit_dataset import Train_Dataset
from digit_dataset import Test_Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import manifold, datasets

parser = argparse.ArgumentParser()
parser.add_argument("pos1")
parser.add_argument("pos2")
parser.add_argument("pos3")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=114, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

testing_image_directory_root = opt.pos1
print('$1=',testing_image_directory_root) #hw3_data/digits/mnistm/test
target_domain_name = opt.pos2
print('$2=',target_domain_name) #mnistm, usps or svhn.
testing_csv_root = opt.pos3
print('$3=',testing_csv_root) #hw3_data/digits/mnistm/test_pred.csv

cuda = True if torch.cuda.is_available() else False


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels * opt.img_size ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(opt.channels * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512)
        )

        input_size = opt.img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, opt.n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label

'''
#shell script
import sys
testing_image_directory_root = sys.argv[1]
print('$1=',testing_image_directory_root) #hw3_data/digits/mnistm/test
target_domain_name = sys.argv[2]
print('$2=',target_domain_name) #mnistm, usps or svhn.
output_csv_file_root = sys.argv[3]
print('$3=',output_csv_file_root) #hw3_data/digits/mnistm/test_pred.csv
'''

# Initialize generator and discriminator
generator = Generator()
classifier = Classifier()

if cuda:
    generator.cuda()
    classifier.cuda()

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)

if target_domain_name == 'mnistm':
    g_checkpoint_path = './pixelda_g_epoch=459_usps_mnistm.pth'
    c_checkpoint_path = './pixelda_c_epoch=459_usps_mnistm.pth'
    load_checkpoint(g_checkpoint_path, generator) ##
    load_checkpoint(c_checkpoint_path, classifier) ##
elif target_domain_name == 'usps':
    g_checkpoint_path = './pixelda_g_epoch=129_svhn_usps.pth'
    c_checkpoint_path = './pixelda_c_epoch=129_svhn_usps.pth'
    load_checkpoint(g_checkpoint_path, generator) ##
    load_checkpoint(c_checkpoint_path, classifier) ##
elif target_domain_name == 'svhn':
    g_checkpoint_path = './pixelda_g_epoch=60_mnistm_svhn.pth'
    c_checkpoint_path = './pixelda_c_epoch=60_mnistm_svhn.pth'
    load_checkpoint(g_checkpoint_path, generator) ##
    load_checkpoint(c_checkpoint_path, classifier) ##
else:
    print('wrong target domain name')


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

augmentation = transforms.Compose([#transforms.RandomRotation(10),
                                    #transforms.CenterCrop((24,20)),
                                    transforms.Resize(32, interpolation=2),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
test_dataset = Test_Dataset(img_root=testing_image_directory_root, transform=augmentation) # use Train_dataset for 'label'
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
for epoch in range(1):
    with torch.no_grad():
        result = []
        classifier.eval()
        generator.eval()
        img_index_list = []
        z_vector = torch.zeros(1,3,32,32)
        z_vector = z_vector.cuda()
        for i, (test_data, image_fn) in enumerate(test_dataloader):
            for j in range(len(image_fn)):
                    img_index = (image_fn[j].split('/', -1))[-1]
                    #print(img_index)
                    img_index_list.append(img_index)
            batch_size = test_data.shape[0]
            #print(batch_size)
            z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))))
            test_data = test_data.cuda()
            test_data = Variable(test_data.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
            z_vector_batch = generator(test_data, z)
            print(z_vector_batch.shape)
            z_vector_batch = z_vector_batch.cuda()
            z_vector = torch.cat((z_vector, z_vector_batch), 0)
        z_vector = z_vector.view(z_vector.shape[0], 32*32*3)
        print(z_vector.shape)
        z_vector = z_vector[1:, :]
        #print(z_vector.shape)
        z_vector = z_vector.cpu()
        print(z_vector.shape)
        #tSNE
        df = pd.read_csv(testing_csv_root)
        #print(df)
        df = np.array(df)
        #print(df)
        #select = df['Male'][0:5]  
        #print(df[0,8]) #male在第八行
        label_list = df[0: , 1]
        print(label_list)
        print(label_list.shape)

        z_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(z_vector)
        #Data Visualization
        z_min, z_max = z_tsne.min(0), z_tsne.max(0)
        z_norm = (z_tsne - z_min) / (z_max - z_min)  #Normalize
        plt.figure(figsize=(8, 8))
        for i in range(z_norm.shape[0]):
            plt.text(z_norm[i, 0], z_norm[i, 1], str(label_list[i]), color=plt.cm.Set1(label_list[i]), fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig("./tsne_output_svhn.png")
        plt.show()
