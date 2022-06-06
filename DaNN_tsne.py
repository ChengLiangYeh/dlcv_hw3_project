import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from digit_dataset import Train_Dataset
from digit_dataset import Test_Dataset

from DaNN_model import FeatureExtractor
from DaNN_model import LabelPredictor
from DaNN_model import DomainClassifier

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import manifold, datasets

#shell script
import sys
testing_image_directory_root = sys.argv[1]
print('$1=',testing_image_directory_root) #hw3_data/digits/mnistm/test
target_domain_name = sys.argv[2]
print('$2=',target_domain_name) #mnistm, usps or svhn.
testing_csv_root = sys.argv[3]
print('$3=',target_domain_name)

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)

if target_domain_name == 'mnistm':
    label_predictor_checkpoint_path = './label_predictor_checkpoint_51_usps_mnistm.pth' ##
    feature_extractor_checkpoint_path = './feature_extractor_checkpoint_51_usps_mnistm.pth' ##
    load_checkpoint(label_predictor_checkpoint_path, label_predictor) ##
    load_checkpoint(feature_extractor_checkpoint_path, feature_extractor) ##
elif target_domain_name == 'usps':
    label_predictor_checkpoint_path = './label_predictor_checkpoint_51_svhn_usps.pth' ##
    feature_extractor_checkpoint_path = './feature_extractor_checkpoint_51_svhn_usps.pth' ##
    load_checkpoint(label_predictor_checkpoint_path, label_predictor) ##
    load_checkpoint(feature_extractor_checkpoint_path, feature_extractor) ##
elif target_domain_name == 'svhn':
    label_predictor_checkpoint_path = './label_predictor_checkpoint_36_mnistm_svhn.pth' ##
    feature_extractor_checkpoint_path = './feature_extractor_checkpoint_36_mnistm_svhn.pth' ##
    load_checkpoint(label_predictor_checkpoint_path, label_predictor) ##
    load_checkpoint(feature_extractor_checkpoint_path, feature_extractor) ##
else:
    print('wrong target domain name')

augmentation = transforms.Compose([transforms.Resize(32, interpolation=2),
                                        transforms.ToTensor()
                                        ])
test_dataset = Test_Dataset(img_root=testing_image_directory_root, transform=augmentation)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False) #mnistm=10000 svhn=26032 usps=2007

for epoch in range(1):
    with torch.no_grad():  
            label_predictor.eval()
            feature_extractor.eval()
            img_index_list = []
            z_vector = torch.zeros(1, 512)
            z_vector = z_vector.cuda()
            #print(z_vector.shape)
            for i, (test_data, image_fn) in enumerate(test_dataloader):
                #print(len(image_fn)#128
                for j in range(len(image_fn)):
                    img_index = (image_fn[j].split('/', -1))[-1]
                    #print(img_index)
                    img_index_list.append(img_index)

                test_data = test_data.cuda()
                z_vector_batch = (feature_extractor(test_data))
                z_vector_batch = z_vector_batch.cuda()
                #print(z_vector_batch.shape)
                z_vector = torch.cat((z_vector, z_vector_batch), 0)
                #print(z_vector.shape)
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
            plt.savefig("./tsne_output_usps.png")
            plt.show()