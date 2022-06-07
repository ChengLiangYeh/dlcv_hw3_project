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

#shell script
import sys
testing_image_directory_root = sys.argv[1]
print('$1=',testing_image_directory_root) #hw3_data/digits/mnistm/test
target_domain_name = sys.argv[2]
print('$2=',target_domain_name) #mnistm, usps or svhn.
output_csv_file_root = sys.argv[3]
print('$3=',output_csv_file_root) #hw3_data/digits/mnistm/test_pred.csv

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
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

for epoch in range(1):
    with torch.no_grad():  
            result = []
            label_predictor.eval()
            feature_extractor.eval()
            img_index_list = []
            for i, (test_data, image_fn) in enumerate(test_dataloader):
                #print(len(image_fn)#128
                for j in range(len(image_fn)):
                    img_index = (image_fn[j].split('/', -1))[-1]
                    #print(img_index)
                    img_index_list.append(img_index)

                test_data = test_data.cuda()
                class_logits = label_predictor(feature_extractor(test_data))
                x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
                result.append(x)
            import pandas as pd
            result = np.concatenate(result)
            df = pd.DataFrame({'image_name': img_index_list, 'label': result})
            df.to_csv(output_csv_file_root, index=False) ##
