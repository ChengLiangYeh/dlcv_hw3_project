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


feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

lr = 0.0001
optimizer_F = optim.Adam(feature_extractor.parameters(),lr=lr, betas=(0.5, 0.999))
optimizer_C = optim.Adam(label_predictor.parameters(),lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(domain_classifier.parameters(),lr=lr, betas=(0.5, 0.999))


augmentation = transforms.Compose([transforms.Resize(32, interpolation=2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
source_dataset = Train_Dataset(img_root='hw3_data/digits/mnistm/train', csv_root='hw3_data/digits/mnistm/train.csv', transform=augmentation)
target_dataset = Test_Dataset(img_root='hw3_data/digits/svhn/train', transform=augmentation)
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)


def train_epoch(source_dataloader, target_dataloader, lamb):
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        #print('source_data=',source_data.shape)
        source_label = source_label.cuda()
        #print('source_label=',source_label.shape)
        target_data = target_data.cuda()
        
        # 把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣) 參考自李宏毅老師課程
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # source data label＝1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        #print(feature.shape)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。參考自李宏毅老師課程
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier 參考自李宏毅老師課程
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。參考自李宏毅老師課程
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num
'''
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)

label_predictor_checkpoint_path = './Dann_output/pth/label_predictor_checkpoint_200.pth' ##
feature_extractor_checkpoint_path = './Dann_output/pth/feature_extractor_checkpoint_200.pth' ##
load_checkpoint(feature_extractor_checkpoint_path, feature_extractor) ##
load_checkpoint(label_predictor_checkpoint_path, label_predictor) ##
'''
# 訓練200 epochs
lamb = 0.07 #mnistm-svhn : 0.07
for epoch in range(100):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=lamb)

    torch.save(feature_extractor.state_dict(), './Dann_output/pth/feature_extractor_checkpoint_{}.pth'.format(epoch+1)) ##
    torch.save(label_predictor.state_dict(),'./Dann_output/pth/label_predictor_checkpoint_{}.pth'.format(epoch+1)) ##

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

    if epoch % 5 == 0:
        target_dataset_with_label = Train_Dataset(img_root='hw3_data/digits/svhn/test', csv_root='hw3_data/digits/svhn/test.csv', transform=augmentation) # use Train_dataset for 'label'
        test_dataloader = DataLoader(target_dataset_with_label, batch_size=128, shuffle=False)
        result = []
        correct = 0
        label_predictor.eval()
        feature_extractor.eval()
        for i, (test_data, test_data_label) in enumerate(test_dataloader):
            test_data = test_data.cuda()
            class_logits = label_predictor(feature_extractor(test_data))
            x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            result.append(x)
            #print(x.shape[0])
            #print(test_data_label.shape)
            for j in range(x.shape[0]):
                #print(x[j])
                #print(test_data_label[j])
                if x[j] == test_data_label[j]:
                    correct += 1
        #print(correct)
        print('Accuracy in target domain test data=', correct / len(test_dataloader.dataset))

#predict
print('Final Predict and Save .csv')
target_dataset_with_label = Train_Dataset(img_root='hw3_data/digits/svhn/test', csv_root='hw3_data/digits/svhn/test.csv', transform=augmentation) # use Train_dataset for 'label'
test_dataloader = DataLoader(target_dataset_with_label, batch_size=128, shuffle=False)
result = []
correct = 0
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, test_data_label) in enumerate(test_dataloader):
    test_data = test_data.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)
    #print(x.shape[0])
    #print(test_data_label.shape)
    for j in range(x.shape[0]):
        #print(x[j])
        #print(test_data_label[j])
        if x[j] == test_data_label[j]:
            correct += 1
#print(correct)
print('Accuracy in target domain test data=', correct / len(test_dataloader.dataset))
import pandas as pd
result = np.concatenate(result)
# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('./Dann_output/csv/DaNN_submission.csv',index=False)