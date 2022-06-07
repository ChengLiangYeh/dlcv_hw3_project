import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os 
import numpy as np
from PIL import Image
import pandas as pd 

class Train_Dataset(Dataset):
    def __init__(self, img_root, csv_root, transform=None):
        self.filenames = []
        self.labels = []
        self.img_root = img_root
        self.csv_root = csv_root 
        self.transform = transform
        filenames = glob.glob(os.path.join(img_root, '*.png'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        self.filenames.sort()
        #print(len(self.filenames))

        df = pd.read_csv(csv_root)
        self.labels = list(df.loc[:, 'label'])
        #print(len(self.labels))
        #print(self.labels)

    def __getitem__(self, index):
        image_fn = self.filenames[index]
        
        image = Image.open(image_fn)

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
            if image.shape[0] == 1:
                image_c2 = torch.cat((image, image), 0)
                #print(image_c2.shape)
                image_c3 =  torch.cat((image_c2, image), 0)
                image = image_c3
                #print(image.shape)
        return image, label, image_fn
    
    def __len__(self):
        return self.len

class Test_Dataset(Dataset):
    def __init__(self, img_root, transform=None):
        self.filenames = []
        self.img_root = img_root
        self.transform = transform
    
        filenames = glob.glob(os.path.join(img_root, '*.png'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        self.filenames.sort()

    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)
            #print(image.shape)
            if image.shape[0] == 1:
                image_c2 = torch.cat((image, image), 0)
                #print(image_c2.shape)
                image_c3 =  torch.cat((image_c2, image), 0)
                image = image_c3
                #print(image.shape)
        return image, image_fn
    
    def __len__(self):
        return self.len



if __name__ == '__main__':
    augmentation = transforms.Compose([transforms.Resize(32, interpolation=2),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    trainset = Train_Dataset(img_root='hw3_data/digits/usps/train', csv_root='hw3_data/digits/usps/train.csv', transform=augmentation)
    print('# images in trainset:', len(trainset))
    trainset_loader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4)
    dataiter = iter(trainset_loader)
    images, labels = dataiter.next()
    print('Image tensor in each batch:', images.shape, images.dtype)
    print('labels=', labels)
    import matplotlib.pyplot as plt
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    imshow(torchvision.utils.make_grid(images))
    


    testset = Test_Dataset(img_root='hw3_data/digits/usps/test', transform=augmentation)
    print('# images in testset:', len(testset))
    testset_loader = DataLoader(testset, batch_size=10, shuffle=False, num_workers=4)
    dataiter = iter(testset_loader)
    images = dataiter.next()
    print('Image tensor in each batch:', images.shape, images.dtype)
    import matplotlib.pyplot as plt
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    imshow(torchvision.utils.make_grid(images))
