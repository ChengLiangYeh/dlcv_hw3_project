import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os 
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.filenames = []
        self.root = root 
        self.transform = transform
    
        filenames = glob.glob(os.path.join(root, '*.png'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        self.filenames.sort()
        #print(self.filenames)

    def __getitem__(self, index):
        image_fn = self.filenames[index]
        #print(image_fn)
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return self.len



if __name__ == '__main__':
    augmentation = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    trainset = FaceDataset(root='../hw3_data/face/test', transform=augmentation)
    print('# images in trainset:', len(trainset))
    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)

    dataiter = iter(trainset_loader)
    images = dataiter.next()
    print('Image tensor in each batch:', images.shape, images.dtype)
    print(images[0].shape)
    print(images[0])
    import matplotlib.pyplot as plt
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    imshow(torchvision.utils.make_grid(images))
    
