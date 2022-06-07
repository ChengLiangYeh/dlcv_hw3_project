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

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=938, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2 ** 4)
patch = (1, patch, patch)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


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
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label


# Loss function
adversarial_loss = torch.nn.MSELoss()
task_loss = torch.nn.CrossEntropyLoss()

# Loss weights
lambda_adv = 1     # 1 
lambda_task = 0.08  # 0.1 跟domain A像不像

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()

if cuda:
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    adversarial_loss.cuda()
    task_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)

# Configure data loader
dataloader_A = torch.utils.data.DataLoader(Train_Dataset(img_root='hw3_data/digits/mnistm/train',
                                                            csv_root='hw3_data/digits/mnistm/train.csv',
                                                            transform=transforms.Compose([transforms.RandomRotation(10),
                                                                                            transforms.CenterCrop((22,22)),
                                                                                            transforms.Resize((opt.img_size,opt.img_size)), 
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                                                            ]
                                                                                        )
                                                        ),
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            )

dataloader_B = torch.utils.data.DataLoader(Train_Dataset(img_root='hw3_data/digits/svhn/train',
                                                            csv_root='hw3_data/digits/svhn/train.csv',
                                                            transform=transforms.Compose([transforms.RandomRotation(10),
                                                                                            transforms.CenterCrop((24,20)),
                                                                                            transforms.Resize((opt.img_size,opt.img_size)), 
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                                                            ]
                                                                                        )
                                                        ),
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            )

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(generator.parameters(), classifier.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

'''
#####load weight#####
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    print('model loaded from %s' % checkpoint_path)

g_checkpoint_path = './pth/pixelda_g_epoch=1.pth'
c_checkpoint_path = './pth/pixelda_c_epoch=1.pth'
load_checkpoint(g_checkpoint_path, generator)
load_checkpoint(c_checkpoint_path, classifier)
'''

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []
history_correct = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):

        classifier.train()

        batch_size = imgs_A.size(0)
        '''
        print('batch_size=', batch_size)
        print('imgs_A_size=', imgs_A.shape)
        print('labels_A_size=', labels_A.shape)
        print('imgs_B_size=', imgs_B.shape)
        print('labels_B_size=', labels_B.shape)
        '''
        #因為不同dataset中資料數量不同，因此會有多出幾張的情況!
        if imgs_A.size(0) > imgs_B.size(0):
            batch_size = imgs_B.size(0)
            imgs_A = imgs_A[0:batch_size, :, :, :]
            labels_A = labels_A[0:batch_size]
            '''
            print('=====after batch_size balance=====')
            print('batch_size=', batch_size)
            print('imgs_A_size=', imgs_A.shape)
            print('labels_A_size=', labels_A.shape)
            print('imgs_B_size=', imgs_B.shape)
            print('labels_B_size=', labels_B.shape)
            '''
        elif imgs_B.size(0) > imgs_A.size(0):
            batch_size = imgs_A.size(0)
            imgs_B = imgs_B[0:batch_size, :, :, :]
            labels_B = labels_B[0:batch_size]
            '''
            print('=====after batch_size balance=====')
            print('batch_size=', batch_size)
            print('imgs_A_size=', imgs_A.shape)
            print('labels_A_size=', labels_A.shape)
            print('imgs_B_size=', imgs_B.shape)
            print('labels_B_size=', labels_B.shape)            
            '''
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs_A = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
        labels_A = Variable(labels_A.type(LongTensor))
        imgs_B = Variable(imgs_B.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise
        z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        fake_B = generator(imgs_A, z)

        # Perform task on translated source image
        label_pred = classifier(fake_B)

        # Calculate the task loss
        task_loss_ = (task_loss(label_pred, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2 #G生成的fake圖片經過classifier預測label -> pred_label跟label_A 越多一樣越好 -> 代表生成的圖片有domain A 的label資訊

        # Loss measures generator's ability to fool the discriminator
        g_loss = lambda_adv * adversarial_loss(discriminator(fake_B), valid) + lambda_task * task_loss_

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        #print('imgs_B=',imgs_B.shape)
        #print('valid=',valid.shape)
        real_loss = adversarial_loss(discriminator(imgs_B), valid)
        fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        acc = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())  #跟原本domain的數字像不像!
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.numpy()) #在新的Domain上表現
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        if len(dataloader_A) > len(dataloader_B):
            data_leng = len(dataloader_B)
        else:
            data_leng = len(dataloader_A)
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]"
            % (
                epoch,
                opt.n_epochs,
                i,
                data_leng, #有可能錯，不是A就是B
                d_loss.item(),
                g_loss.item(),
                100 * acc,
                100 * np.mean(task_performance),
                100 * target_acc,
                100 * np.mean(target_performance),
            )
        )

        batches_done = len(dataloader_A) * epoch + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((imgs_A.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            save_image(sample, "images/%d.png" % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)
    
    if epoch % 1 == 0:
        augmentation = transforms.Compose([transforms.Resize(32, interpolation=2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        target_dataset_with_label = Train_Dataset(img_root='hw3_data/digits/svhn/test', csv_root='hw3_data/digits/svhn/test.csv', transform=augmentation) # use Train_dataset for 'label'
        test_dataloader = DataLoader(target_dataset_with_label, batch_size=128, shuffle=False)
        result = []
        correct = 0
        classifier.eval()
        for i, (test_data, test_data_label) in enumerate(test_dataloader):
            test_data = test_data.cuda()
            class_logits = classifier(test_data)
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
        history_correct.append((correct / len(test_dataloader.dataset)))
        print(history_correct)
        generator.train()
        classifier.train()
        store_dir = './pth'
        torch.save(generator.state_dict(), os.path.join(store_dir + '/pixelda_g_epoch=%s' %(epoch) + '.pth'))
        torch.save(classifier.state_dict(), os.path.join(store_dir + '/pixelda_c_epoch=%s' %(epoch) + '.pth'))




