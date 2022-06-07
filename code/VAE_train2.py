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

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)
z_sample = torch.randn(64, 1024)

model = VAE().cuda()
criterion = nn.MSELoss(reduction='sum')
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0)

model.train()
n_epoch = 1000

#load weight
#checkpoint_path = './VAEcheckpoints/checkpoint_70.pth'
#state = torch.load(checkpoint_path)
#model.load_state_dict(state)

# 準備 dataloader, model, loss criterion 和 optimizer
augmentation = transforms.Compose([     transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.CenterCrop(64),
                                        #transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
trainset = FaceDataset(root='hw3_data/face/train', transform=augmentation)
dataloader = DataLoader(trainset, batch_size=64, shuffle=True)

dataloader_fix = DataLoader(trainset, batch_size=64, shuffle = False)
dataiter = iter(dataloader_fix)
imgs_fix = dataiter.next()
imgs_fix = imgs_fix.cuda()

epoch_loss = 0
epoch_MSE_loss = 0
epoch_KLD = 0

epoch_MSE_loss_store = np.zeros(n_epoch)
epoch_KLD_loss_store = np.zeros(n_epoch)

for epoch in range(n_epoch):

    if epoch == 150 :
        learning_rate = learning_rate / 2
    elif epoch == 300 :
        learning_rate = learning_rate / 2
    elif epoch == 450 :
        learning_rate = learning_rate / 2

    epoch_loss = 0
    epoch_MSE_loss = 0
    epoch_KLD = 0
    for data in dataloader:
        img = data
        img = img.cuda()
        recon, mu, log_var = model(img)

        MSE_loss = criterion(recon, img)
        #print('MSE_loss=',MSE_loss)
        KLD = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #print('KLD=',KLD)
        loss = MSE_loss + 1 * KLD
        #print('loss=',loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), './VAEcheckpoints2/checkpoint_{}.pth'.format(epoch+1))

        epoch_loss += loss.item()
        epoch_MSE_loss += MSE_loss.item()
        epoch_KLD += KLD.item()
    #print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))
    #print('epoch [{}/{}], MSE_loss:{:.5f}'.format(epoch+1, n_epoch, epoch_MSE_loss))
    #print('epoch [{}/{}], KLD:{:.5f}'.format(epoch+1, n_epoch, epoch_KLD))
    #print(len(dataloader))

    epoch_MSE_loss_store[epoch] = epoch_MSE_loss / 40000
    #print(epoch_MSE_loss_store)
    #print(epoch_MSE_loss_store.shape)
    epoch_KLD_loss_store[epoch] = epoch_KLD / 40000
    #print(epoch_KLD_loss_store)
    #print(epoch_MSE_loss_store.shape)
    print('epoch [{}/{}], loss:{:.5f}, MSE_loss:{:.5f}, KLD:{:.5f}'.format(epoch+1, n_epoch, epoch_loss/len(dataloader.dataset), epoch_MSE_loss/len(dataloader.dataset), epoch_KLD/len(dataloader.dataset)))

    with torch.no_grad():       
        sample_dir = './VAE_output_img2/'
        
        # Save the sampled images
        z = z_sample.cuda()
        z = model.decoder_fc(z)
        z = z.view(64,512,4,4)
        out = model.decoder(z)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))
        
        # Save the reconstructed images
        recon, mu, log_var = model(imgs_fix)
        x_concat = torch.cat([imgs_fix.view(-1, 3, 64, 64), recon.view(-1, 3, 64, 64)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
# 訓練完成後儲存 model
torch.save(model.state_dict(), './VAEcheckpoints2/last_checkpoint.pth')

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
plt.plot(epoch_MSE_loss_store)
plt.title("MSE LOSS") # title
plt.ylabel("LOSS") # y label
plt.xlabel("EPOCH") # x label
ax2 = fig.add_subplot(2,2,2)
plt.plot(epoch_KLD_loss_store)
plt.title("MSE LOSS") # title
plt.ylabel("LOSS") # y label
plt.xlabel("EPOCH") # x label
plt.savefig('./VAE_loss_fig2/loss.png')
