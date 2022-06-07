import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        # need to reshape
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            #nn.ReLU(True)
        )
        self.encoder_fc2 = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            #nn.ReLU(True)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(1024, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(True)
        )
        #need to reshape back
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.BatchNorm2d(3),
            #nn.Tanh()
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.encoder(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = self.encoder_fc1(x1)
        #print(x1.shape)
        x2 = self.encoder_fc2(x2)
        #print(x2.shape)
        mu = x1
        log_var = x2
        z = self.reparameterize(mu, log_var)
        z = self.decoder_fc(z)
        #print(z.shape)
        z = z.view(64, 512, 4, 4)
        #print(z.shape)
        recon = self.decoder(z)
        return recon, mu, log_var

if __name__ == '__main__':
    model = VAE().cuda()
    print(model)
