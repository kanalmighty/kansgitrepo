# coding=UTF-8
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import myimplemention.tools.settings as settings

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

datasets = datasets.CIFAR10(root=settings.DCGAN_IMAGE_ROOT,
                       transform=transforms.Compose([transforms.Resize(settings.IMAGE_SIZE),
                                                    transforms.CenterCrop(settings.IMAGE_SIZE),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

dataloader = data.DataLoader(datasets, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.WORKERS)

device = torch.device("cuda:0" if (torch.cuda.is_available() and settings.NGPU > 0) else "cpu")
# one_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")#关闭坐标轴
# plt.title("Training Images")
# images = one_batch[0]
# image_total = np.transpose(vutils.make_grid(images, nrow=8), axes=(1, 2, 0))
# plt.imshow(image_total)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(settings.NZ, settings.NGF*8, 4, 1, 0, bias=False),
                                  nn.BatchNorm2d(settings.NGF*8),
                                  nn.ReLU(True),
                                  # state size. (ngf*8) x 4 x 4
                                  nn.ConvTranspose2d(settings.NGF*8, settings.NGF*4, 4, 2, 1, bias=False),
                                  nn.BatchNorm2d(settings.NGF*4),
                                  nn.ReLU(True),
                                  # state size. (ngf*4) x 8 x 8
                                  nn.ConvTranspose2d(settings.NGF * 4, settings.NGF * 2, 4, 2, 1, bias=False),
                                  nn.BatchNorm2d(settings.NGF * 2),
                                  nn.ReLU(True),
                                  # state size. (ngf*2) x 16 x 16
                                  nn.ConvTranspose2d(settings.NGF * 2, settings.NGF, 4, 2, 1, bias=False),
                                  nn.BatchNorm2d(settings.NGF),
                                  nn.ReLU(True),
                                  # state size. (ngf) x 32 x 32
                                  nn.ConvTranspose2d(settings.NGF, settings.NC, 4, 2, 1, bias=False),
                                  nn.Tanh()
                                  # state size. (nc x 64 x 64
                                  )

    def forward(self, input):
        return self.main(input)


netG = Generator(settings.NGPU)

if (settings.NGPU > 1):
    netG = nn.DataParallel(netG, list(range(settings.NGPU)))


netG.apply(weights_init)
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(settings.NC, settings.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(settings.NDF, settings.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(settings.NDF*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(settings.NDF * 2, settings.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(settings.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(settings.NDF * 4, settings.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(settings.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(settings.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(settings.NGPU)

# Handle multi-gpu if desired
if  (settings.NGPU > 1):
    netD = nn.DataParallel(netD, list(range(settings.NGPU)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, settings.NZ, 1, 1,  device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=settings.LR, betas=(settings.BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=settings.LR, betas=(settings.BETA1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(5):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):


        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)#batch_size
        label = torch.full((b_size, ), real_label, device=device)#make real label
        output = netD(data[0]).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, settings.NZ, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()


        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, 5, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output \
        # fixed_noise
        if (iters % 500 == 0) or ((epoch == 5-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# %%capture
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())