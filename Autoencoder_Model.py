
import torch.nn as nn
import numpy as np
import os
import fnmatch
import glob
from medpy.io import load
import cv2
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.autograd import Variable
import torch.nn.functional as F
import image_slicer
from torch.nn import Linear
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import pickle as pkl
from Data_generator import data_gen

''' Import Training Data For autoencoder
'''
train_filepath = []
train_labels = []

filepath_cn = "Path to preprocessed CN files"
filepath_ad = "Path to preprocessed AD files"
filepath_mci = "Path to preprocessed MCI files"

for i in range(len(filepath_cn)):
        train_filepath.append(filepath_cn[i])
        train_labels.append(0)

for i in range(len(filepath_ad)):
        train_filepath.append(filepath_ad[i])
        train_labels.append(1)

for i in range(len(filepath_mci)):
        train_filepath.append(filepath_mci[i])
        train_labels.append(2)



train_data = data_gen(file_path = train_filepath,label = train_labels, size = custom_size,transform = transforms.Compose([
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize([0], [1])]))

dataloader_train = torch.utils.data.DataLoader(train_data, batch_size = 32,
                        shuffle= True, num_workers= 0)

''' Autoencoder Model Definition
'''
class autoencoder3d(nn.Module):
    def __init__(self):
        super(autoencoder3d, self).__init__()
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,64, 3, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(128,128, 3, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16384 , 1024 ),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
        )

        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, 4,2,1),
            nn.ReLU(inplace=True),
            )
        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose3d(128,64,4,2,1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            )
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, 4,2,1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enco_out = self.encoder(x)
        feature_out = enco_out.view(x.size()[0],-1)
        latent_out = self.fc1(feature_out)
        out_layer1 = self.decoder_layer1(enco_out)
        out_layer2 = self.decoder_layer2(F.upsample(out_layer1,scale_factor=2))
        deco_out =  self.decoder_layer3(F.upsample(out_layer2,scale_factor=2))

        return deco_out , latent_out

autoen = autoencoder3d()
optimizer = torch.optim.Adam(autoen.parameters(), lr= 0.00009, weight_decay = 1e-5)
criterian = nn.MSELoss()

''' Model Training
'''
for epoch in range(100):
        loss_total = 0
        train_acc = 0
        count = 0
        for data in tqdm(dataloader_train):
            images, images_noise, label, fn = data
            batch_size = images.shape[0]
            optimizer.zero_grad()

            images_noise = (images_noise - images_noise.mean())/images_noise.std()
            output,latent = autoen(images_noise.reshape(batch_size,1,64,256,256).float())
            loss_recon = criterian(output, images.reshape(batch_size,1,64,256,256).float())
            loss_recon.backward()
            optimizer.step()
            print('Epoch: {} \tLoss Recon: {:.6f}'.format(epoch, loss_recon))

feature_to_be_saved = []
save_filepath = "Files to saved for GNN"
save_labels = "Labels for the files"

save_data = data_gen(file_path = save_filepath, label = save_labels, size = custom_size,transform = transforms.Compose([
                                                                                   transforms.ToTensor(),
                                                                                   transforms.Normalize([0], [1])]))


dataloader_save = torch.utils.data.DataLoader(save_data, batch_size = 1,
                        shuffle= False)

for data in tqdm(dataloader_save):
     images, images_noise, label, fn = data
     batch_size = images.shape[0]
     output,latent = autoen(images.reshape(batch_size,1,64,256,256).float())
     with torch.no_grad():
         feature_to_be_saved.append([np.array(latent.detach()), fn, int(label.detach())])

# Save files for GNN
file_path_save = "Path to save file"

with open(file_path_save + "//" + "File Name", "wb") as fp:
  pkl.dump(feature_to_be_saved, fp)