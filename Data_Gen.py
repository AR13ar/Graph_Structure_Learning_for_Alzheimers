import numpy as np
import fnmatch
import glob
from medpy.io import load
import cv2
from torch.utils.data import Dataset, DataLoader
import image_slicer
import pickle as pkl

''' Define Data Generator
'''

class data_gen(Dataset):
    def __init__(self, file_path, label, transform, size):
        self.file_path = file_path
        self.transform = transform
        self.size = size
        self.label = label

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, indx):
         images_full = np.ones((self.size,1,256,256))
         images_noise = np.ones((self.size,1,256,256))
         fp = self.file_path[indx][0].rsplit("\\")[-5]
         for i in range(self.size):
             img, _ = load(self.file_path[indx][i])
             img = cv2.resize(img.squeeze(), (256,256))

	           ''' Add noise for Denoising Autoencoder Model
             '''
             noise = np.random.normal(0, 1, (256,256))
             img_noise = img + noise
             img_noise = img_noise.reshape((1,256,256))
             img = img.reshape((1,256,256))
             images_full[i] = img
             images_noise[i] = img_noise
         label = self.label[indx]
         return [ images_full, images_noise, label, fp]