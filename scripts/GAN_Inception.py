import os
import numpy as np
import glob
from keras.models import Sequential
from keras.layers import Dense, Reshape, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam


#              E X T R A C T I N G  &  I N I T I A L I Z I N G  L A B E L S

# path to selected data directory
imagedir = "/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/SelectedData"

cur_dir = os.getcwd() # getting the current directory
os.chdir(imagedir)  # the parent folder with sub-folders

# Get number of samples per family

# vector of strings with family names
list_fams = sorted(os.listdir(os.getcwd()), key=str.lower)
no_imgs = []  # No. of samples per family
for i in range(len(list_fams)):
    os.chdir(list_fams[i])
    # assuming the images are stored as 'png'
    len1 = len(glob.glob('*.png'))
    no_imgs.append(len1)
    os.chdir('..')
# total number of all samples
num_samples = np.sum(no_imgs)

# Compute the labels
y = np.zeros(num_samples)
pos = 0
label = 0
for i in no_imgs:
    # Label: 0 Family: adload Number of images: 1050
    print ("Label:%2d\tFamily: %15s\tNumber of images: %d" % (label, list_fams[label], i))
    for j in range(i):
        y[pos] = label
        pos += 1
    label += 1
num_classes = label

#              L O A D I N G  D A T A

#              L O A D I N G  M O D E L

#              G A N  (Generative Adversarial Network)

#              G E N E R A T I N G  S A M P L E S


