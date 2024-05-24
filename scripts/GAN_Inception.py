import os
import numpy as np
import glob


#              E X T R A C T I N G  L A B E L S

# path to selected data directory
imagedir = "/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/SelectedData"

cur_dir = os.getcwd() # getting the current directory
os.chdir(imagedir)  # the parent folder with sub-folders

# Get number of samples per family
list_fams = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names
no_imgs = []  # No. of samples per family
for i in range(len(list_fams)):
    os.chdir(list_fams[i])
    len1 = len(glob.glob('*.png'))  # assuming the images are stored as 'png'
    no_imgs.append(len1)
    os.chdir('..')
num_samples = np.sum(no_imgs)  # total number of all samples