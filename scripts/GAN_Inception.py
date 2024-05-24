import os
import numpy as np
import glob


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

