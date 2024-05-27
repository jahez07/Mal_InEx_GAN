import os
import numpy as np
import glob
from keras.models import Sequential
from keras.layers import Dense, Reshape, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam
from numpy.random import randint, randn
from numpy import ones, zeros


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

#              Discriminator Model

# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.
def define_discriminator(in_shape=(128,128,3)):
	model = Sequential()

	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape)) #16x16x128
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same')) #8x8x128
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten()) #shape of 8192
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid')) #shape of 1
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

test_discr = define_discriminator()
print(test_discr.summary())

#              Generator Model

def define_generator(latent_dim):
    model = Sequential()

    # Initial dense layer
    n_nodes = 256 * 8 * 8  # 4096 nodes
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 256)))  # Reshape to 4x4x256

    # Upsampling blocks
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    # Output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))  # Output shape: 32x32x3

    return model

test_gen = define_generator(100)
print(test_gen.summary())

# Define the combined generator and discriminator model, for updating the generator
# Discriminator is trained separately so here only generator will be trained by keeping
# the discriminator constant.

def define_gan(generator, discriminator):

  # Discriminator is trained separately. So set to not trainable.
  discriminator.trainable = False

  # Connect generator and discriminator
  model = Sequential()
  model.add(generator)
  model.add(discriminator)

  # compile model
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)

  return model

# load cifar training images
def load_real_samples():

  trainX = X_train

  # cConvert to float and scale.
  X = trainX.astype('float32')

  # scale from [0,255] to [-1,1]
  X = (X - 127.5) / 127.5
  #Generator uses tanh activation so rescale
  #original images to -1 to 1 to match the output of generator.
  return X

# Pick a batch of random real samples to train the GAN
# In fact, we will train the GAN on a half batch of real images and another
# half batch of fake images.
# For each real image we assign a label 1 and for fake we assign label 0.

def generate_real_samples(dataset, n_samples):

  # Choose random images
	ix = randint(0, dataset.shape[0], n_samples)

  # Select the random images and assign it to X
	X = dataset[ix]

  # Generate class labels and assign to y
  # Label=1 indicating they are real
	y = ones((n_samples, 1))
	return X, y

# Generate n_samples number of latent vectors as input for the generator

def generate_latent_points(latent_dim, n_samples):

  # generate points in the latent space
	x_input = randn(latent_dim * n_samples)

  # reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# Generate n_samples number of latent vectors as input for the generator

def generate_latent_points(latent_dim, n_samples):

  # generate points in the latent space
	x_input = randn(latent_dim * n_samples)

  # reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# Use the generator to generate n fake examples, with class labels
# Supply the generator, latent_dim and number of samples as input.
# Use the above latent point generator to generate latent points.

def generate_fake_samples(generator, latent_dim, n_samples):

  # generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)

  # predict using generator to generate fake samples.
	X = generator.predict(x_input)

  # Class labels will be 0 as these samples are fake.
	y = zeros((n_samples, 1))  #Label=0 indicating they are fake
	return X, y



#              G E N E R A T I N G  S A M P L E S


