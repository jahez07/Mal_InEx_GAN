import os
import glob
import pickle
import numpy as np
from numpy import ones, zeros
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from numpy.random import randint, randn
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Flatten, Dropout


#              E X T R A C T I N G  &  I N I T I A L I Z I N G  L A B E L S

# path to selected data directory
imagedir = "/MalHub/SelectedData"

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

# Compute the features
width, height,channels = (128,128,3)
X = np.zeros((num_samples, width, height, channels))
cnt = 0
list_paths = [] # List of image paths
print("Processing images ...")
for i in range(len(list_fams)):
    for img_file in glob.glob(list_fams[i]+'/*.png'):
        #print("[%d] Processing image: %s" % (cnt, img_file))
        list_paths.append(os.path.join(os.getcwd(),img_file))
        img = image.load_img(img_file, target_size=(128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X[cnt] = x
        cnt += 1
print("Images processed: %d" %(cnt))

os.chdir(cur_dir)
print("shape:",X.shape)
#print(X)
#print(y)
images = np.array(X)
labels = np.array(y)

# Loading each family
X_train = []
for i in range(len(labels)):
    if labels[i] == 0:
        X_train.append(images[i])
X_train = np.array(X_train)


#              L O A D I N G  M O D E L

with open('/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/Models/rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

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

# load MalHub training images
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

# Train the generator and discriminator
# We loop through a number of epochs to train our Discriminator by first selecting
# a random batch of images fromv our true/real dataset.
# Then, generating a set of images using the generator.
# Feed both set of images into the Discriminator.
# Finally, set the loss parameters for both the real and fake images, as well as the combined loss.
Gen_Loss = []
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=10):

  #the discriminator model is updated for a half batch of real samples
  # and a half batch of fake samples, combined a single batch.
  bat_per_epo = int(dataset.shape[0] / n_batch)
  half_batch = int(n_batch / 2)

  # manually enumerate epochs and bacthes.
  for i in range(n_epochs):

    # enumerate batches over the training set
    for j in range(bat_per_epo):

      # Train the discriminator on real and fake images, separately (half batch each)
      # Research showed that separate training is more effective.
			# get randomly selected 'real' samples
      X_real, y_real = generate_real_samples(dataset, half_batch)

      # Update discriminator model weights
      # Train_on_batch allows you to update weights based on a collection
      # of samples you provide
      # Let us just capture loss and ignore accuracy value (2nd output below)
      d_loss_real, _ = d_model.train_on_batch(X_real, y_real)

      # Generate 'fake' examples
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

      # Update discriminator model weights
      d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)

      # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..

			# Prepare points in latent space as input for the generator
      X_gan = generate_latent_points(latent_dim, n_batch)
      #print(X_gan.shape)

      # The generator wants the discriminator to label the generated samples
      # as valid (ones)
      # This is where the generator is trying to trick discriminator into believing
      # the generated image is true (hence value of 1 for y)
      y_gan = ones((n_batch, 1))

      # Generator is part of combined model where it got directly linked with the discriminator
      # Train the generator with latent_dim as x and 1 as y.
      # Again, 1 as the output as it is adversarial and if generator did a great
      # job of folling the discriminator then the output would be 1 (true)
			# update the generator via the discriminator's error
      g_loss = gan_model.train_on_batch(X_gan, y_gan)

      # Print losses on this batch
      print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
       (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))

      #if i % 20 == 0:
       #     prediction_model(i)
      #Gen_Loss.append(g_loss)

# save the generator model
  with open('/Generators/GAN_adload_generator.pkl', 'wb') as file:
    pickle.dump(g_model, file)

# size of the latent space
latent_dim = 100

# create the discriminator
discriminator = define_discriminator()

# create the generator
generator = define_generator(latent_dim)

# create the gan
gan_model = define_gan(generator, discriminator)

# load image data
dataset = load_real_samples()

# train model
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs = 800)

#              G E N E R A T I N G  S A M P L E S

# Loading the generator 
generator = load_model('/GAN/Models/cifar[EG_Gatak]_generator_epochs.h5')

# Generating images by passing noise into the trained Generator
noise = np.random.normal(0, 1, (1 * 1, 100))
gen_imgs = generator.predict(noise)

for i in range(len(gen_imgs)):
  prediction = loaded_model.predict(gen_imgs[i].reshape(1, 128, 128, 3))
  

# Create a figure with adjusted size
plt.figure(figsize=(gen_imgs[i].shape[1] / 100, gen_imgs[i].shape[0] / 100), dpi=100)
# Plot the image without padding
plt.imshow(gen_imgs[i])
plt.axis('off')
# Save the image without extra white space
plt.savefig(f"/Sh_Gatak_Sample.png", bbox_inches='tight', pad_inches=0)
# Close the plot
plt.close()

