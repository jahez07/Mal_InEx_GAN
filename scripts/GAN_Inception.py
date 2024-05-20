#              N E C E S S A R Y  P A C K A G E S

import numpy as np
np.random.seed(1)

import os
import os.path
import glob
import tensorflow as tf
import pandas as pd
import time
import csv
from numpy.random import randint, randn
from numpy import ones, zeros

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold 
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split                                                                  
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,precision_score, recall_score
from sklearn import metrics
from sklearn.manifold import TSNE
from PIL import Image
from keras.models import load_model


import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input


imagedir = "Malhub"

cur_dir = os.getcwd()
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

# Compute the labels
y = np.zeros(num_samples)
pos = 0
label = 0
for i in no_imgs:
    print ("Label:%2d\tFamily: %15s\tNumber of images: %d" % (label, list_fams[label], i))
    for j in range(i):
        y[pos] = label
        pos += 1
    label += 1
num_classes = label

#              P R O C E S S I N G  I M A G E S

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
#print(images)
#print(labels)

#              E N C O D E R  &  S P L I T

le = preprocessing.LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)
unique_labels_encoded = np.unique(labels_encoded)

# Splitting to train & test
images_train, images_test, labels_train, labels_test = train_test_split(images, labels_encoded, test_size=0.20, random_state=42)

# One-hot encoding
y_train_one_hot = to_categorical(labels_train)
y_test_one_hot = to_categorical(labels_test)

image_shape = (128,128, 3)

# Assigning the base model
base_model = InceptionV3(weights='imagenet', input_shape=image_shape, include_top=False, pooling='avg')
start_time = time.time()

#              F E A T U R E  E X T R A C T I O N

# Creating a feature extractor using the base model
feature_extractor = base_model.predict(images_train)

# Extracting features of the train images using the base model (InceptionV3)
train_features = feature_extractor.reshape(feature_extractor.shape[0], -1)


rf_model = RandomForestClassifier()
rf_model.fit(train_features, labels_train)

test_feature = base_model.predict(images_test)
test_features = test_feature.reshape(test_feature.shape[0], -1)

prediction_rf = rf_model.predict(test_features)
#print(prediction_svm)
prediction_rf = le.inverse_transform(prediction_rf)
#print(prediction_svm)
end_time = time.time()
execution_time = end_time - start_time

#              D A T A  L O A D I N G  


# path to one family
data = os.path.join("/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/SelectedData/adload")
images = []
for file_ in os.listdir(data):
  img_path = os.path.join(data, file_)
  with Image.open(img_path) as img:
    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = img_array[:, :, :3]
    images.append(img_array)

# Convert list of images to array
X_train = np.array(images)

#              E X E C U T I O N  R E S U L T S  B E F O R E  A T T A C K

print("Execution time:", execution_time, "seconds")

print("Accuracy SVM= ", metrics.accuracy_score(labels_test, prediction_rf))

print(classification_report(labels_test, prediction_rf))


f1_micro = f1_score(labels_test, prediction_rf, average='micro')
print("Micro-Averaged F1 Score:", f1_micro)

f1_macro = f1_score(labels_test, prediction_rf, average='macro')
print("Macro-Averaged F1 Score:", f1_macro)

precision_micro = precision_score(labels_test, prediction_rf, average='micro')
print("Micro-Averaged Precision:", precision_micro)

precision_macro = precision_score(labels_test, prediction_rf, average='macro')
print("Macro-Averaged Precision:", precision_macro)

recall_micro = recall_score(labels_test, prediction_rf, average='micro')
print("Micro-Averaged Recall:", recall_micro)

recall_macro = recall_score(labels_test, prediction_rf, average='macro')
print("Macro-Averaged Recall:", recall_macro)

csv_data = [
    ["Metric", "Value"],
    ["Execution Time (seconds)", execution_time],
    ["Accuracy SVM", metrics.accuracy_score(labels_test, prediction_rf)],
    ["Micro-Averaged F1 Score", f1_micro],
    ["Macro-Averaged F1 Score", f1_macro],
    ["Micro-Averaged Precision", precision_micro],
    ["Macro-Averaged Precision", precision_macro],
    ["Micro-Averaged Recall", recall_micro],
    ["Macro-Averaged Recall", recall_macro],
    [classification_report(labels_test, prediction_rf)]

]
with open('INCEP-rf-avg.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)

print("Plotting the confusion matrix")
cm = confusion_matrix(labels_test, prediction_rf)
cm = cm.T  # since rows and cols are interchangeable
# Compute confusion matrix with normalized values between 0 and 1
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix with normalized values
plt.figure(figsize=(24, 18))
plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
for row in range(len(list_fams)):
    for col in range(len(list_fams)):
        text_color = 'white' if row == col else 'black'
        plt.annotate(format(cm_normalized[row, col], ".2f"), xy=(col, row), ha='center', va='center', color=text_color)
plt.xticks(range(len(list_fams)), list_fams, rotation=90, fontsize=10)
plt.yticks(range(len(list_fams)), list_fams, fontsize=10)
plt.title('Normalized Confusion Matrix')
plt.colorbar()

# Save the confusion matrix as a PNG image
plt.savefig('Inception_rf_avg_confusion_matrix_tt.png', bbox_inches='tight', format='png')


#             G E N E R A T I V E  A D V E R S R I A L  N E T W O R K 

from keras.models import Sequential
from keras.layers import Dense, Reshape, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam

# Discriminator Model 
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

# Generator Model

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

# Latent Points Generation

# Generate n_samples number of latent vectors as input for the generator

def generate_latent_points(latent_dim, n_samples):

  # generate points in the latent space
	x_input = randn(latent_dim * n_samples)

  # reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# Fake Sample Generation

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

# save the generator model
  g_model.save('/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/Models/GAN[adload]_generator.h5')

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
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs = 10)

generator = load_model('/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/Models/GAN[adload]_generator.h5')

#              G E N E R A T I N G  A D V E R S A R I A L  S A M P L E S

noise = np.random.normal(0, 1, (1 * 1, 100))
gen_imgs = generator.predict(noise)
