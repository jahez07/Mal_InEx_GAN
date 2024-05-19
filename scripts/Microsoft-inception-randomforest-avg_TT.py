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

#              D A T A  S E L E C T I O N  

predictions = rf_model.predict(train_features)
predictions = le.inverse_transform(predictions)

# Path to SelectedData Directory
from keras.preprocessing.image import array_to_img
path = os.path.join("/content/drive/MyDrive/Jahez_Vinod_2023/MalHub/SelectedData") # Path to selectedData directory
count = 0
folders = list_fams
x_train = []
y_train = []

for i in range(len(predictions)):
    pred_class_index = predictions[i]
    pred_class = list_fams[int(pred_class_index)] # Getting the class name
    
    if pred_class == list_fams[int(labels_train)]:
      x_train.append(images_train[i])
      y_train.append(pred_class)
      dir_path = os.path.join(path, pred_class) 
      if not os.path.exists(dir_path): # Checking for existence of a directory
          os.makedirs(dir_path) # Creating directory
      img = array_to_img(images_train[i]) # Converting numpy array to image
      img.save(f"{path}/{pred_class}/Train({pred_class})_{i}.png") # Saving the converted image
      count += 1
print(count)

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