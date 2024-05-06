import numpy as np
np.random.seed(1)

import os
import os.path
import glob
import tensorflow as tf
import pandas as pd
import time
import sklearn


from sklearn.model_selection import StratifiedKFold   
from sklearn.ensemble import RandomForestClassifier                                                                                                                    
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score, recall_score


import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception,preprocess_input

accuracy_list = []
f1_score_micro_list = []
f1_score_macro_list = []
recall_micro_list = []
recall_macro_list = []
precision_micro_list = []
precision_macro_list = []

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

# Compute the features
width, height,channels = (224,224,3)
X = np.zeros((num_samples, width, height, channels))
cnt = 0
list_paths = [] # List of image paths
print("Processing images ...")
for i in range(len(list_fams)):
    for img_file in glob.glob(list_fams[i]+'/*.png'):
        #print("[%d] Processing image: %s" % (cnt, img_file))
        list_paths.append(os.path.join(os.getcwd(),img_file))
        img = image.load_img(img_file, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X[cnt] = x
        cnt += 1
print("Images processed: %d" %(cnt))

os.chdir(cur_dir)
print("shape:",X.shape)

image_shape = (224,224, 3)    
base_model = Xception(weights='imagenet', input_shape=image_shape, include_top=False ,pooling='avg')


filename = 'Microsoft-xceptionfeatures-avgpool.npy'
if os.path.exists(filename):
    print("Loading Xception extracted features from %s ..." %(filename))
    xceptionfeatures = np.load(filename)
else:
    print("Extracting features from Xception layers ...")
    xceptionfeatures = base_model.predict(X)
    print("Saving Xception extracted features into %s ..." %(filename))
    np.save(filename, xceptionfeatures)
    
    
print("xceptionfeatures_shape:",xceptionfeatures.shape)
xceptionfeatures = np.reshape(xceptionfeatures,(xceptionfeatures.shape[0],-1))
print("xceptionfeatures_shape:",xceptionfeatures.shape)

# Create stratified k-fold subsets                                                                                                                                        
kfold = 5  # no. of folds                                                                 
skf = StratifiedKFold(kfold, shuffle=True,random_state=1)
skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
cnt = 0                                              
for index in skf.split(X, y):         
    skfind[cnt] = index                                                 
    cnt += 1 
    
    
# Training top_model and saving min training loss weights
conf_mat = np.zeros((len(list_fams),len(list_fams))) # Initializing the Confusion Matrix
start_time = time.time()
for i in range(kfold):
    train_indices = skfind[i][0]
    test_indices = skfind[i][1]
    X_train = xceptionfeatures[train_indices]
    y_train = y[train_indices]
    X_test = xceptionfeatures[test_indices]
    y_test = y[test_indices]
    
    top_model = RandomForestClassifier()
    top_model.fit(X_train,y_train)  # Training
    y_pred = top_model.predict(X_test)  # Testing
    print("[%d] Test acurracy: %.4f" %(i,accuracy_score(y_test,y_pred)))
    accuracy = accuracy_score(y_test, y_pred)

    f1_micro = f1_score(y_test,y_pred, average='micro')
    print("Micro-Averaged F1 Score:", f1_micro)

    f1_macro = f1_score(y_test,y_pred, average='macro')
    print("Macro-Averaged F1 Score:", f1_macro)

    precision_micro = precision_score(y_test,y_pred, average='micro')
    print("Micro-Averaged Precision:", precision_micro)

    precision_macro = precision_score(y_test,y_pred, average='macro')
    print("Macro-Averaged Precision:", precision_macro)

    recall_micro = recall_score(y_test,y_pred, average='micro')
    print("Micro-Averaged Recall:", recall_micro)

    recall_macro = recall_score(y_test,y_pred, average='macro')
    print("Macro-Averaged Recall:", recall_macro)

    cm = confusion_matrix(y_test,y_pred)  # Compute confusion matrix for this fold
    conf_mat = conf_mat + cm  # Compute global confusion matrix
    

    accuracy_list.append(accuracy)
    f1_score_micro_list.append(f1_micro)
    f1_score_macro_list.append(f1_macro)
    recall_micro_list.append(recall_micro)
    recall_macro_list.append(recall_macro)
    precision_micro_list.append(precision_micro)
    precision_macro_list.append(precision_macro)

end_time = time.time()
execution_time = end_time - start_time   
metrics_df = pd.DataFrame({
    'Accuracy': accuracy_list,
    'F1 Score micro': f1_score_micro_list,
    'F1 Score macro': f1_score_macro_list,
    'Precision micro': precision_micro_list,
    'Precision macro': precision_macro_list,
    'Recall micro': recall_micro_list,
    'Recall macro': recall_macro_list
})

    
avg_acc = np.trace(conf_mat)/sum(no_imgs)
print("Average acurracy: %.4f" %(avg_acc))



conf_mat = conf_mat.T  # since rows and cols are interchangeable
conf_mat_norm = conf_mat/no_imgs


print("Plotting the confusion matrix")
conf_mat = np.around(conf_mat_norm,decimals=2)  # rounding to display in figure
figure = plt.gcf()
figure.set_size_inches(24, 18)
plt.imshow(conf_mat,interpolation='nearest', cmap='Blues')
for row in range(len(list_fams)):
    for col in range(len(list_fams)):
        text_color = 'white' if row == col else 'black'
        plt.annotate(str(conf_mat[row][col]),xy=(col,row),ha='center',va='center',color=text_color)
plt.xticks(range(len(list_fams)),list_fams,rotation=90,fontsize=10)
plt.yticks(range(len(list_fams)),list_fams,fontsize=10)
plt.title('Confusion matrix')
plt.colorbar()
# Save the confusion matrix as a PNG image
plt.savefig('random_forest_avg_confusion_matrix.png', bbox_inches='tight', format='png')

#plt.show()

metrics_df.to_csv('random_forest_avg.csv', index=False)
pd.DataFrame({'Average Accuracy': [avg_acc]}).to_csv('random_forest_avg.csv', mode='a', index=False, header=True)
pd.DataFrame({'Execution time': [execution_time]}).to_csv('random_forest_avg.csv', mode='a', index=False, header=True)

