#!/usr/bin/env python
# coding: utf-8

# # DIABETIC RETINOPATHY DETECTION

# # STEP #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE

# Data Source: https://www.kaggle.com/c/diabetic-retinopathy-detection

# ### Difference Between Normal retina and Diabetic Retina
# 
# 
# ![image.png](attachment:image.png)
# 

# ### Severity level of Diabetic Retinopathy.
# ![image.png](attachment:image.png)

# #### Reference for the images
# https://raw.githubusercontent.com/dimitreOliveira/MachineLearning/master/Kaggle/APTOS%202019%20Blindness%20Detection/aux_img.png

# # STEP #2: IMPORT LIBRARIES/DATASETS

# In[1]:


# Import the necessary packages

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler


# In[2]:


# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 


# In[3]:


#listing the classes for the training data
os.listdir('./train')


# In[4]:


#The os.path.join() function is used to construct a file path by combining the 'train' and 'Mild' directories. 
#This ensures that the file path is correctly formatted for the operating system.
os.listdir(os.path.join('train', 'Mild'))


# In[5]:


# Check the number of images in the dataset
train = []
label = []

# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir('./train'):
  train_class = os.listdir(os.path.join('train', i))
  for j in train_class:
    img = os.path.join('train', i, j)
    train.append(img)
    label.append(i)

print('Number of train images : {} \n'.format(len(train)))


# In[6]:


train


# In[7]:


label


# # STEP #3: PERFORM DATA EXPLORATION AND DATA VISUALIZATION

# In[32]:


# Visualize 5 images for each class in the dataset

fig, axs = plt.subplots(5, 5, figsize = (20, 20))
count = 0
for i in os.listdir('./train'):
  # get the list of images in a given class
  train_class = os.listdir(os.path.join('train', i))
  # plot 5 images per class
  for j in range(5):
    img = os.path.join('train', i, train_class[j])
    img = PIL.Image.open(img)
    axs[count][j].title.set_text(i)
    axs[count][j].imshow(img)  
  count += 1

fig.tight_layout()


# In[9]:


# check the number of images in each class in the training dataset

No_images_per_class = []
Class_name = []
for i in os.listdir('./train'):
  train_class = os.listdir(os.path.join('train', i))
  No_images_per_class.append(len(train_class))
  Class_name.append(i)
  print('Number of images in {} = {} \n'.format(i, len(train_class)))


# In[10]:


#printing the image name and its corresponding label
retina_df = pd.DataFrame({'Image': train,'Labels': label})
retina_df


# In[33]:


#plotting the classified dataset
sns.countplot(x=label)


# ### Notable Obeservations
# The ‘Moderate’ category has the highest count among those with some level of DR severity.
# The ‘No_DR’ category significantly outnumbers the others, indicating a larger number of cases without DR.

# # STEP #4: PERFORM DATA AUGMENTATION AND CREATE DATA GENERATOR

# In[11]:


# Shuffle the data and split it into training and testing
retina_df = shuffle(retina_df)
train, test = train_test_split(retina_df, test_size = 0.2)


# In[12]:


# Create run-time augmentation on training and test dataset
# For training datagenerator, we add normalization, shear angle, zooming range and horizontal flip
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        validation_split = 0.15)

# For test datagenerator, we only normalize the data.
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[13]:


# Creating datagenerator for training, validation and test dataset.

train_generator = train_datagen.flow_from_dataframe(
    train,
    directory='./',
    x_col="Image",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    subset='training')

validation_generator = train_datagen.flow_from_dataframe(
    train,
    directory='./',
    x_col="Image",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    subset='validation')

test_generator = test_datagen.flow_from_dataframe(
    test,
    directory='./',
    x_col="Image",
    y_col="Labels",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32)


# # STEP #5: UNDERSTAND THE THEORY AND INTUITION BEHIND CONVOLUTIONAL NEURAL NETWORKS (CNN) AND RESIDUAL BLOCKS

# ### Deep Neural Network Model:
# The neural network architecture is based on residual blocks (often denoted as “RES-BLOCKS”).
# These blocks allow the model to learn hierarchical features from the input images.
# The network processes the input image through multiple layers, extracting relevant information.
# ### Training Process:
# The neural network is trained using a dataset of 3553 color images.
# During training, the model learns to associate specific features in the images with the severity levels.
# It adjusts its internal parameters (weights and biases) to minimize the classification error.
# ### Classification:
# Once trained, the model can take any new eye image as input.
# It computes a probability distribution over the severity categories.
# The category with the highest probability becomes the predicted severity level for that image.
# 

# ![image.png](attachment:image.png)

# ### Convolutional Neural Networks (CNNs):
# CNNs are a class of deep neural networks specifically designed for processing grid-like data, such as images.
# They excel at capturing spatial hierarchies and patterns within images.
# CNNs are widely used in computer vision tasks, including image classification, object detection, and segmentation.
# Processing Stages in the Image:
# The image illustrates the stages of a CNN processing an input image (likely a retinal scan) for diabetic retinopathy severity classification.
# Let’s break down the components:
# ### Convolutional Layer:
# The left side of the image shows an eye labeled as “CONVOLUTION.”
# This represents the initial stage where the input image undergoes convolution.
# Convolutional kernels (also known as feature detectors) slide over the image, extracting local features.
# These features capture edges, textures, and other relevant patterns.
# ### Pooling Layer:
# Next, the image passes through a pooling layer (indicated by blue rectangles labeled “POOLING FILTERS”).
# Pooling reduces the spatial dimensions of the feature maps while retaining essential information.
# Common pooling methods include max-pooling and average-pooling.
# ### Flattening:
# After convolution and pooling, the feature maps are flattened into a 1D vector.
# This step prepares the data for input into a traditional neural network.
# Neural Network Architecture:
# The flattened features feed into a neural network with layers labeled as “input,” “hidden,” and “output.”
# The neural network learns to associate the extracted features with different severity levels.
# ### Severity Classification:
# The output layer has five nodes, each corresponding to a severity category:
# NO_DR: No diabetic retinopathy
# MILD: Mild disease
# MODERATE: Moderate severity
# SEVERE: Severe retinopathy
# PROLIFERATIVE_E_DR: Rapidly growing disease
# ### Clinical Significance:
# Accurate classification helps clinicians identify the severity of diabetic retinopathy.
# Early detection allows timely intervention to prevent vision loss.

# ![image.png](attachment:image.png)

# ### Residual Networks (ResNets):
# Residual Networks, commonly known as ResNets, are a type of Convolutional Neural Network (CNN) architecture.
# They were introduced to address the vanishing gradient problem in deep neural networks.
# The vanishing gradient problem occurs when gradients become too small during backpropagation, making it challenging for deep networks to learn effectively.
# ResNets mitigate this issue by introducing skip connections (also called identity mappings).
# ## Key Concepts:
# 
# ### Vanishing Gradient Problem:
# As neural networks grow deeper (more layers), gradients can become extremely small during backpropagation.
# Small gradients hinder weight updates, leading to poor convergence and learning.
# Skip Connections (Identity Mappings):
# ResNets include skip connections that allow gradients to flow directly through the network.
# Instead of learning the difference between the input and output (as in traditional networks), ResNets learn the residual (the difference between the input and output).
# The residual is added back to the input, allowing gradients to propagate more effectively.
# ### Deep Architectures:
# ResNets can be very deep (e.g., ResNet-50, ResNet-101, or even ResNet-152).
# Despite their depth, they maintain good performance due to the skip connections.
# ### Benefits of ResNets:
# 
# #### Increased Depth:
# ResNets can be trained with hundreds of layers without suffering from vanishing gradients.
# This depth allows them to capture intricate features and patterns.
# #### Improved Accuracy:
# Skip connections enable better feature learning, leading to improved accuracy on various tasks.
# #### Transfer Learning:
# Pre-trained ResNets (trained on large datasets like ImageNet) serve as excellent feature extractors for other tasks.
# Fine-tuning a pre-trained ResNet on a specific dataset can yield impressive results.
# #### Training and ImageNet:
# ResNets are often trained on large-scale datasets like ImageNet, which contains millions of labeled images across thousands of categories.
# ImageNet-trained ResNets learn general features that can be fine-tuned for specific tasks

# ![image.png](attachment:image.png)

# ### ResNet Block:
# The left section of the image illustrates a ResNet block:
# It includes an “INPUT,” a “CONVOLUTION BLOCK,” two “IDENTITY BLOCKS,” and an “OUTPUT.”
# Skip connections allow gradients to propagate effectively.
# ### ResNet-18 Model:
# The right section shows the architecture of a complete ResNet-18 model:
# Layers include zero padding, Conv2D, BatchNorm, ReLU activation, MaxPool2D, multiple residual blocks (RES-BLOCK), average pooling, flattening, and a dense layer with softmax activation for classification.

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # STEP #6: BUILD RES-BLOCK BASED DEEP LEARNING MODEL

# In[14]:


def res_block(X, filter, stage):
  
  # Convolutional_block
  X_copy = X

  f1 , f2, f3 = filter
    
  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


  # Short path
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 1
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 2
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X


# In[15]:


input_shape = (256,256,3)

#Input tensor shape
X_input = Input(input_shape)

#Zero-padding

X = ZeroPadding2D((3,3))(X_input)

# 1 - stage

X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2- stage

X = res_block(X, filter= [64,64,256], stage= 2)

# 3- stage

X = res_block(X, filter= [128,128,512], stage= 3)

# 4- stage

X = res_block(X, filter= [256,256,1024], stage= 4)

# # 5- stage

# X = res_block(X, filter= [512,512,2048], stage= 5)

#Average Pooling

X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

#Final layer

X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)


model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model.summary()


# # STEP #7: COMPILE AND TRAIN DEEP LEARNING MODEL

# In[16]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])


# In[19]:


#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="weights.keras", verbose=1, save_best_only=True)


# In[22]:


retina_weights= model.fit(train_generator, steps_per_epoch = train_generator.n // 32, epochs = 10, validation_data= validation_generator, validation_steps= validation_generator.n // 32, callbacks=[checkpointer , earlystopping])


# In[23]:


##creates a line plot that displays the training and validation loss values during the training of a machine learning model.
plt.plot(retina_weights.history['loss'])
plt.plot(retina_weights.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


# # TASK #8: ASSESS THE PERFORMANCE OF THE TRAINED MODEL

# In[24]:


#pretrained model which was run by 20 epochs on a seperate file
model.load_weights("retina_weights.hdf5")


# In[25]:


# Evaluate the performance of the model
evaluate = model.evaluate(test_generator, steps = test_generator.n // 32, verbose =1)

print('Accuracy Test : {}'.format(evaluate[1]))


# In[26]:


# Assigning label names to the corresponding indexes
labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}


# In[27]:


# Loading images and their predictions 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import cv2

prediction = []
original = []
image = []
count = 0

for item in range(len(test)):
  # code to open the image
  img= PIL.Image.open(test['Image'].tolist()[item])
  # resizing the image to (256,256)
  img = img.resize((256,256))
  # appending image to the image list
  image.append(img)
  # converting image to array
  img = np.asarray(img, dtype= np.float32)
  # normalizing the image
  img = img / 255
  # reshaping the image in to a 4D array
  img = img.reshape(-1,256,256,3)
  # making prediction of the model
  predict = model.predict(img)
  # getting the index corresponding to the highest value in the prediction
  predict = np.argmax(predict)
  # appending the predicted class to the list
  prediction.append(labels[predict])
  # appending original class to the list
  original.append(test['Labels'].tolist()[item])


# In[28]:


# Getting the test accuracy 
score = accuracy_score(original, prediction)
print("Test Accuracy : {}".format(score))


# # Accuracy of the model is: 82.12%

# In[29]:


# Visualizing the results
import random
fig=plt.figure(figsize = (100,100))
for i in range(20):
    j = random.randint(0,len(image))
    fig.add_subplot(20, 1, i+1)
    plt.xlabel("Prediction: " + prediction[j] +"   Original: " + original[j])
    plt.imshow(image[j])
fig.tight_layout()
plt.show()


# In[30]:


# Print out the classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))


# In[34]:


# plot the confusion matrix
plt.figure(figsize = (20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')


# ![image.png](attachment:image.png)

# In[ ]:




