import pandas as pd
import tensorflow as tf
import numpy as np
import medmnist
import platform
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from medmnist import PneumoniaMNIST
from medmnist import PathMNIST

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D

#Import the Datasets
dataX =np.load('Datasets/pneumoniamnist.npz')

dataI =np.load('Datasets/pathmnist.npz')


#Extract the Data
#---pneumoniamnist
Xray = dataX.files

x_train=np.array(dataX[Xray[0]])
x_val=np.array(dataX[Xray[1]])
x_test=np.array(dataX[Xray[2]])

y_train=np.array(dataX[Xray[3]])
y_val=np.array(dataX[Xray[4]])
y_test=np.array(dataX[Xray[5]])

#---pathmnist
Image = dataI.files

X_train=np.array(dataI[Image[0]])
X_val=np.array(dataI[Image[1]])
X_test=np.array(dataI[Image[2]])

Y_train=np.array(dataI[Image[3]])
Y_val=np.array(dataI[Image[4]])
Y_test=np.array(dataI[Image[5]])


#Pre-process Data
#---pneumoniamnist
x = np.concatenate((x_train, x_val, x_test), axis=0)
y = np.concatenate((y_train, y_val, y_test), axis=0)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15) #split the data with test size 15%

#---pathmnist
X = np.concatenate((X_train, X_val, X_test), axis=0)
Y = np.concatenate((Y_train, Y_val, Y_test), axis=0)

#convert categorical data to multidimensional binary vectors
enc = OneHotEncoder()
Y=enc.fit_transform(Y).toarray()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15)


#Import the pre-trained models
#---pneumoniamnist
binary_classifier_model = 'A/binary_classifier_model.pkl' # Path to the pickle file

with open(binary_classifier_model, 'rb') as file:
    modelBinary = pickle.load(file)

#---pathmnist
multi_classifier_model = 'B/multi_classifier_model.pkl'

with open(multi_classifier_model, 'rb') as file:
    modelMulti = pickle.load(file)



# Use Models to predict each test set
#---pneumoniamnist
ypred = modelBinary.predict(x_test)
#---pathmnist
Ypred = modelMulti.predict(X_test)




# Classification Report for each
#---pneumoniamnist
binary_predictions = np.where(ypred > 0.5, 1, 0)
print("binary predictions (pneumoniamnist) classification report")
print(classification_report(y_test, binary_predictions))

#---pathmnist
multi_predictions = np.where(Ypred > 0.5, 1, 0)
print("multi predictions (pathmnist) classification report")
print(classification_report(Y_test, multi_predictions, zero_division=0))


#Create a Confusion Matrix for multi-class CNN
# Convert probabilities to predicted class labels
y_pred_classes = np.argmax(Ypred, axis=1)

# Convert one-hot encoded y_test to class labels
y_test_classes = np.argmax(Y_test, axis=1) if Y_test.shape[1] > 1 else Y_test.squeeze()

# Compute the confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for CNN Model')
plt.show()


