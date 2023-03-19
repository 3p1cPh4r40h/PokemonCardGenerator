from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------
#           Data Preperation
#-----------------------------------

#getting the data from the pickle files
with open("test_data/oh_encoded_df.pkl","rb") as k:
    ohdf = pickle.load(k)
with open("test_data/label_encoded_df.pkl","rb") as l:
    labeldf = pickle.load(l)

x = np.array(ohdf)
y = np.array(labeldf)

#preparing the test and train sets with the test set being 30% of the data
traindata, testdata, trainlabel, testlabel = train_test_split(x,y, test_size=0.3, random_state=42)


# Reshape the data to fit the input shape of the SimpleRNN layer
steps = 4
features = traindata.shape[1]
traindata = traindata.reshape((traindata.shape[0], steps, features))
testdata = testdata.reshape((testdata.shape[0], steps, features))


#-----------------------------------
#          Model Creation
#-----------------------------------

model = Sequential()

model.add(SimpleRNN(32,input_shape=(steps, features),activation="relu"))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
#-----------------------------------
#           Training
#-----------------------------------


model.fit(traindata,trainlabel, epochs=50, batch_size=32, validation_data=(testdata, testlabel))

# Evaluate the model
loss, acc = model.evaluate(testdata,testlabel)
print("Test loss: ", loss)
print("Test accuracy: ", acc)
