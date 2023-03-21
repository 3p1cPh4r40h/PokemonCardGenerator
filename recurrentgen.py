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

#getting the data from the pickle file
with open("test_data/label_encoded_df.pkl","rb") as l:
    df = pickle.load(l)

x = np.array(df,dtype=object)
x = np.asarray(x).astype('float32')

#preparing the test and train sets with the test set being 30% of the data
train, test = train_test_split(x, test_size=0.3, random_state=42)


# Reshape the data to fit the input shape of the SimpleRNN layer
steps = 1
features = train.shape[1] -1
train_x = train[:,:-1].reshape((train.shape[0], steps, features))
train_y = train[:,-1]
test_x = test[:,:-1].reshape((test.shape[0], steps, features))
test_y = test[:,-1]



#-----------------------------------
#          Model Creation
#-----------------------------------

model = Sequential()

model.add(SimpleRNN(32,input_shape=(steps,features),activation="relu"))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.summary()
#-----------------------------------
#           Training
#-----------------------------------


model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data=(test_x,test_y))

# Evaluate the model
loss, acc = model.evaluate(test_x,test_y)
print("Test loss: ", loss)
print("Test accuracy: ", acc)
