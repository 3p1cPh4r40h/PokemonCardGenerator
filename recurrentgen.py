from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Dropout
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
with open("data/card_df.pkl","rb") as k:
    df = pickle.load(k)

#converting the data from strings into numeric values
enc = LabelEncoder()
df['name'] = enc.fit_transform(df['name'])

#standardizing the integers
sca = StandardScaler()
#df['hp'] = sca.fit_transform(df['hp'])

#preparing the test and train sets with the test set bing 30% of the data
traindata, testdata = train_test_split(df, test_size=0.3, random_state=42)

#-----------------------------------
#          Model Creation
#-----------------------------------

model = Sequential()

model.add(LSTM(64,input_shape=(None, 10)))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#-----------------------------------
#           Training
#-----------------------------------

def prepdata(data):
    x, y = [], []
    for i in range(len(data)-1):
        x.append(data.iloc[i:i+1,1:].values)
        y.append(data.iloc[i+1,0])
    return np.asarray(x), np.asarray(y)

train_x, train_y = prepdata(traindata)

model.fit(train_x,train_y,epochs=50,batch_size=32)

test_x, test_y = prepdata(testdata)

loss, acc = model.evaluate(test_x,test_y)
print("Accuracy: ", acc)
