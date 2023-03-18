from keras.models import Sequential
from keras.models import LSTM, Dense, Masking, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------
#           Data Preperation
#-----------------------------------

#getting the data from the pickle file
with open("card_df.pkl", "") as k:
    df = pickle.load(k)

#converting the data from strings into numeric values
enc = LabelEncoder()
df['name'] = enc.fit_transform(df['name'])

#standardizing the integers
sca = StandardScaler()
df['hp'] = sca.fit_transform(df['hp'])

#preparing the test and train sets
trainsize = int(len(df)*0.7)
traindata = df.iloc[:trainsize]
testdata =  df.iloc[trainsize:]

#-----------------------------------
#          Model Creation
#-----------------------------------
model = Sequential()

model.add(LSTM(64,input_shape=(None, 10)))
model.add(Dense(1,activation='sigmoid'))

#todo
model.compile()