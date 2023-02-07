import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def decode_df(decoding, encoded):
    def lookup_encoding(col_name, value):
        encoding = decoding.loc[(decoding['column_name'] == col_name) & (decoding['decoding'] == value.iloc[0]), 'encoding'].values
        return encoding[0] if encoding.size > 0 else value
    
    return encoded.apply(lambda x: lookup_encoding(x.name, x) if x.name in decoding['column_name'].unique() else x)

# Import one hot encoded data and decoding dictionary from label encoding step
ohe_card_df = pd.read_pickle('ohe_card_df.pkl')
decoding_dict = pd.read_pickle('decoding_dict.pkl')

# Split data into train and test sets
train_df, test_df = train_test_split(ohe_card_df, test_size=0.2, random_state=42)

# Get the number of columns in the one hot encoded dataframe
n_cols = train_df.shape[1]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(n_cols,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_cols)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_df, train_df, epochs=10)

# Predict the values
predictions = model.predict(test_df)
print(predictions.shape)
print(predictions)

# Set up prediction dataframe, round values to integers, and decode predictions
prediction_df = pd.DataFrame(predictions, columns=ohe_card_df.columns)
prediction_df = prediction_df.apply(lambda x: round(x)).astype(int)
decoded_predictions = decode_df(decoding_dict, prediction_df)
print(decoded_predictions)

# Save predicted values and network
pd.to_pickle(prediction_df,'prediction_df.pkl')
model.save('model.h5')