import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def decode_df(decoding, encoded):
    def lookup_encoding(col_name, value):
        encoding = decoding.loc[(decoding['column_name'] == col_name) & (decoding['decoding'] == value.iloc[0]), 'encoding'].values
        return encoding[0] if encoding.size > 0 else value
    
    return encoded.apply(lambda x: lookup_encoding(x.name, x) if x.name in decoding['column_name'].unique() else x)

# Import one hot encoded data and decoding dictionary from label encoding step
ohe_card_df = pd.read_pickle('data\ohe_card_df.pkl')
decoding_dict = pd.read_pickle('data\decoding_dict.pkl')

# Split data into train and test sets
train_df, test_df = train_test_split(ohe_card_df, test_size=0.2, random_state=42)

# Reshape the data into 3D tensors for use in the CNN
train_data = np.reshape(train_df.values, (train_df.shape[0], train_df.shape[1], 1))
test_data = np.reshape(test_df.values, (test_df.shape[0], test_df.shape[1], 1))

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_data.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(ohe_card_df.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
train_labels = np.argmax(train_data, axis=1)
train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=ohe_card_df.shape[1])
model.fit(train_data, train_labels_cat, epochs=10)

# Predict the values
predictions = model.predict(test_data)
print(predictions.shape)
print(predictions)

# Decode predictions
decoded_predictions = decode_df(decoding_dict, pd.DataFrame(predictions, columns=ohe_card_df.columns))
print(decoded_predictions)

# Save predicted values and network
pd.to_pickle(decoded_predictions, 'data\decoded_predictions_cnn.pkl')
model.save('model_cnn.h5')

	
print('Accuracy: %.3f' % accuracy_score(test_data, predictions))
