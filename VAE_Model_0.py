import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

# Read label encoded dataframe from pickle file
label_encoded_df = pd.read_pickle('data\label_encoded_df.pkl')

# Normalize data between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(label_encoded_df)
label_encoded_df = pd.DataFrame(scaled_data, columns=label_encoded_df.columns)

# Split data into train and test sets
train_df, test_df = train_test_split(label_encoded_df, test_size=0.2, random_state=42)

# Drop column names from train and test dataframes
train_array = train_df.values
test_array = test_df.values

# Drop ID column from the array
train_array = train_array[:, 1:]
test_array = test_array[:, 1:]

# create a new array with shape (2, 5, 6)
reshaped_train_data = np.zeros((train_array.shape[0], 5, 6))

for i in range(reshaped_train_data.shape[0]):
    for j in range(reshaped_train_data.shape[1]):
        for k in range(reshaped_train_data.shape[2]):
            if j == 0:
                reshaped_train_data[i,j,k] = int(train_array[i, k])
            else:
                reshaped_train_data[i,j,k] = int(train_array[i, j+5])

# create a new array with shape (2, 5, 6)
reshaped_test_data = np.zeros((test_array.shape[0], 5, 6))

for i in range(reshaped_test_data.shape[0]):
    for j in range(reshaped_test_data.shape[1]):
        for k in range(reshaped_test_data.shape[2]):
            if j == 0:
                reshaped_test_data[i,j,k] = int(test_array[i, k])
            else:
                reshaped_test_data[i,j,k] = int(test_array[i, j+5])

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(16, 3, activation='relu', padding='same', input_shape=(5,6)),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim),
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(5*6, activation='relu', input_shape=(latent_dim,)),
            layers.Reshape((5, 6)),
            layers.Conv1DTranspose(16, 3, activation='relu', padding='same'),
            layers.Conv1DTranspose(1, 3, activation='sigmoid', padding='same'),
        ])
        
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar

# Load your data here
train_data = reshaped_train_data

# Define the loss function
def vae_loss(x, x_recon, mean, logvar):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_recon))
    kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_divergence

# Create an instance of the VAE model
latent_dim = 2
vae = VAE(latent_dim)

# Define the optimizer and the batch size
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
batch_size = 32

# Train the model
epochs = 10
for epoch in range(epochs):
    print('Epoch:', epoch+1)
    for step in range(train_data.shape[0] // batch_size):
        x = train_data[step*batch_size : (step+1)*batch_size]
        with tf.GradientTape() as tape:
            x_recon, mean, logvar = vae(x)
            loss = vae_loss(x, x_recon, mean, logvar)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        if step % 100 == 0:
            print('Step:', step, 'Loss:', loss.numpy())


# Load the test data
test_data = reshaped_test_data

# Encode the test data and get the mean and logvar
mean, logvar = vae.encode(test_data)

# Use the mean to decode the test data
decoded_data = vae.decode(mean)
