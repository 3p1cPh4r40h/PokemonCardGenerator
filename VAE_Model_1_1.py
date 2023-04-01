import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

# Read label encoded dataframe from pickle file
label_encoded_df = pd.read_pickle('data\label_encoded_df.pkl')
print(label_encoded_df.head(4))
# Normalize data between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(label_encoded_df)
scaled_label_encoded_df = pd.DataFrame(scaled_data, columns = label_encoded_df.columns)

# Split data into train and test sets
train_df, test_df = train_test_split(scaled_label_encoded_df, test_size=0.2, random_state=42)

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
            layers.LSTM(16, input_shape=(5,6), return_sequences=False),
            layers.Dense(latent_dim + latent_dim),
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(5*6, activation='relu', input_shape=(latent_dim,)),
            layers.Reshape((5, 6)),
            layers.Conv1DTranspose(16, 3, activation='relu', padding='same'),
            layers.Conv1DTranspose(6, 3, activation='sigmoid', padding='same')
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
test_data = reshaped_test_data

# Define the loss function
def vae_loss(x, x_recon, mean, logvar):
    # Use mean squared error as the loss function with KL divergence
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_recon))
    kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_divergence


# Create an instance of the VAE model
latent_dim = 6
vae = VAE(latent_dim)

# Define the optimizer and the batch size
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
batch_size = 32

# Create empty lists to store loss and accuracy
losses = []
accuracies = []

# Train the model
epochs = 30
for epoch in range(epochs):
    print('Epoch:', epoch+1)

    seed = epoch

    for step in range(train_data.shape[0] // batch_size):
        x = train_data[step*batch_size : (step+1)*batch_size]
        with tf.GradientTape() as tape:
            x_recon, mean, logvar = vae(x)
            loss = vae_loss(x, x_recon, mean, logvar)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        
        # Append the loss value to the list
        losses.append(loss.numpy())
        
        if step % 100 == 0:
            print('Step:', step, 'Loss:', loss.numpy())
        
    # Evaluate the model on the train set after each epoch
    x_train = train_data
    x_train_recon, _, _ = vae(x_train)
    train_loss = vae_loss(x_train, x_train_recon, *vae.encode(x_train))
    losses.append(train_loss.numpy())

    # Calculate the accuracy of the generated samples
    rng = np.random.RandomState(seed)
    generated_data = vae.decode(rng.normal(0, 1, size=(train_data.shape[0], latent_dim)))
    generated_data = np.where(generated_data > 0.5, 1, 0)
    accuracy = np.mean(np.all(generated_data == train_data, axis=(1, 2)))
    accuracies.append(accuracy)

    print('Train Loss:', train_loss.numpy())
    print('Accuracy:', accuracy)

# Plot the loss values over time
import matplotlib.pyplot as plt

step_to_epoch_ratio = int(len(losses) / len(accuracies))
fig, ax = plt.subplots()
ax.plot(losses[::step_to_epoch_ratio], color='b')
#add x-axis label
ax.set_xlabel('Epoch', fontsize=14)
# Set x-axis to use integers
plt.xticks(range(1,epochs+1), rotation=45)
#add y-axis label
ax.set_ylabel('Loss', color='b', fontsize=16)
#define second y-axis that shares x-axis with current plot
ax2 = ax.twinx()
#add second line to plot
ax2.plot(accuracies, color='r')
#add second y-axis label
ax2.set_ylabel('Accuracy', color='r', fontsize=16)
fig.suptitle('Training Loss vs Accuracy')
plt.show()

x_test = test_data
x_test_recon, _, _ = vae(x_test)
test_loss = vae_loss(x_test, x_test_recon, *vae.encode(x_test))
# Calculate the accuracy of the generated samples
generated_data = vae.decode(np.random.normal(0, 1, size=(test_data.shape[0], latent_dim)))
generated_data = np.where(generated_data > 0.5, 1, 0)
print(generated_data.shape)
print(test_array.shape)
val_accuracy = np.mean(np.all(generated_data == test_data, axis=(1, 2)))

print("Validation Loss: " + str(test_loss))
print("Validation accuracy: " + str(val_accuracy))

# Load the test data
test_data = reshaped_test_data
n = 1  # number of samples
mu, sigma = 0, 1  # mean and standard deviation of the noise
shape = (n, 5, 6)  # shape of the noise array

noise = np.random.normal(mu, sigma, shape)
# Encode the test data and get the mean and logvar
mean, logvar = vae.encode(noise)

# Use the mean to decode the test data
decoded_data = vae.decode(mean)

print(decoded_data)
scaled_output_array = np.zeros((test_array.shape[0], 10))
attack_average_var = 0

for i in range(decoded_data.shape[0]):
    for j in range(decoded_data.shape[1]):    
        for k in range(decoded_data.shape[2]):
            if j == 0:
                scaled_output_array[i, k] = decoded_data[i,j,k]
            else:
                attack_average_var = attack_average_var + decoded_data[i,j,k]
                if k == 5:
                    attack_average_var = attack_average_var/6
                    scaled_output_array[i, k+j] = attack_average_var
                    attack_average_var = 0

def decode_df(decoding, encoded):
    for column, label_encoder in decoding.items():
        if column != 'name':
            print(column)
            encoded[column] = encoded[column].map(dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)))
    return encoded

# Import decoding dictionary
decoding_dict = pd.read_pickle('data\label_encoder_objects.pkl')

decoded_df = pd.DataFrame(scaled_output_array, columns=label_encoded_df.columns[1:])
print(decoded_df.head(10))
# Add a nameless column for inverse transformation
decoded_df.insert(0, '', 0)
unscaled_output_array = scaler.inverse_transform(decoded_df)
print(unscaled_output_array)
unscaled_df = pd.DataFrame(unscaled_output_array, columns=label_encoded_df.columns)
unscaled_df = unscaled_df.apply(lambda x: round(x)).astype(int)
decoded_df = decode_df(decoding_dict, unscaled_df)
print(decoded_df.head(1))
# Save the decoded predictions
decoded_df.to_csv('data.csv')