import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def decode_df(decoding, encoded):
    def lookup_encoding(col_name, value):
        encoding = decoding.loc[(decoding['column_name'] == col_name) & (decoding['decoding'] == value.iloc[0]), 'encoding'].values
        return encoding[0] if encoding.size > 0 else value
    
    return encoded.apply(lambda x: lookup_encoding(x.name, x) if x.name in decoding['column_name'].unique() else x)

# Import label encoded data and decoding dictionary from label encoding step
label_encoded_df = pd.read_pickle('data\label_encoded_df.pkl')
decoding_dict = pd.read_pickle('data\label_decoding_dict.pkl')

# Normalize data between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(label_encoded_df)
label_encoded_df = pd.DataFrame(scaled_data, columns=label_encoded_df.columns)

# Split data into train and test sets
train_df, test_df = train_test_split(label_encoded_df, test_size=0.2, random_state=42)

# Drop column names from train and test dataframes
train_array = train_df.values
test_array = test_df.values

# Define the model
latent_dim = 2
input_shape = (train_df.shape[1],)
hidden_size = 64
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim),
                                              mean=0., stddev=epsilon_std)
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

# Encoder network
inputs = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Dense(hidden_size, activation='relu')(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(hidden_size, activation='relu')(x)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

# Sample from the latent distribution
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decoder network
latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(hidden_size, activation='relu')(latent_inputs)
x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
outputs = tf.keras.layers.Reshape(input_shape)(x)

# Define the VAE as a Model
encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.models.Model(inputs, outputs, name='vae')

# Define the loss function
reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, outputs)
kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

# Compile the model
vae.add_loss(vae_loss)
# Define the optimizer with desired learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
vae.compile(optimizer=optimizer)

# Train the model
vae.fit(train_df, epochs=15, batch_size=32)

# Generate new data by sampling from the latent space
n_samples = 100
z_sample = np.random.normal(size=(n_samples, latent_dim))
x_decoded = decoder.predict(z_sample)

# Convert decoded values back to original dataset
decoded_array_normalized = decoder.predict(z_sample)
decoded_array = scaler.inverse_transform(decoded_array_normalized)
decoded_df = pd.DataFrame(decoded_array, columns=label_encoded_df.columns)
decoded_df = decoded_df.apply(lambda x: round(x)).astype(int)
decoded_df = decode_df(decoding_dict, decoded_df)
print(decoded_df.head())
# Save the VAE model
vae.save('model_vae.tf', save_format='tf')

# Save the decoded predictions
pd.to_pickle(decoded_df, 'vae_results.pkl')
