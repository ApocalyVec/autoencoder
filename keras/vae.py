from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# this is the size of the latent representation
encoding_dim = 32
input_len = 784
# load the dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_img = Input(shape=(input_len, ))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_len, activation='sigmoid')(encoded)

# the encoder-decoder model
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# create a layer of latent input for the decoder
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1] # for one layer autodecoders
decoder = Model(encoded_input, decoder_layer(encoded_input))

# configure the model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train the simple autoencoder
autoencoder.fit(x_train, x_train, epochs=500, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# test the encoder/decoder
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# display some of the results
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()