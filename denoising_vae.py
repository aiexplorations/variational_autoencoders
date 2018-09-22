'''
Denoising Variational Autoencoder

0. Code is more or less an exact reproduction of the excellent tutorial at : https://blog.keras.io/building-autoencoders-in-keras.html 
1. Used for Digit Data Denoising
2. Run on Google's Colaboratory (free GPU power!)
3. Modified the output plots to include:
    a. Original digit
    b. Digit with noise
    c. Reconstructed digits with noise removed

'''


'''
Standard imports
'''

import numpy as np
import matplotlib.pyplot as plt


# Keras imports here include the Upsampling layer which is used for reconstruction of the image data

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.models import Model

'''
Loading the MNIST images into the variables and adding noise to them


1. Noise factor controls the amount of noise that will be added to the images
2. Train and test images are both modified with the noise
3. Noisy images are plotted for reference

'''

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

'''

In the below section, details of the network are defined:

1. Convolution and Pooling layers for the encoder network
2. Input shape is set to the size of the images, i.e., (28,28)
3. The encoding process converts the image to a (7,7,32) tensor

'''

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

'''
The network defined from this point onwards is a decoder network.
1. This has alternating convolution and upsampling steps
2. The final result of the decoder network is an image of (28,28) dimensions
'''


x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

'''
1. Note that the training step where we fit the model expects examples of non-noisy images as the ground truth. 
2. Depending on the quality of input data and targets, we can get different levels of performance from the denoising autoencoder

'''

autoencoder.fit(x_train_noisy, x_train, epochs=32, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test) )

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 6))
for i in range(1,n):
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display noisy version
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    # display reconstruction
    ax = plt.subplot(3, n, i + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()