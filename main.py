import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', "Sneaker", 'Bag', 'Ankle boot'
]
trainX = trainX.reshape( (trainX.shape[0], 28,28,1) )
testX = testX.reshape( (testX.shape[0], 28,28,1) )

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)

from keras.utils.vis_utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

########################################################################################

#Input -> Flatten -> Dense -> Reshape
input1 = tf.keras.layers.Input(shape=[28, 28])
flatten1 = tf.keras.layers.Flatten()(input1)
dense1 = tf.keras.layers.Dense(28 * 28, activation='relu')(flatten1)
reshape1 = tf.keras.layers.Reshape((28, 28))(dense1)

concat1 = tf.keras.layers.Concatenate()([input1, reshape1]) #ㄹㅔ이어 합치기
flatten2 = tf.keras.layers.Flatten()(concat1)
output = tf.keras.layers.Dense(10, activation='softmax')(flatten2)

m1 = tf.keras.Model(input1, output)
m1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
plot_model(m1, to_file='model.png', show_shapes=True, show_layer_names=True)


