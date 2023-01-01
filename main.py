from imp import new_module
from operator import mod
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scale weights form 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

#input layer
model.add(tf.keras.layers.Flatten()) #flatten 28x28 dataset  

#hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # (128 neurons) (Rectified linear unit = filter)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # (softmax = probability distrubition)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )

#train
model.fit(x_train, y_train, epochs=3) # 3 iterations (1: 0.9218, 2: 0.9667, 3: 0.9762)

#val_loss, val_acc = model.evaluate(x_test, y_test)
#print (val_loss, val_acc) #loss: 0.0879, acc: 0.9721

model.save('demo.model')
new_model = tf.keras.models.load_model('demo.model')

predictions = new_model.predict([x_test])
print(np.argmax(predictions[0])) #output: 7 so it predicts first index in dataset is an image of the number 7


