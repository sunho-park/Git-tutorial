import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers
from keras.utils.np_utils import to_categorical
# %matplotlib inline

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train =x_train.reshape(x_train.shape[0], 784)[:6000]
x_test = x_test.reshape(x_test.shape[0], 784)[:1000]


y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]


model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=500, epochs=5, verbose=1, validation_data=(x_test, y_test))

plt.plot(history.history["acc"], label="acc", ls="-", market="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", market="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()


