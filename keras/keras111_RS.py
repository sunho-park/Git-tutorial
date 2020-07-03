# 107번을 Activation 넣어서 완성하시오.

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, LSTM
from keras.layers import MaxPooling2D
import numpy as np
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

# 1. 데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape) # (60000, 784)
print(y_train.shape) # (60000, 10)

# 2. 모델
def build_model(drop, optimizer, lr, act):
    inputs = Input(shape=(784,), name='input')
    x = Dense(512, activation=act, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=act, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=act, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameters(): # epochs, node, activation 추가 가능

    batches = [30, 50]
    lr = [0.01, 0.1]
    # lr = np.linspace(0.1, 0.5, 4)
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    activation = ['tanh','relu','leaky','elu','selu']
  
    return {"batch_size" : batches, "lr": lr, "optimizer" : optimizers, "drop" : dropout, 'act': activation}
            
# wrapper
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

hyperparameters = create_hyperparameters()

# grid, random search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3, n_jobs=1)
search = RandomizedSearchCV(model, hyperparameters, cv=3, n_jobs=1, n_iter=10)

# fit
search.fit(x_train, y_train)
print(search.best_params_)

# acc
score = search.score(x_test, y_test)
print("score: ", score)
