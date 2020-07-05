# 100번을 카피해서 lr 을 넣고 튠하시오.
# LSTM - > Dense 로 바꿀것

'''keras100_hyper_lstm.py 의 주석 
# 97번을 RandomizedSearchCV 로 변경하시오
# score 넣어보기 
'''
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, LSTM
from keras.layers import MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

import numpy as np

# 1. 데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape) # (60000, 784)
print(y_train.shape) # (60000, 10)

# 2. 모델
# 2) 모델링 함수 바꾸기
# def build_model(drop=0.5, optimizer = Adam, lr=0.01):
def build_model(drop, optimizer, lr):
    inputs = Input(shape=(784,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs=outputs)

    # opt = optimizer(lr=lr)
    
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')

    return model

 # 하이퍼파라미터 함수 바꾸기
def create_hyperparameters():

    batches = [64,  126]
    # lr = [0.1, 0.01, 0.001, 0.005, 0.007]
    learning_rate = [0.05, 0.1, 0.5]
    # optimizers = ['rmsprop', 'adam', 'adadelta']
    optimizers = [Adam, RMSprop, SGD]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    
    return {"batch_size" : batches, "lr": learning_rate, "optimizer" : optimizers,
            "drop" : dropout}

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1, epochs = 1)

hyperparameters = create_hyperparameters()

# random, grid search 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3, n_jobs=1)
search = RandomizedSearchCV(model, hyperparameters, cv=3, n_jobs=1, n_iter=5)

search.fit(x_train, y_train)
print(search.best_params_)

#
score = search.score(x_test, y_test)
print("score: ", score)
