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
import numpy as np

# 1. 데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255
#################################################################
# x_train = x_train.reshape(x_train.shape[0], 28*28, 1)/255
# x_test = x_test.reshape(x_test.shape[0], 28*28, 1)/255
#################################################################

x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape) # (60000, 784)
print(y_train.shape) # (60000, 10)

# 2. 모델
# 2) 모델링 함수 바꾸기
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
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

 # 하이퍼파라미터 함수 바꾸기
def create_hyperparameters():

    batches = [10, 30, 50]
    # lr = [0.1, 0.01, 0.001, 0.005, 0.007]
    lr = np.linspace(0.1, 0.5, 4)
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    
    return {"batch_size" : batches, "lr": lr, "optimizer" : optimizers,
            "drop" : dropout}

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3, n_jobs=1)
search = RandomizedSearchCV(model, hyperparameters, cv=3, n_jobs=1)

search.fit(x_train, y_train)

# score = search.score(x_test, y_test)

print(search.best_params_)

# print("score    : ", score)