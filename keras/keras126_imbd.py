from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000) # , test_split=0.2)
'''
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=None,skip_top=0, max_len=None,
                                                      seed=113, start_char=1, oov_char=2, index_from=3)

 - 위와같이 데이터셋을 불러들일 수 있다.
 - 위 명령어들에 대한 설명
 - 그중 알고있어야 할 명령어로는
 - num_words : 고려해야할 가장 빈도수가 높은 단어. 
 - 이 인자는 매우 애매한 단어 (ex. Ultracrepidarian)를 거르고 싶을 때 유용하다.
 - skip_top : 무시할 단어 중 가장 상위 단어(인덱스상). 
 - 중복되는 단어 또는 일반적인 단어(most common word)를 거르고 싶을 때 유용하다.
 - 예를들면 "the"라는 단어는 네트워크가 설계될 때 도움을 주는 단어가 아니기 때문에
 - skip_top의 값을 2 이상으로 설정하여 건너 뛸 수 있다.
 '''

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

#list 이므로 shape 안됨
print('len(x_train[0] :', len(x_train[0]))

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리 : ", category)  # 카테고리 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()

print(bbb)
print(bbb.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=111, padding='pre')
x_test = pad_sequences(x_test, maxlen=111, padding='pre')

# print(len(x_train[0]))
# print(len(x_train[-1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Dropout, Activation, MaxPool1D

model = Sequential()
model.add(Embedding(2000, 128, input_length=111))
# model.add(Embedding(2000, 128))

model.add(Conv1D(64, 5, padding='same', activation='relu', strides=1))
model.add(MaxPool1D(pool_size=4))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Dropout(0.3))
model.add(Dense(2, activation='sigmoid'))

# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=15, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)


# 그림
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# 1. imdb검색해서 데이터 내용 확인.
# word_size 전체 데이터 부분 변경해서 최상값 확인.
# groupby
# 원핫인코딩하기에 너무 큰 데이터를 Embedding으로 

# 인덱스를 단어로 바꿔주는 함수
word_index = imdb.get_word_index()

word_index = {k : (v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

'''
· The dataset

 - 250000개의 IMDB 데이터셋을 이용
 - Movie Data의 각 review에는 label이 붙어있음
 - Negative : 0 / Positive : 1
 - review의 단어를 기반으로 review의 sentiment를 예측하는 모델을 만드는 프로젝트를 진행해보자.
 - 이미 사전에 처리된 input data를 기반으로 네트워크 설계.
 - 각 review의 단어에 해당하는 인덱스로 접근이 가능하다.
 - 단어는 빈도수에 비례하게 정렬되므로, 예를들어 가장 빈도수가 높은 "the"에 인덱스 1이 대응됨
 - 인덱스 0은 알 수 없는 단어에 해당
 - sentence는 이 인덱스와 연결하여 벡터로 바뀜
 - 예를들어 "To be or not to be" 라는 문장은
 - 위와같이 대응됨
 - 따라서 위의 문장은 [5, 8, 21, 3, 5, 8] 이라는 벡터로 인코딩 된다.
 '''