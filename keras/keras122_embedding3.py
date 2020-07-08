from keras.preprocessing.text import Tokenizer
import numpy as np


docs = ["너무 재밋어요", "참 최고에요", "참 잘 만든 영화에요","추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", 
        "글쎄요", "별로에요", "생각보다 지루해요", "연기가 어색해요", "재미없어요",
        "너무 재미없다", "참 재밋네요"]

# 긍정 1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화 
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre')             # ex) 0 이 앞에서 채워짐 0 0 0 3 7 
print(pad_x)        # (12, 5)

pad_y = pad_sequences(x, padding='post', value=1.0) # ex ) 0 이 뒤에서 채워짐 3 7 0 0 0 / value = 1  0대신 1이 채워짐
print(pad_y)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 : ", word_size)  # 25 전체 단어의 갯수

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

# 시계열
model = Sequential()
# model.add(Embedding(word_size, 10, input_length=5))       # 벡터형식, 와꾸 맞춰주는 부분 (25(임의의숫자넣어도 돌아감), output 다음층으로 전달되는?, input) 
# model.add(Embedding(25, 10, input_length=5))              # (None, 5, 10)
model.add(Embedding(25, 10))                                # (None, None, 10)
model.add(LSTM(3))      # 4*(10+3+1)*3
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) # label가 0이냐 1이냐 구분만해주면되서

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1] # [1] metrics값을 출력하겠다
print('acc : ', acc)






