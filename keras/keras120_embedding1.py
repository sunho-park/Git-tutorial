from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다"

token = Tokenizer()
token.fit_on_texts([text]) # 띄어쓰기로 단어 구분
print(token.word_index)
# {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

x = token.texts_to_sequences([text])
print(x) # [[1, 2, 3, 4]]

from keras.utils import to_categorical

word_size = len(token.word_index) + 1
print(word_size)

x = to_categorical(x, num_classes=word_size)
# or x = to_categorical(x, 5) 
# or x = to_categorical(x) 
print(x)
