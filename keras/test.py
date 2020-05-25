import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

a = np.array(range(1, 101))

size = 5

def split_x(seq, size):
    aaa=[]

    for i in range(len(seq)- size + 1): #96행
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])

    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[ :, :4]
y = dataset[ :, 4:5]
x = x.reshape(96, 4, 1)

from keras.models import load_model
model = load_model('./model/save_keras44.h5')
model.add(Dense(30, name='new1'))
model.add(Dense(30, name='new2'))
model.add(Dense(30, name='new3'))
model.add(Dense(30, name='new4'))
model.add(Dense(1, name='new5'))
model.summary()

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0 ,write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['adam'])

hist = model.fit(x, y, epochs=100, batch_size=1, shuffle=True, callbacks=[early_stopping])

import matplotlib as plt
plt.plot(hist.histogram['loss'])
plt.plot(hist.histogram['val_loss'])
plt.plot(hist.histogram['acc'])
plt.plot(hist.histogra['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()
