from keras.layers import LSTM, Dense, Input, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard

import keras.backend as K

import numpy as np

def diff(y_true, y_pred):
	if K.backend() == 'tensorflow':
		import tensorflow as tf
		return tf.abs(y_true - y_pred)

model = Sequential(name='Model')

model.add(LSTM(20, input_shape=[5, 5], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(30, return_sequences=False))
model.add(Dense(3, activation='softmax'))

X = np.random.random((1000, 5, 5))
y = np.zeros((1000, 3))

# model.summary()

model.compile(RMSprop(), loss='categorical_crossentropy', metrics=['acc'])

# history = model.fit(X, y, verbose=2, batch_size=5, epochs=10, callbacks=[EarlyStopping(monitor='loss', patience=2, verbose=1), TensorBoard()])

inp1 = Input((5, 5))
inp2 = Input((5, 5))

lstm1 = LSTM(20, dropout=0.1, return_sequences=False)
out1 = lstm1(inp1)

lstm2 = LSTM(30, dropout=0.1, return_sequences=False)
out2 = lstm2(inp2)

merged = Concatenate()([out1, out2])
dense = Dense(3, activation='softmax')
out = dense(merged)

model2 = Model(inputs=[inp1, inp2], outputs=[out])
model2.summary()
model2.compile(RMSprop(), loss=diff, metrics=['acc'])
history = model2.fit([X, X], y, verbose=2, batch_size=5, epochs=10, callbacks=[EarlyStopping(monitor='loss', patience=2, verbose=1), TensorBoard()])


