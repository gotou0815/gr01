import numpy as np
from keras.models import Sequential
from keras.layers import Dense

input_dim = 1
output_dim = 1
X_train = np.array([[1],[2],[3],[4],[5]])
y_train = np.array([[2],[4],[6],[8],[10]])

# モデルの構築
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(input_dim,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))

# モデルのコンパイル
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# モデルの学習
model.fit(X_train, y_train, epochs=10, batch_size=32)

# モデルの保存
model.save('../model/test.h5')