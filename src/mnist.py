from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

#mnistを読み込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#データの成形。kerasは0.0~1.0までの、float型の配列じゃないと扱うことが出来ない
X_train = X_train.reshape(60000, 784)/255
X_test = X_test.reshape(10000, 784)/255

#kerasのラベルデータはバイナリ型じゃないといけない(めんどくさっ)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#入力層は28x28の784個、中間層は活性化関数reluの512個、出力層は活性化関数softmaxの10個(0-9)
model = Sequential()
model.add(Dense(512, input_dim=(784)))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

#モデルのコンパイル。損失関数は交差エントロピー、最適化関数は確率的勾配降下法
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

#学習開始
hist = model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=1)
print("正解率(acc):", score[1])

model.save('model/MNIST.h5')