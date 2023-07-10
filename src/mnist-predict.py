# これはMNISTのテストデータの１つ目をpredictして正しく認識できているか確かめるコード
# 参考元 https://note.nkmk.me/python-tensorflow-keras-basics/
from keras.datasets import mnist
import numpy as np
import cv2
import tensorflow as tf

# mnistを読み込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 画像の保存
img = X_test[0].reshape(28, 28)
cv2.imwrite("./image/MNIST_X_test[0].png", img)

#データの成形。kerasは0.0~1.0までの、float型の配列じゃないと扱うことが出来ない
X_test = X_test.reshape(10000, 784)/255

# モデルを読み込む
model_path = "model/MNIST.h5"
model = tf.keras.models.load_model(model_path)

# Kerasではバッチ処理を前提としているため、データの型を(1,784)とする必要がある
img_expand = X_test[0][np.newaxis, ...]
print(img_expand.shape)

# X_test[0]をpredict
predictions = model.predict(img_expand)
print(predictions[0].argmax())