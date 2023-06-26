from keras.datasets import mnist
import numpy as np
import cv2

# mnistを読み込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 画像の表示
img = X_train[0].reshape(28, 28)
cv2.imwrite('test.png', img)

# モデル読み込んでpredictしたい