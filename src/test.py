import numpy as np
import tensorflow as tf

# テストモデルの読み込み
model_path = "../model/test.h5"
model = tf.keras.models.load_model(model_path)

answer = model.predict(np.array([7]))

print(answer)
print(answer.shape)
print(answer[0].argmax())