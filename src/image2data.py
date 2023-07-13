# これは./image/pro/内の.pngファイルを一次元化＆正規化されたnumpy配列に変換し、
# .npyファイル形式で保存するコード
import cv2
import numpy as np
import pathlib

data = []

# ./image/pro/内の.pngをリストに格納していく
image_list = list(pathlib.Path("./image/pro").glob("**/*.png"))

for i in range(len(image_list)):
    # 画像読み込み
    img = cv2.imread(str(image_list[i]))

    # numpy配列化
    img = np.asarray(img)

    # 一次元化
    img = np.ravel(img)

    # 0~1で正規化
    img = img.reshape(43200,)/255

    # 配列に追加
    data.append(img)

# .npyファイル形式で保存
np.save("./npy/test.npy", data)
