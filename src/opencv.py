import sys,cv2

# 画像の読み込み
img = cv2.imread("image/Lenna.jpg")

# 画像の大きさを取得
height, width, channnels = img.shape[:3]
print("width: " + str(width))
print("height: " + str(height))