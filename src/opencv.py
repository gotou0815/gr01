# これは.image/raw/*.pngをすべて160:90に変換し.image/pro/に保存するコード
import cv2
import pathlib

image_list = list(pathlib.Path("./image/raw").glob("**/*.png"))

for i in range(len(image_list)):
    # 画像の読み込み
    img = cv2.imread(str(image_list[i]))

    # 画像を1980:980から1600:900にトリミング
    # img[top : bottom, left : right]
    img_trim = img[90 : 990, 160 : 1760]

    print(image_list[i].name)

    # 画像を160:90に縮小
    img_shrink = cv2.resize(img_trim, (160, 90))

    # 画像を保存
    output_path = "./image/pro/" + image_list[i].name
    cv2.imwrite(output_path, img_shrink)