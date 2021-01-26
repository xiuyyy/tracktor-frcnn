import cv2
import os
from PIL import Image

def img2video():
    imgPath = "output/mysite/showvideo_result_img/"

    images = os.listdir(imgPath)
    img_ = Image.open(imgPath + images[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output/mysite/showvideo_result.mp4", fourcc, 24.0, img_.size)
    imgNum = len(os.listdir(imgPath))

    for i in range(len(images)):
        img = cv2.imread(imgPath + str(i+1).zfill(6) + '.jpg', 1)
        out.write(img)

    out.release()

# print("../output/tracktor/showvideo.mp4", 'Synthetic success!')
