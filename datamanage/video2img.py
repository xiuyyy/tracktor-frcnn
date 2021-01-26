import cv2
import os

def video2img():
    videoPath = 'data/mysite/showvideo.mp4'
    imgPath = 'data/mysite/showvideo_img/'
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)

    cap = cv2.VideoCapture(videoPath)

    numFrame = 1
    success, img = cap.read()
    while success:
        imgSavePath = os.path.join(imgPath, str(numFrame).zfill(6) + '.jpg')
        cv2.imencode('.jpg', img)[1].tofile(imgSavePath)

        numFrame += 1
        success, img = cap.read()

    print('完成！')