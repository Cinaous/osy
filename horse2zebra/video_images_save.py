import os
import os.path as path
import cv2
import dlib

name = 'trainA'
save_dir = 'faces/' + name
if not path.exists(save_dir):
    os.mkdir(save_dir)
video_path = name + '.mp4'
vc = cv2.VideoCapture(video_path)
detector: dlib.fhog_object_detector = dlib.get_frontal_face_detector()
rate = ix = 0

while True:
    ret, img = vc.read()
    if not ret:
        break
    rate += 1
    if rate < 23:
        continue
    rects = detector.run(img)[0]
    if len(rects) == 0:
        continue
    print('image length is %d' % len(rects))
    imgs = [img[rect.top(): rect.bottom(), rect.left():rect.right()] for rect in rects]
    for img in imgs:
        ix += 1
        try:
            img = cv2.resize(img, (128, 128))
        except cv2.error:
            continue
        cv2.imwrite('faces/%s/%05d.jpg' % (name, ix), img)
vc.release()
