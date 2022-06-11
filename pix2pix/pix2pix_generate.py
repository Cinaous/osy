import cv2


def video2images(vpath, spath, label):
    vc = cv2.VideoCapture(vpath)
    i = 0
    while True:
        ret, img = vc.read()
        if not ret:
            vc.release()
            return
        i += 1
        cv2.imwrite(f'{spath}/{label}/{i}.jpg', img)


if __name__ == '__main__':
    video2images('../horse2zebra/zebra_Trim.mp4', 'h2z', 'trainB')
