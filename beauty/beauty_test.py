import glob
from beauty.beauty_model4 import model, model_save_path
from beauty.preprocessing_dataset2 import scaler
import numpy as np
import cv2
import shutil

img_dir = r'D:\workspace\oyhcrawler\91Crawler\images'
model.load_weights(model_save_path)
addresses = glob.glob(f'{img_dir}/*/*.jpg')[500:]
for address in addresses:
    img = cv2.imdecode(np.fromfile(address, dtype=np.uint8), cv2.IMREAD_COLOR)
    x = cv2.resize(img, (128, 128))
    xp = model.predict(np.expand_dims(scaler.fit_transform(x), 0))
    label = np.argmax(xp)
    cv2.putText(img, f'label is {label}', (64, 64), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2)
    if label == 0:
        cv2.imshow('beauty', img)
    else:
        cv2.imshow('normal', img)
    shutil.copy(address, f'label_{label}')
    cv2.waitKey(60)

