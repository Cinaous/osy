from pix_models import Pix2PixModel
from dataset_cv import Dataset
import tensorflow.keras as kr
import os.path as path
import numpy as np
from pix_scaler import StandardScaler
import cv2

model = Pix2PixModel(2, 'same')
model_save_path = 'models/pix2pix_fixed.ckpt'
if path.exists(model_save_path + '.index'):
    model.load_weights(model_save_path)
model.compile(optimizer=kr.optimizers.Adam(), loss=kr.losses.mae)

scaler = StandardScaler()
dataset = Dataset(12, image_size=(64, 64))
epoch, image_save_path = 0, 'images/glz'
for x_train, y_train in dataset:
    epoch += 1
    print('current epoch is %d' % epoch)
    model.fit(scaler.fit_transform(x_train), scaler.transform(y_train), workers=12, use_multiprocessing=True)
    try:
        model.save_weights(model_save_path)
    except Exception as ex:
        print(ex)
    if epoch % 25 == 0:
        x_test = x_train[:1]
        cv2.imwrite(f'{image_save_path}/{epoch}-frs.jpg', x_test[0])
        y_test = model(scaler.fit_transform(x_test))
        image = scaler.inverse_transform(y_test)
        image = np.hstack([image[0], y_train[0]])
        cv2.imwrite(f'{image_save_path}/{epoch}-fts.jpg', image)
