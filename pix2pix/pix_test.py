from pix_models import Pix2PixModel
from dataset_cv import load_data
import numpy as np
import matplotlib.pyplot as plt
from pix_scaler import StandardScaler

model = Pix2PixModel()
model_save_path = 'models/pix2pix.ckpt'
model.load_weights(model_save_path)

big_data, _ = load_data(False, None)
scaler = StandardScaler()
idx = np.random.choice(len(big_data), 3, replace=False)
x_test = big_data[idx]
y_test = model(scaler.fit_transform(x_test))
y_test = scaler.inverse_transform(y_test)
for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test[i])
    plt.subplot(2, 3, i + 4)
    plt.imshow(y_test[i])
plt.show()
