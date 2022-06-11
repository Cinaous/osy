import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
import copy
import time
import os.path as path
import cv2
import glob


def load_data(data_path, label, size):
    label_path = f'{data_path}/{label}-{size[0]}x{size[1]}.npy'
    if path.exists(label_path):
        return np.load(label_path, allow_pickle=True)
    data = [cv2.resize(cv2.imread(filename), size) for filename in glob.glob(f'{data_path}/{label}/*')]
    data = np.array(data)
    np.save(label_path, data)
    return data


class CycleGAN:
    def __init__(self, label, dataset, gen_model, dis_model, lr=2e-4, m=25, load=True):
        self.dataset = dataset
        self.models = [gen_model, copy.copy(gen_model), dis_model, copy.deepcopy(dis_model)]
        self.model_save_path = [f'./models/{label}.{i + 1}' for i in range(len(self.models))]
        self.loss_fn = [kr.losses.MeanAbsoluteError(), kr.losses.MeanSquaredError()]
        self.real_label = np.ones(dis_model.compute_output_shape(dataset.output_shape))
        self.fake_label = np.zeros_like(self.real_label)
        self.optimizer = kr.optimizers.Adamax(lr)
        self.epoch = 0
        self.m = m
        self.file_save_path = f'./images/{label}'
        self.load = load

    def step(self, trainA, trainB):
        A2B, B2A, DA, DB = self.models
        mae, mse = self.loss_fn
        real_label, fake_label = self.real_label, self.fake_label
        start = time.time()
        with tf.GradientTape(True) as tape:
            a2b, b2a = A2B(trainA), B2A(trainB)
            d_losses = [mse(real_label, DA(trainA)), mse(real_label, DA(trainB)),
                        mse(fake_label, DB(a2b)), mse(fake_label, DA(b2a))]
            g_losses = [10. * mae(trainA, B2A(a2b)), 10. * mae(trainB, A2B(b2a)),
                        mse(real_label, DB(a2b)), mse(real_label, DA(b2a))]
            model_loss = (tf.add_n(d_losses), tf.add_n(g_losses))
        vs = (DA.trainable_variables + DB.trainable_variables, A2B.trainable_variables + B2A.trainable_variables)
        grads = [tape.gradient(ml, v) for ml, v in zip(model_loss, vs)]
        [self.optimizer.apply_gradients(zip(grad, v)) for grad, v in zip(grads, vs)]
        print(f'current epoch is {self.epoch}: <%.3fs>' % (time.time() - start),
              'Disc loss is %.5f, Cycle loss is %.5f\n\tDis ->' % model_loss,
              *[f'output-{i} is %.5f' % loss for i, loss in enumerate(d_losses)], "\n\tCycle ->",
              *[f'output-{i} is %.5f' % loss for i, loss in enumerate(g_losses)])

    def save_model(self):
        try:
            [model.save_weights(save_path) for model, save_path in zip(self.models, self.model_save_path)]
        except Exception as ex:
            print(ex)

    def load_model(self):
        for model, save_path in zip(self.models, self.model_save_path):
            if path.exists(save_path + '.index'):
                model.load_weights(save_path)

    def test_model(self, testA, testB):
        A2B, B2A, DA, DB = self.models
        a2b = A2B(testA)
        b2a = B2A(a2b)
        img1 = np.hstack([testA[0], a2b[0], b2a[0]])
        a2b = B2A(testB)
        b2a = A2B(a2b)
        img2 = np.hstack([testB[0], a2b[0], b2a[0]])
        img = self.dataset.ms.inverse_transform(np.vstack([img1, img2]))
        cv2.imwrite(f'{self.file_save_path}/gan-{self.epoch}.jpg', img)

    def __call__(self, *args, **kwargs):
        if self.load:
            self.load_model()
        for trainA, trainB in self.dataset:
            if self.epoch % self.m == 0:
                self.test_model(*self.dataset.load_test_data())
            self.step(trainA, trainB)
            self.save_model()
            self.epoch += 1
