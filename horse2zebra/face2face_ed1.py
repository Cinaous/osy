import os.path as path
import cv2
import numpy as np
import tensorflow.keras as kr
from face2face_datasets1 import Datasets
from models_model.model_encoder_decoder import Encoder, Decoder
import matplotlib.pyplot as plt

batch_size, epoch, first, losses, init_lr = 1, 0, True, [], 1e-3
dataset = Datasets(batch=batch_size)
encoder = Encoder()
decoder_h = Decoder()
decoder_z = Decoder()
ed_h = kr.Sequential([encoder, decoder_h])
ed_z = kr.Sequential([encoder, decoder_z])

dir_label = '43'
encoder_save_path = 'outputs/encoder%s.ckpt' % dir_label
decoder_h_save_path = 'outputs/decoder_h%s.ckpt' % dir_label
decoder_z_save_path = 'outputs/decoder_z%s.ckpt' % dir_label
epoch_save_path = 'outputs/images%s/epoch.npy' % dir_label
loss_save_path = 'outputs/images%s/loss.npy' % dir_label
if path.exists(epoch_save_path):
    epoch, init_lr = np.load(epoch_save_path).tolist()
    losses = np.load(loss_save_path)
    er = int(epoch // 100)
    ls = [np.mean(losses[i * er:(i + 1) * er], 0) for i in range(100)]
    ls = np.array(ls)
    plt.plot(ls[:, 0], label='g_loss')
    plt.plot(ls[:, 1], label='d_loss')
    losses = losses.tolist()
    plt.legend()
    plt.title('loss')
    plt.show()
if path.exists(encoder_save_path + '.index'):
    encoder.load_weights(encoder_save_path)
if path.exists(decoder_h_save_path + '.index'):
    decoder_h.load_weights(decoder_h_save_path)
if path.exists(decoder_z_save_path + '.index'):
    decoder_z.load_weights(decoder_z_save_path)
ed_h.compile(optimizer=kr.optimizers.Adam(init_lr), loss=kr.losses.mae)
ed_z.compile(optimizer=kr.optimizers.Adam(init_lr), loss=kr.losses.mae)
for horse, zebra in dataset:
    if epoch % 50 == 0 or first:
        test_horse, test_zebra = horse[-1:], zebra[-1:]
        encode_h = encoder(test_horse)
        h2z = decoder_z(encode_h)
        h2h = decoder_h(encode_h)
        img1 = np.concatenate((test_horse, h2z, h2h), axis=2)[0]
        encode_z = encoder(test_zebra)
        z2h = decoder_h(encode_z)
        z2z = decoder_z(encode_z)
        img2 = np.concatenate((test_zebra, z2h, z2z), axis=2)[0]
        img3 = np.vstack((img1, img2))
        img3 = Datasets.convert_data(img3)
        if first:
            first = False
            encoder.summary()
            decoder_z.summary()
            decoder_h.summary()
            cv2.imshow('trainA&trainB', img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        cv2.imwrite('outputs/images%s/trainA&z_%d.jpg' % (dir_label, epoch), img3)
    h_loss = ed_h.train_on_batch(horse, horse)
    z_loss = ed_z.train_on_batch(zebra, zebra)
    try:
        encoder.save_weights(encoder_save_path)
        decoder_h.save_weights(decoder_h_save_path)
        decoder_z.save_weights(decoder_z_save_path)
    except Exception:
        pass
    losses.append([h_loss, z_loss])
    epoch += 1
    np.save(loss_save_path, losses)
    np.save(epoch_save_path, [epoch, init_lr])
    print('current epoch is %d: \n\ttrainA is %.5f, trainB is %.5f, lr is %.8f' % (epoch, h_loss, z_loss, init_lr))
