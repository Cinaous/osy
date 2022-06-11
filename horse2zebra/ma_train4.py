from horse2zebra_datasets3 import Datasets
import tensorflow.keras as kr
import tensorflow as tf
import cv2
import numpy as np
from GAN.bma.ma_model4 import edcoder, disc, dis
import os.path as path

dataset = Datasets()
bs = 1
train_db = dataset.train_db(bs)
# x_noise, x_real = dataset.load_test_dataset()
# x_a, x_b = x_noise, x_real
y_real, y_fake = np.ones([bs, 8, 8]), np.zeros([bs, 8, 8])
# w, trainA = x_real.shape[1:3]
# ns = int(np.log2(w))
ns = 7
g = edcoder(ns)
f = edcoder(ns)
dg = dis(ns)
df = dis(ns)

g_optimizer = kr.optimizers.Adam()
# g_optimizer = kr.optimizers.Adam()
# f_optimizer = kr.optimizers.Adam()
d_optimizer = kr.optimizers.Adam()
# dg_optimizer = kr.optimizers.Adam()
# df_optimizer = kr.optimizers.Adam()

g_save_path = 'g.ckpt'
f_save_path = 'f.ckpt'
dg_save_path = 'dg.ckpt'
df_save_path = 'df.ckpt'
# if path.exists(g_save_path + '.index'):
#     g.load_weights(g_save_path)
#     f.load_weights(f_save_path)
#     dg.load_weights(dg_save_path)
#     df.load_weights(df_save_path)

epoch = 0
while True:
    for x_a, x_b in train_db:
        with tf.GradientTape(persistent=True) as tape:
            # a -> b, b -> a
            g_b = g(x_a)
            fg_a = f(g_b)
            f_a = f(x_b)
            gf_b = g(f_a)

            # same a, b
            s_a = f(x_a)
            s_b = g(x_b)

            dg_fake = dg(g_b)
            dg_real = dg(x_a)

            df_fake = df(f_a)
            df_real = df(x_b)

            loss_cycle_gf = kr.losses.mean_squared_error(x_a, fg_a)
            loss_cycle_fg = kr.losses.mean_squared_error(x_b, gf_b)
            loss_cycle_total = loss_cycle_gf + loss_cycle_fg
            loss_cycle_total = tf.reduce_mean(loss_cycle_total)

            loss_same_f = kr.losses.mean_squared_error(x_a, s_a)
            loss_same_f = tf.reduce_mean(loss_same_f)
            loss_same_g = kr.losses.mean_squared_error(x_b, s_b)
            loss_same_g = tf.reduce_mean(loss_same_g)

            loss_g = kr.losses.sparse_categorical_crossentropy(y_real, dg_fake)
            loss_g = tf.reduce_mean(loss_g)
            loss_f = kr.losses.sparse_categorical_crossentropy(y_real, df_fake)
            loss_f = tf.reduce_mean(loss_f)

            loss_g_total = loss_g + 3 * loss_cycle_total + 1.5 * loss_same_g
            loss_f_total = loss_f + 3 * loss_cycle_total + 1.5 * loss_same_f
            loss_cycle_full = loss_f + loss_g + 10. * loss_cycle_total

            loss_dg_total = kr.losses.sparse_categorical_crossentropy(y_fake, dg_fake) + \
                            kr.losses.sparse_categorical_crossentropy(y_real, dg_real)
            loss_dg_total = tf.reduce_mean(loss_dg_total)
            loss_df_total = kr.losses.sparse_categorical_crossentropy(y_fake, df_fake) + \
                            kr.losses.sparse_categorical_crossentropy(y_real, df_real)
            loss_df_total = tf.reduce_mean(loss_df_total)
            loss_d_total = loss_dg_total + loss_df_total
        g_grads = tape.gradient(loss_cycle_full, g.trainable_variables + f.trainable_variables)
        d_grads = tape.gradient(loss_d_total, dg.trainable_variables + df.trainable_variables)
        # g_grads = tape.gradient(loss_cycle_full, g.trainable_variables)
        # f_grads = tape.gradient(loss_f_total, f.trainable_variables)
        # dg_grads = tape.gradient(loss_dg_total, dg.trainable_variables)
        # df_grads = tape.gradient(loss_df_total, df.trainable_variables)

        print('Current epoch is %d, g loss is %f, dg loss is %f, f loss is %f, df loss is %f, full loss is %f' % (
            epoch, loss_d_total, loss_dg_total, loss_f_total, loss_df_total, loss_cycle_full))
        g_optimizer.apply_gradients(zip(g_grads, g.trainable_variables + f.trainable_variables))
        d_optimizer.apply_gradients(zip(d_grads, dg.trainable_variables + df.trainable_variables))
        # g_optimizer.apply_gradients(zip(g_grads, g.trainable_variables))
        # f_optimizer.apply_gradients(zip(f_grads, f.trainable_variables))
        # dg_optimizer.apply_gradients(zip(dg_grads, dg.trainable_variables))
        # df_optimizer.apply_gradients(zip(df_grads, df.trainable_variables))

        if epoch % 100 == 0:
            g.save_weights(g_save_path)
            f.save_weights(f_save_path)
            dg.save_weights(dg_save_path)
            df.save_weights(df_save_path)
            x_fake = g(x_a)
            x_fake2 = f(x_fake)
            x_fake4 = f(x_b)
            x_fake3 = g(x_fake4)
            img1 = np.hstack((x_a[0], x_fake[0], x_fake2[0]))
            img1 = dataset.convert_data(img1)
            img2 = np.hstack((x_b[0], x_fake3[0], x_fake4[0]))
            img2 = dataset.convert_data(img2)
        cv2.imshow('fake', np.vstack((img1, img2)))
        cv2.waitKey(10)
        epoch += 1
