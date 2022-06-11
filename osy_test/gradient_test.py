import tensorflow as tf
import tensorflow.keras as kr
import numpy as np

if __name__ == '__main__':
    optimizer = kr.optimizers.SGD(1.)
    x = tf.Variable(3.)
    y = tf.Variable(5.)
    tape = tf.GradientTape(True)
    with tape:
        for _ in range(3):
            l1 = x * x + y * x
            l2 = y * y + x * y
            lg1 = tape.gradient(l1, x)
            lg2 = tape.gradient(l2, y)
            optimizer.apply_gradients([(lg1, x)])
            optimizer.apply_gradients([(lg2, y)])
            print(lg1, lg2, x, y)
