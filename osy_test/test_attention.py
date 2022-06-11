import tensorflow.keras as kr
import numpy as np

if __name__ == '__main__':
    x = np.random.uniform(size=[3, 5, 2])
    print(x)
    layer = kr.layers.MultiHeadAttention(7, 17, 19, attention_axes=(0, 1, 2))
    y, w = layer(x, x, return_attention_scores=True)
    print(y)
    print(w.shape)
