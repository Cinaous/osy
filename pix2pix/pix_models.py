import abc
import tensorflow.keras as kr
import tensorflow as tf
import numpy as np


class HWTransformer(kr.Model, abc.ABC):
    def __init__(self, num_heads=8, key_dim=32, dropout=.2):
        super(HWTransformer, self).__init__()
        self.h_attention = kr.layers.MultiHeadAttention(num_heads, key_dim, attention_axes=1, dropout=dropout)
        self.h_norm = kr.layers.LayerNormalization((1, 2))
        self.w_attention = kr.layers.MultiHeadAttention(num_heads, key_dim, attention_axes=2, dropout=dropout)
        self.w_norm = kr.layers.LayerNormalization((1, 2))

    def call(self, inputs, training=None, mask=None):
        rx = x = inputs
        x = self.h_norm(x)
        x = self.h_attention(x, x)
        rx = x + rx
        x = self.w_norm(rx)
        x = self.w_attention(x, x)
        x += rx
        return x


class SwinTransformer(kr.Model, abc.ABC):
    def __init__(self, deeps, units, kernel_size=3, activation=kr.activations.swish,
                 dropout=.2, last_layer=None, num_heads=8):
        super(SwinTransformer, self).__init__()
        self.conv = kr.layers.Conv2D(units, kernel_size, 2, 'same')
        self.norm = kr.layers.LayerNormalization((1, 2))
        self.activation = activation
        self.attention = [[HWTransformer(num_heads, units, dropout) for _ in range(deep)] for deep in deeps]
        self.merging = [PatchMerging(activation, dropout) for _ in range(1, len(deeps))]
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.activation(self.norm(x))
        for i, attention in enumerate(self.attention):
            x = attention[0](self.merging[i - 1](x) if i else x)
            for atten in attention[1:]:
                x = atten(x)
        return x if self.last_layer is None else self.last_layer(self.activation(x))


class ConvMerging(kr.layers.Layer):
    def __init__(self, kernel_size=3, activation=kr.activations.swish, dropout=.2):
        super(ConvMerging, self).__init__()
        self.conv = None
        self.kernel_size = kernel_size
        self.norm = kr.layers.LayerNormalization()
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)

    def build(self, input_shape):
        self.conv = kr.layers.Conv2D(2 * input_shape[-1], self.kernel_size, 2, 'same')

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.dropout(self.activation(self.norm(x)))
        return x


class PatchMerging(kr.layers.Layer):
    def __init__(self, activation=kr.activations.swish, dropout=.2):
        super(PatchMerging, self).__init__()
        self.liner = None
        self.norm = kr.layers.LayerNormalization((1, 2))
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)

    def build(self, input_shape):
        self.liner = kr.layers.Dense(2 * input_shape[-1])

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x1 = x[:, ::2, ::2]
        x2 = x[:, ::2, 1::2]
        x3 = x[:, 1::2, ::2]
        x4 = x[:, 1::2, 1::2]
        x = tf.concat([x1, x2, x3, x4], -1)
        x = self.liner(x)
        x = self.dropout(self.activation(self.norm(x)))
        return x


class HWAttention(kr.layers.Layer):
    def __init__(self, num_heads, key_dim, activation=kr.activations.swish, dropout=.2):
        super(HWAttention, self).__init__()
        self.norm = kr.layers.LayerNormalization()
        self.activation = activation
        self.h_attention = kr.layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout, attention_axes=1)
        self.w_attention = kr.layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout, attention_axes=2)

    def call(self, inputs, *args, **kwargs):
        rx = x = inputs
        x = self.h_attention(x, x)
        x = self.w_attention(x, x)
        x += rx
        x = self.activation(self.norm(x))
        return x


class CombinationNet(kr.Model, abc.ABC):
    def __init__(self, nets, output_dim, activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization(), dropout=.2):
        super(CombinationNet, self).__init__()
        self.nets = nets
        dropout = kr.layers.Dropout(dropout)
        self.dab = lambda x: dropout(activation(normalize(x)))
        self.liner = kr.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        x = [net(inputs) for net in self.nets]
        x = tf.concat(x, -1)
        return self.liner(self.dab(x))


class ComplexNet(kr.Model, abc.ABC):
    def __init__(self, rest, base_layer, normalize=kr.layers.LayerNormalization(),
                 activation=kr.activations.swish, dropout=.2, last_layer=None):
        super(ComplexNet, self).__init__()
        self.base_layers = [base_layer() for _ in range(rest[0])]
        self.rest = rest[1]
        dropout = kr.layers.Dropout(dropout)
        self.ab = lambda x: activation(normalize(x))
        self.dab = lambda x: dropout(self.ab(x))
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        cx = rx = x = inputs
        for i, base_layer in enumerate(self.base_layers):
            if i % 2 == 0 and not i:
                x += rx
                rx = cx
            cx = x = base_layer(self.dab(x) if i else x)
            rx = rx if i else x
        return x if self.last_layer is None else self.last_layer(self.ab(x))


class SingleCycleNet(kr.Model, abc.ABC):
    def __init__(self, x2y, y2x, dx, dy, lr=2e-4, lambdas=10):
        super(SingleCycleNet, self).__init__()
        self.x2y = x2y
        self.y2x = y2x
        self.dx = dx
        self.dy = dy
        self.x2y.trainable = True
        self.y2x.trainable = False
        self.dx.trainable = True
        self.dy.trainable = False
        gloss, dloss = kr.losses.MeanAbsoluteError(), kr.losses.MeanSquaredError()
        self.compile(optimizer=kr.optimizers.Adam(lr),
                     loss=[gloss, dloss, dloss, dloss],
                     loss_weights=[1, lambdas, 1, 1])

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        fy = self.x2y(x)
        return self.y2x(fy), self.dy(fy), self.dx(x), self.dx(self.y2x(y))


class CycleNet(kr.Model, abc.ABC):
    def __init__(self, a2b, b2a, da: kr.Model, db, lr=1e-3, lambdas=None, identifier=False):
        super(CycleNet, self).__init__()
        self.a2b = a2b
        self.b2a = b2a
        self.da = da
        self.db = db
        self.a2b.trainable = True
        self.b2a.trainable = True
        self.da.trainable = False
        self.db.trainable = False
        self.identifier = identifier
        mae, mse = kr.losses.MeanAbsoluteError(), kr.losses.MeanSquaredError()
        self.compile(optimizer=kr.optimizers.Adam(lr),
                     loss=[mae, mae, mse, mse] if not identifier else [mae, mae, mse, mse, mae, mae],
                     loss_weights=lambdas)

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        a2b, b2a = self.a2b(a), self.b2a(b)
        if self.identifier:
            return self.b2a(a2b), self.a2b(b2a), self.da(b2a), self.db(a2b), self.b2a(a), self.a2b(b)
        return self.b2a(a2b), self.a2b(b2a), self.da(b2a), self.db(a2b)


class DiscriminatorNet(kr.Model, abc.ABC):
    def __init__(self, a2b, b2a, da, db, lr=1e-3):
        super(DiscriminatorNet, self).__init__()
        self.a2b = a2b
        self.b2a = b2a
        self.da = da
        self.db = db
        self.a2b.trainable = False
        self.b2a.trainable = False
        self.da.trainable = True
        self.db.trainable = True
        self.compile(optimizer=kr.optimizers.Adam(lr),
                     loss=kr.losses.MeanSquaredError())

    def call(self, inputs, training=None, mask=None):
        a, b = inputs
        return self.da(a), self.db(b), self.da(self.b2a(b)), self.db(self.a2b(a))


class RestUCNet(kr.Model, abc.ABC):
    def __init__(self, rest, units, kernel_size, activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization, dropout=.2, last_layer=None):
        super(RestUCNet, self).__init__()
        self.rest = RestNetModel(rest, units, kernel_size, activation, normalize, dropout)
        self.conv = [kr.Sequential([
            kr.layers.Conv2DTranspose(units, kernel_size, strides, 'same'),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.Activation(activation),
            kr.layers.Conv2D(units // 2, 1),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.Activation(activation),
            kr.layers.Conv2D(units, kernel_size, padding='same')
        ]) for strides in self.rest.strides[::-1]]
        self.norm = [normalize(epsilon=1e-5) for _ in range(1, rest[0])]
        self.last_layer = last_layer if last_layer is None else kr.Sequential([
            normalize(epsilon=1e-5),
            kr.layers.Activation(activation),
            last_layer
        ])

    def call(self, inputs, training=None, mask=None):
        downs = self.rest.rest_call(inputs)
        downs = downs[::-1]
        x = downs[0]
        for i, conv in enumerate(self.conv):
            x = conv(x if i == 0 else self.rest.after_call(self.norm[i - 1](x + downs[i])))
        return x if self.last_layer is None else self.last_layer(x)


class EmbedNet(kr.Model, abc.ABC):
    def __init__(self, net, output_dim, input_dim=256):
        super(EmbedNet, self).__init__()
        self.embed = kr.layers.Embedding(input_dim, output_dim)
        self.net = net

    def call(self, inputs, training=None, mask=None):
        x = self.embed(inputs)
        x = tf.unstack(x, axis=-2)
        x = tf.concat(x, -1)
        x = self.net(x)
        return x


class AutoEncoderDecoder(kr.Model, abc.ABC):
    def __init__(self, deeps, units, kernel_size, activation=kr.activations.swish, dropout=.2):
        super(AutoEncoderDecoder, self).__init__()
        self.down = [(kr.layers.Conv2D(units + deep * units, kernel_size, 2, 'same'),
                      kr.layers.LayerNormalization()) for deep in range(deeps)]
        self.up = [(kr.layers.Conv2DTranspose((deeps - deep - 1) * units, kernel_size, 2, 'same'),
                    kr.layers.LayerNormalization()) for deep in range(deeps - 1)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.kernel_size = kernel_size
        self.last = None

    def build(self, input_shape):
        self.last = self.last or kr.layers.Conv2DTranspose(input_shape[-1], self.kernel_size, 2, 'same')
        super(AutoEncoderDecoder, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for down, norm in self.down:
            x = norm(down(x))
            x = self.dropout(self.activation(x))
        for up, norm in self.up:
            x = norm(up(x))
            x = self.dropout(self.activation(x))
        x = self.last(x)
        return x


class RestUNetModel(kr.Model, abc.ABC):
    def __init__(self, rest, units, kernel_size, activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization, dropout=.2, last_layer=None, axis=(1, 2)):
        super(RestUNetModel, self).__init__()
        self.rest = RestNetModel(rest, units, kernel_size, activation, normalize, dropout, axis=axis)
        self.conv = [kr.layers.Conv2DTranspose(
            units, kernel_size, strides, 'same') for strides in self.rest.strides[::-1]]
        self.norm = [normalize(epsilon=1e-5, axis=axis) for _ in range(1, rest[0])]
        self.last_layer = last_layer
        self.up = kr.layers.UpSampling2D()

    def call(self, inputs, training=None, mask=None):
        downs = self.rest.rest_call(inputs)
        downs = downs[::-1]
        rx = x = downs[0]
        for i, conv in enumerate(self.conv):
            if i % 2 == 0 and i != 0:
                cx = x
                x += rx if rx.shape[1] == x.shape[1] else self.up(rx)
                rx = cx
            x = conv(x if i == 0 else self.rest.after_call(self.norm[i - 1](x + downs[i])))
        return x if self.last_layer is None else self.last_layer(x)


class RestNetModel(kr.Model, abc.ABC):
    def __init__(self, rest, units, kernel_size, activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization, dropout=.2, last_layer=None, axis=(1, 2)):
        super(RestNetModel, self).__init__()
        assert len(rest) > 1
        self.strides = [2 if i % rest[1] == 0 else 1 for i in range(rest[0])]
        self.conv = [kr.layers.Conv2D(units, kernel_size, strides, 'same') for strides in self.strides]
        self.norm = [normalize(epsilon=1e-5, axis=axis) for _ in range(1, rest[0])]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.last_layer = last_layer
        self.pool = kr.layers.MaxPool2D()

    def after_call(self, x):
        return self.dropout(self.activation(x))

    def rest_call(self, inputs):
        cx = rx = x = inputs
        results = []
        for i, conv in enumerate(self.conv):
            if i % 2 == 0 and i != 0:
                x = tf.concat([x, self.pool(rx) if rx.shape[1] != x.shape[1] else rx], -1)
                rx = cx
            cx = x = conv(x if i == 0 else self.after_call(self.norm[i - 1](x)))
            results.append(x)
        return results

    def call(self, inputs, training=None, mask=None):
        x = self.rest_call(inputs)[-1]
        return x if self.last_layer is None else self.last_layer(x)


class SRestUNetModel(kr.Model, abc.ABC):
    def __init__(self, layers, units, kernel_size,
                 activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization,
                 dropout=.2, last_layer=None):
        super(SRestUNetModel, self).__init__()
        self.down = [SelfRestNet(n, units, kernel_size, activation, normalize, dropout) for n in layers]
        self.up = [SelfRestNet(n, units, kernel_size, activation, normalize, dropout) for n in layers[::-1]]
        self.pool = kr.layers.MaxPool2D()
        self.sampling = kr.layers.UpSampling2D()
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        x = inputs
        downs = []
        for i, down in enumerate(self.down):
            x = down(x if i == 0 else self.pool(x))
            downs.append(x)
        downs = downs[::-1]
        for i, up in enumerate(self.up):
            x = up(x if i == 0 else self.sampling(x + downs[i - 1]))
        return x if self.last_layer is None else self.last_layer(x)


class SRestNetModel(kr.Model, abc.ABC):
    def __init__(self, layers, units, kernel_size,
                 activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization,
                 dropout=.2, last_layer=None):
        super(SRestNetModel, self).__init__()
        self.rest = [SelfRestNet(n, units, kernel_size, activation, normalize, dropout) for n in layers]
        self.down = kr.layers.MaxPool2D()
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i, rest in enumerate(self.rest):
            x = rest(x if i == 0 else self.down(x))
        return x if self.last_layer is None else self.last_layer(x)


class RNNConvModel(kr.Model, abc.ABC):
    def __init__(self, seq, units, kernel_size, activation=kr.activations.swish, dropout=.2, keep_dims=True,
                 last_layer=None):
        super(RNNConvModel, self).__init__()
        self.conv = [kr.layers.Conv2D(units, kernel_size, padding='same') for _ in range(seq[0])]
        self.norm = [kr.layers.LayerNormalization() for _ in range(seq[0] - 1)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.keep_dims = keep_dims
        self.units = units
        self.kernel_size = kernel_size
        self.rnn = []
        self.rnn_seq = seq[1]
        self.last_layer = last_layer

    def build(self, input_shape):
        for i in range(self.rnn_seq):
            islast = i == self.rnn_seq - 1
            activation = None if islast and self.last_layer is None else 'tanh'
            if self.keep_dims:
                self.rnn.append(
                    kr.layers.ConvLSTM2D(input_shape[-1] if islast else self.units, self.kernel_size, padding='same',
                                         activation=activation, return_sequences=not islast))
            else:
                self.rnn.append(kr.layers.ConvLSTM2D(self.units, self.kernel_size, 2, 'same',
                                                     activation=activation, return_sequences=not islast))
        super(RNNConvModel, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        xs = []
        for conv, norm in zip(self.conv, self.norm):
            x = conv(x)
            xs.append(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        xs.append(self.conv[-1](x))
        x = tf.stack(xs, 1)
        for rnn in self.rnn:
            x = rnn(x)
        return x if self.last_layer is None else self.last_layer(x)


class RestUModel(kr.Model, abc.ABC):
    def __init__(self, layers, units, kernel_size, activation=kr.activations.swish, dropout=.2,
                 normalize=kr.layers.LayerNormalization, last_layer=None):
        super(RestUModel, self).__init__()
        self.down = [RestUnit(n, units, kernel_size, activation, dropout, normalize) for n in layers]
        self.up = [RestUnit(n, units, kernel_size, activation, dropout, normalize, kr.layers.Conv2DTranspose) for n in
                   layers]
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        x = inputs
        downs = []
        for down in self.down:
            x = down(x)
            downs.append(x)
        downs = downs[:-1][::-1]
        for up, down in zip(self.up, downs):
            x = up(x)
            x = tf.concat([x, down], -1)
        x = self.up[-1](x)
        return x if self.last_layer is None else self.last_layer(x)


class RestModel(kr.Model, abc.ABC):
    def __init__(self, layers, units, kernel_size, activation=kr.activations.swish, dropout=.2,
                 normalize=kr.layers.LayerNormalization, last_layer=None):
        super(RestModel, self).__init__()
        self.units = [RestUnit(n, units, kernel_size, activation, dropout, normalize) for n in layers]
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for unit in self.units:
            x = unit(x)
        return x if self.last_layer is None else self.last_layer(x)


class RestUnit(kr.Model, abc.ABC):
    def __init__(self, n, units, kernel_size, activation=kr.activations.swish, drop=.2,
                 normalize=kr.layers.LayerNormalization, conv=kr.layers.Conv2D):
        super(RestUnit, self).__init__()
        self.conv = [(conv(units, kernel_size, 2 if i == 0 else 1, padding='same'),
                      normalize(epsilon=1e-5)) for i in range(n)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(drop)

    def call(self, inputs, training=None, mask=None):
        rx = x = inputs
        for i, (conv, norm) in enumerate(self.conv):
            x = conv(x)
            if i % 2 == 0:
                rx = x
            else:
                x = tf.concat([x, rx], -1)
            x = self.activation(norm(x))
            x = self.dropout(x)
        return x


class SelfRestNet(kr.Model, abc.ABC):
    def __init__(self, n, units, kernel_size,
                 activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization,
                 dropout=.2, last_layer=None):
        super(SelfRestNet, self).__init__()
        self.conv = [(kr.layers.Conv2D(units, kernel_size, padding='same'),
                      normalize()) for _ in range(n)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        rx = x = inputs
        for i, (conv, norm) in enumerate(self.conv):
            cx = x = conv(x if i == 0 else self.dropout(x))
            if i % 2 == 1:
                x = tf.concat([x, rx], -1)
                rx = cx
            x = self.activation(norm(x))
        return x if self.last_layer is None else self.last_layer(x)


class RestUNetPLayer(kr.Model, abc.ABC):
    def __init__(self, rest_n, units, kernel_size, activation=kr.activations.swish, dropout=.2, last_layer=None):
        super(RestUNetPLayer, self).__init__()
        strides = [1 if i % rest_n[1] != 0 else 2 for i in range(rest_n[0])]
        self.downs = [(
            kr.layers.Conv2D(units, kernel_size, strides[i], padding='same'),
            kr.layers.LayerNormalization(epsilon=1e-5)
        ) for i in range(rest_n[0])]
        strides = strides[::-1]
        self.ups = [(
            kr.layers.Conv2DTranspose(units, kernel_size, strides[i], padding='same'),
            kr.layers.LayerNormalization(epsilon=1e-5)
        ) for i in range(rest_n[0])]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.down = kr.layers.MaxPool2D()
        self.up = kr.layers.UpSampling2D()
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        rx = x = inputs
        downs = []
        for i, (conv, norm) in enumerate(self.downs):
            x = conv(x)
            if i == 0:
                rx = x
            if i % 2 == 0 and i != 0:
                if rx.shape[1] != x.shape[1]:
                    rx = self.down(rx)
                rx = x = rx + x
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            downs.append(x)
        downs = downs[:-1][::-1]
        for i, (conv, norm) in enumerate(self.ups):
            x = conv(x)
            if i == 0:
                rx = x
            if i % 2 == 0 and i != 0:
                if rx.shape[1] != x.shape[1]:
                    rx = self.up(rx)
                rx = x = rx + x
            if i < len(downs):
                x += downs[i]
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = x if self.last_layer is None else self.last_layer(x)
        return x


class RestNetPLayer(kr.Model, abc.ABC):
    def __init__(self, rest_n, units, kernel_size, activation=kr.activations.swish, dropout=.2, last_layer=None):
        super(RestNetPLayer, self).__init__()
        self.conv = [(
            kr.layers.Conv2D(units, kernel_size, 1 if i % rest_n[1] != 0 else 2, padding='same'),
            kr.layers.LayerNormalization(epsilon=1e-5)
        ) for i in range(rest_n[0])]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.down = kr.layers.MaxPool2D()
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        rx = x = inputs
        for i, (conv, norm) in enumerate(self.conv):
            x = conv(x)
            if i == 0:
                rx = x
            if i % 2 == 0 and i != 0:
                if rx.shape[1] != x.shape[1]:
                    rx = self.down(rx)
                rx = x = rx + x
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = x if self.last_layer is None else self.last_layer(x)
        return x


class RestNetLayer(kr.layers.Layer):
    def __init__(self, rest_n, units, kernel_size, activation=kr.activations.swish, dropout=.2):
        super(RestNetLayer, self).__init__()
        self.conv = [(
            kr.layers.Conv2D(units, kernel_size, 1 if i % rest_n[1] != 0 else 2, padding='same'),
            kr.layers.LayerNormalization(epsilon=1e-5)
        ) for i in range(rest_n[0])]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.down = [kr.layers.Conv2D(units, kernel_size, 2, 'same') for _ in range(rest_n[0] // rest_n[1] + 1)]

    def call(self, inputs, *args, **kwargs):
        rx = x = inputs
        j = 0
        for i, (conv, norm) in enumerate(self.conv):
            x = conv(x)
            if i % 2 == 1:
                if rx.shape[1] != x.shape[1]:
                    rx = self.down[j](rx)
                    j += 1
                rx = x = rx + x
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class WRNModel(kr.Model, abc.ABC):
    def __init__(self, units, depths, activation=kr.activations.swish, dropout=.2):
        super(WRNModel, self).__init__()
        self.first = kr.Sequential([
            kr.layers.Conv2D(units, 7, 2, 'same'),
            kr.layers.LayerNormalization()
        ])
        self.block = [(WRNLayer(depth, activation, dropout),
                       kr.layers.Conv2D(units * 2 ** (i + 1), 5, 2, 'same')) for
                      i, depth in enumerate(depths)]

    def call(self, inputs, training=None, mask=None):
        x = self.first(inputs)
        for block, conv in self.block:
            x = block(x)
            x = conv(x)
        return x


class WRNLayer(kr.layers.Layer):
    def __init__(self, depth, activation=kr.activations.swish, dropout=.2):
        super(WRNLayer, self).__init__()
        self.block = [WRNBlock(activation, dropout) for _ in range(depth)]

    def call(self, inputs, *args, **kwargs):
        x = [block(inputs) for block in self.block]
        x = tf.add_n(x)
        return x + inputs


class WRNBlock(kr.layers.Layer):
    def __init__(self, activation=kr.activations.swish, dropout=.2):
        super(WRNBlock, self).__init__()
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.conv = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        hidden_dim = input_dim // 2
        self.conv = [kr.Sequential([
            kr.layers.Conv2D(hidden_dim if i != 2 else input_dim, 3, 1, 'same'),
            kr.layers.LayerNormalization()
        ]) for i in range(3)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        i, m = 0, len(self.conv) // 2
        for conv in self.conv:
            x = conv(x)
            x = self.activation(x)
            if i == m:
                x = self.dropout(x)
            i += 1
        return x


class ConvModel(kr.Model, abc.ABC):
    def __init__(self, units, kernel_size, n, conv=kr.layers.Conv2D, strides=2, padding='same'):
        super(ConvModel, self).__init__()
        self.conv = ConvLayer(units, kernel_size, n, conv, strides, padding)
        self.last = kr.layers.Conv2D(1, kernel_size, 1, 'same', activation=kr.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        return self.last(x)


class ConvUNet(kr.Model, abc.ABC):
    def __init__(self, units, kernel_size, n, keep_dims=True, strides=2, padding='same'):
        super(ConvUNet, self).__init__()
        self.down = ConvLayer(units, kernel_size, n, strides=strides, padding=padding)
        self.up = ConvLayer(units, kernel_size, n - 1, kr.layers.Conv2DTranspose, strides, padding)
        self.last = None if keep_dims else kr.layers.Conv2DTranspose(units, kernel_size, strides, padding)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs, training=None, mask=None):
        self.last = self.last or kr.layers.Conv2DTranspose(inputs.shape[-1],
                                                           self.kernel_size, self.strides, self.padding)
        x = inputs
        downs = []
        for down in self.down.conv:
            x = down(x)
            downs.append(x)
        downs = downs[:-1][::-1]
        for up, down in zip(self.up.conv, downs):
            x = up(x)
            x += down
        x = self.last(x)
        return tf.nn.tanh(x)


class ConvLayer(kr.layers.Layer):
    def __init__(self, units, kernel_size, n, conv=kr.layers.Conv2D, strides=2, padding='same'):
        super(ConvLayer, self).__init__()
        self.conv = [kr.Sequential([
            conv(units, kernel_size, strides, padding),
            kr.layers.LayerNormalization(epsilon=1e-5),
            kr.layers.Activation(kr.activations.swish),
            kr.layers.Dropout(.2)
        ]) for _ in range(n)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for conv in self.conv:
            x = conv(x)
        return x


class NCycleLayer(kr.layers.Layer):
    def __init__(self, units, kernel_size, n=3, activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization):
        super(NCycleLayer, self).__init__()
        self.down = [kr.Sequential([
            kr.layers.Conv2D(units * (i + 1), kernel_size + 2 * i, 2, padding='same'),
            normalize(),
            kr.layers.Activation(activation)
        ]) for i in range(n)]
        self.up = [kr.Sequential([
            kr.layers.Conv2DTranspose(units * ((n - i - 1) if i != n - 1 else 1), kernel_size + 2 * i, 2,
                                      padding='same'),
            normalize(),
            kr.layers.Activation(activation)
        ]) for i in range(n)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        downs = []
        for down in self.down:
            x = down(x)
            downs.append(x)
        downs = downs[:-1][::-1]
        for up, down in zip(self.up, downs):
            x = up(x)
            x += down
        return self.up[-1](x)


class NScaleLayer(kr.layers.Layer):
    def __init__(self, units, kernel_size, n=3,
                 activation=kr.activations.swish,
                 normalize=kr.layers.LayerNormalization):
        super(NScaleLayer, self).__init__()
        self.down = [kr.Sequential([
            kr.layers.Conv2D(units * (i + 1), kernel_size + 2 * i, 2, padding='same'),
            normalize(),
            kr.layers.Activation(activation)
        ]) for i in range(n)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for down in self.down:
            x = down(x)
        return x


class SplitDimsLayer(kr.layers.Layer):
    def __init__(self, n, units, kernel_size, strides=1, padding='valid', dropout=.2,
                 normalize=kr.layers.LayerNormalization,
                 activation=tf.nn.swish,
                 conv=kr.layers.Conv2DTranspose):
        super(SplitDimsLayer, self).__init__()
        self.conv = [conv(units, kernel_size, strides, padding) for _ in range(n)]
        self.norm = None if normalize is None else normalize()
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout) if dropout is not None else None
        self.n, self.nums = n, None

    def split_dims(self, dims):
        q, w = dims // self.n, dims % self.n
        outputs = []
        for i in range(self.n):
            if i < w:
                outputs.append(q + 1)
            else:
                outputs.append(q)
        return outputs

    def call(self, inputs, *args, **kwargs):
        self.nums = self.nums or self.split_dims(inputs.shape[-1])
        x = tf.split(inputs, self.nums, axis=-1)
        x = [conv(x) for conv, x in zip(self.conv, x)]
        x = tf.concat(x, -1)
        if self.activation is None:
            return x
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x)
        x = self.dropout(x) if self.dropout is not None else x
        return x


class Pix2PixModel(kr.Model, abc.ABC):
    def __init__(self, strides=1, padding='valid'):
        super(Pix2PixModel, self).__init__()
        self.down = [SplitDimsLayer(1 + i, 48, (9 - i * 2, 11 - i * 2), strides, padding, conv=kr.layers.Conv2D) for i
                     in range(3)]
        self.up = [SplitDimsLayer(4 - i, 48, (5 + i * 2, 7 + i * 2), strides, padding) for i in range(3)]
        self.last = kr.layers.Conv2DTranspose(3, (11, 13), 2, 'same')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        downs = []
        for down in self.down:
            x = down(x)
            downs.append(x)
        downs = downs[:-1][::-1]
        downs.append(inputs)
        for up, down in zip(self.up, downs):
            x = up(x)
            x = tf.concat([x, down], -1)
        x = self.last(x)
        return x


class Pix2PixUNet(kr.Model, abc.ABC):
    def __init__(self):
        super(Pix2PixUNet, self).__init__()
        self.down = [SplitDimsLayer(1 + i, 48, 9 - i * 2, 2, 'same', conv=kr.layers.Conv2D) for i
                     in range(3)]
        self.dl = kr.layers.Conv2D(24 * 3, 5, padding='same')
        self.up = [SplitDimsLayer(4 - i, 48, 5 + i * 2, 2, 'same') for i in range(3)]
        self.last = kr.layers.Conv2DTranspose(3, 11, padding='same')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        downs = []
        for down in self.down:
            x = down(x)
            downs.append(x)
        x = self.dl(x)
        downs = downs[:-1][::-1]
        downs.append(inputs)
        for up, down in zip(self.up, downs):
            x = up(x)
            x = tf.concat([x, down], -1)
        x = self.last(x)
        return x


class Pix2PixEncoder(kr.Model, abc.ABC):
    def __init__(self):
        super(Pix2PixEncoder, self).__init__()
        self.down = [SplitDimsLayer(1 + i, 48, 9 - i * 2, 2, 'same', conv=kr.layers.Conv2D) for i
                     in range(3)]
        self.dl = kr.layers.Conv2D(24 * 3, 5, padding='same')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for down in self.down:
            x = down(x)
        x = self.dl(x)
        return x


class Pix2PixDecoder(kr.Model, abc.ABC):
    def __init__(self):
        super(Pix2PixDecoder, self).__init__()
        self.up = [SplitDimsLayer(4 - i, 48, 5 + i * 2, 2, 'same') for i in range(3)]
        self.last = kr.layers.Conv2DTranspose(3, 11, padding='same')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for up in self.up:
            x = up(x)
        x = self.last(x)
        return x


class RestNet131(kr.Model, abc.ABC):
    def __init__(self, rest, units, kernel_size=3, activation=kr.activations.relu, dropout=.2, last_layer=None):
        super(RestNet131, self).__init__()
        self.head = kr.layers.Conv2D(4 * units, kernel_size * 2 + 1, 2, 'same')
        self.conv = [(kr.layers.Conv2D(units, 1),
                      kr.layers.Conv2D(units, kernel_size, padding='same'),
                      kr.layers.Conv2D(4 * units, 1)) for _ in range(rest)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.norm = [kr.layers.LayerNormalization() for _ in range(1, 3 * rest)]
        self.down = kr.layers.MaxPool2D()
        self.last_layer = last_layer

    def call(self, inputs, training=None, mask=None):
        rx = x = self.head(inputs)
        x = self.activation(x)
        for i, convs in enumerate(self.conv):
            if i:
                rx = x = self.down(x + rx)
            for j, conv in enumerate(convs):
                x = conv(x if not i + j else self.dropout(self.activation(self.norm[i * 3 + j - 1](x))))
        return x if self.last_layer is None else self.last_layer(self.activation(x))


class RestUNet131(kr.Model, abc.ABC):
    def __init__(self, rest, units, kernel_size=3, activation=kr.activations.swish, dropout=.2, last_activation=None):
        super(RestUNet131, self).__init__()
        self.first_kernel = kernel_size * 2 + 1
        self.head = kr.layers.Conv2D(4 * units, self.first_kernel, 2, 'same')
        self.conv = [(kr.layers.Conv2D(units, 1),
                      kr.layers.Conv2D(units, kernel_size, padding='same'),
                      kr.layers.Conv2D(4 * units, 1)) for _ in range(rest)]
        self.activation = activation
        self.dropout = kr.layers.Dropout(dropout)
        self.norm = [kr.layers.LayerNormalization() for _ in range(1, 3 * rest)]
        self.down = kr.layers.MaxPool2D()
        self.up = [(kr.layers.Conv2DTranspose(units, 1),
                    kr.layers.Conv2DTranspose(units, kernel_size, padding='same'),
                    kr.layers.Conv2DTranspose(4 * units, 1)) for _ in range(rest)]
        self.unorm = [kr.layers.LayerNormalization() for _ in range(1, 3 * rest)]
        self.sampling = kr.layers.UpSampling2D()
        self.last_layer = None
        self.last_activation = last_activation

    def build(self, input_shape):
        self.last_layer = kr.layers.Conv2DTranspose(input_shape[-1],
                                                    self.first_kernel, 2, 'same', activation=self.last_activation)
        super(RestUNet131, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        rx = x = self.head(inputs)
        x = self.activation(x)
        downs = []
        for i, convs in enumerate(self.conv):
            if i:
                rx = x = self.down(x + rx)
            for j, conv in enumerate(convs):
                x = conv(x if not i + j else self.dropout(self.activation(self.norm[i * 3 + j - 1](x))))
            downs.append(x)
        downs = downs[::-1]
        rx = x
        for i, ups in enumerate(self.up):
            if i:
                rx = x = self.sampling(x + rx + downs[i - 1]) + downs[i]
            for j, up in enumerate(ups):
                x = up(x if not i + j else self.dropout(self.activation(self.unorm[i * 3 + j - 1](x))))
        return self.last_layer(x)


if __name__ == '__main__':
    x_train = np.random.uniform(size=(17, 64, 64, 3))
    model = ComplexNet((18, 3), HWTransformer, last_layer=kr.layers.Dense(3))
    y_predict = model(x_train)
    model.summary()
    print(y_predict.shape)
