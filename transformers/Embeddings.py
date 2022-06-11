import math
from abc import ABC

from torch import nn
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.functional import F
import tensorflow as tf
import tensorflow.keras as kr
import copy

# 以下演示torch embedding.
embedding = nn.Embedding(10, 3)
long_tensor = torch.LongTensor([[1, 2, 4, 5],
                                [4, 3, 2, 0]])
print(embedding(long_tensor))

# 以下演示padding_idx
embedding = nn.Embedding(10, 3, 0)
print(embedding(long_tensor))


# 以下演示Embedding class的构建
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 500],
                               [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print('embr:', embr)
print(embr.shape)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(max_len) / d_model))
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(1)], requires_grad=False)
        return self.dropout(x)


d_model = 512
dropout = .1
max_len = 60
x = embr
pe = PositionalEmbedding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)

pe = PositionalEmbedding(20, 0)
y = pe(Variable(torch.zeros(1, 100, 20)))
plt.plot(y[0, :].data.numpy())
plt.legend([f'dim {p}' for p in range(20)])
plt.show()

dropout = nn.Dropout(.5)
long_tensor = torch.FloatTensor([1, 2, 3, 4, 5])
x = dropout(long_tensor)
print(x)

x = np.ones([20, 20], dtype=np.uint8)
x = 1 - np.triu(x, 1)
print(x)
plt.figure(figsize=(5, 5))
plt.imshow(x)
plt.show()


def tri_mask(size):
    mask = np.tri(size, dtype=np.uint8)
    return mask


x = tri_mask(20)
print(x)
plt.imshow(x)
plt.show()


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn @ value, p_attn


query = key = value = torch.randn(2, 4, 512)
mask = np.tri(4, dtype=np.uint8)
mask = torch.from_numpy(mask)
attn, p_attn = attention(query, key, value, mask=mask)
print(f'attn: {attn};\t{attn.shape}')
print(f'p_attn: {p_attn},\t{p_attn.shape}')


def attention_tensorflow(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = np.einsum('...ij,...kj->...ik', query, key) / np.sqrt(d_k)
    if mask is not None:
        scores = scores * mask
    p_attn = tf.nn.softmax(scores, -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn @ value, p_attn


query = key = value = np.random.uniform(-1, size=(2, 4, 512))
mask = np.tri(4, dtype=np.uint8)
attn, p_attn = attention_tensorflow(query, key, value, mask)
print('att:', attn, 'shape:', attn.shape)
print('p_attn:', p_attn, 'shape:', p_attn.shape)

x = np.ones([2, 3, 4, 5])
y = np.ones([3, 1, 5])
z = x + y
print(z, z.shape)


def clones(model, N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=.1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        self.liners = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.liners, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.liners[-1](x)


head = 8
embedding_dim = 512
dropout = .2
query = key = value = pe_result
mask = Variable(torch.zeros(8, 4, 4))
mha = MultiHeadAttention(8, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
print(mha_result)
print(mha_result.shape)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout=.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(embedding_dim, d_ff)
        self.w2 = nn.Linear(d_ff, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


embedding_dim = 512
d_ff = 64
dropout = .2
ff = PositionwiseFeedForward(embedding_dim, d_ff, dropout)
ff_result = ff(mha_result)
print(ff_result, ff_result.shape)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.std(x, -1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.eps


features = 512
ln = LayerNorm(features)
ln_result = ln(ff_result)
print(ln_result, ln_result.shape)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


size = 512
head = 8
dropout = .2
sc = SublayerConnection(size, dropout)
mask = Variable(torch.zeros(8, 4, 4))
self_attn = MultiHeadAttention(head, size, dropout)
sc_result = sc(pe_result, lambda x: self_attn(x, x, x, mask))
print(sc_result, sc_result.shape)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


el = EncoderLayer(size, self_attn, PositionwiseFeedForward(size, d_ff, dropout), dropout)
el_result = el(pe_result, mask)
print('encoder layer:', el_result, el_result.shape)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


el = EncoderLayer(size, MultiHeadAttention(head, size, dropout),
                  PositionwiseFeedForward(size, d_ff, dropout), dropout)
en = Encoder(el, 5)
en_result = en(pe_result, torch.zeros(4, 4))
print('encoder:', en_result, en_result.shape)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, sc_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.sc_attn = sc_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.sc_attn(x, m, m, source_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x


size = 512
head = 8
dropout = .2
d_ff = 64
x = pe_result
memory = en_result
source_mask = target_mask = torch.zeros(4, 4)
self_attn = sc_attn = MultiHeadAttention(head, size, dropout)
feed_forward = PositionwiseFeedForward(size, d_ff, dropout)
dl = DecoderLayer(size, self_attn, sc_attn, feed_forward, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
print('DecoderLayer:', dl_result, dl_result.shape)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


x = pe_result
memory = en_result
size = d_model = 512
d_ff = 64
head = 8
dropout = .2
source_mask = target_mask = torch.zeros(4, 4)
self_attn = sc_attn = MultiHeadAttention(head, size, dropout)
feed_forward = PositionwiseFeedForward(size, d_ff, dropout)
dl = DecoderLayer(size, self_attn, sc_attn, feed_forward, dropout)
de = Decoder(dl, 5)
de_result = de(x, memory, source_mask, target_mask)
print('Decoder:', de_result, de_result.shape)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


x = de_result
d_model = 512
vocab_size = 1000
gen = Generator(d_model, vocab_size)
gen_result = gen(x)
print('Generator:', gen_result, gen_result.shape)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encoder(self.src_embed(source), source_mask)
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab = 1000
size = 512
source_embed = nn.Embedding(vocab, size)
target_embed = nn.Embedding(vocab, size)
gen = Generator(size, vocab)
ed = EncoderDecoder(en, de, source_embed, target_embed, gen)
source = target = Variable(torch.LongTensor([[998, 123, 0, 789], [456, 1, 25, 97]]))
source_mask = target_mask = torch.zeros(4, 4)
ed_result = ed(source, target, source_mask, target_mask)
print('EncoderDecoder:', ed_result, ed_result.shape)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model, dropout)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(feed_forward), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(feed_forward), dropout), N),
                           nn.Sequential(Embeddings(d_model, source_vocab), PositionalEmbedding(d_model, dropout)),
                           nn.Sequential(Embeddings(d_model, target_vocab), PositionalEmbedding(d_model, dropout)),
                           Generator(d_model, target_vocab))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


source_vocab = target_vocab = 20
model = make_model(source_vocab, target_vocab)
print(model)

# class MultiHeadAttentionTF(kr.Model, ABC):
#     def __init__(self, head, embedding_dim, dropout=.1):
#         super(MultiHeadAttentionTF, self).__init__()
#         assert embedding_dim % head == 0
#         self.d_k = embedding_dim // head
#         self.head = head
#         self.embedding_dim = embedding_dim
#         self.dropout = kr.layers.Dropout(dropout)
#         self.attn = None
#         self.linears = [kr.layers.Dense(embedding_dim) for _ in range(4)]
#
#     def call(self, inputs, training=None, mask=None):
#         _, vocab, _ = inputs[0].shape
#         query, key, value = [tf.transpose(tf.reshape(linear(x), [-1, vocab, self.head, self.d_k]), [0, 2, 1, 3]) for
#                              linear, x in zip(self.linears, inputs)]
#         x, self.attn = attention_tensorflow(query, key, value, mask, self.dropout)
#         x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, vocab, self.head * self.d_k])
#         x = self.linears[-1](x)
#         return x
#
#
# head = 8
# embedding_dim = 512
# query = key = value = pe_result.data.numpy()
# mask = np.zeros([4, 4], dtype=np.uint8)
# mha = MultiHeadAttentionTF(head, embedding_dim)
# mha_result = mha([query, key, value], mask=mask)
# print(mha_result, mha_result.shape)
# print(mha.attn, mha.attn.shape)
