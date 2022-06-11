import random
import numpy as np


class Cache:
    def __init__(self, pool_size=50):
        self.items = []
        self.pool_size = pool_size
        self.batch_size = None

    def __call__(self, items=None):
        if items is None:
            items = []
        num = len(items) or 1
        for item in items:
            self.items.append(item)
        self.items = self.items[-self.pool_size:]
        self.batch_size = num
        return self.sample(num)

    def sample(self, num=None):
        num = num or self.batch_size
        outputs = random.sample(self.items, num)
        return np.array(outputs)


if __name__ == '__main__':
    cache = Cache(pool_size=5)
    for _ in range(10):
        _item = np.random.uniform(-1, 1, [3, 32, 32, 3])
        print(_item.shape)
        item_ = cache(_item)
        print(item_ == _item, item_.shape)
    print(cache())
