import numpy as np


class FsScaler:
    def __init__(self):
        self.dtype = None

    def fit_transform(self, data):
        data = data if isinstance(data, np.ndarray) else np.array(data)
        self.dtype = data.dtype
        return self.transform(data)

    def transform(self, data):
        return data / 127.5 - 1

    def inverse_transform(self, data):
        assert self.dtype is not None
        data = data.numpy() if hasattr(data, 'numpy') else data
        data = 127.5 * data + 127.5
        return data.astype(self.dtype)


class MsScaler:
    """
    第一个维度为batch, 最后一个维度为通道数.
    """

    def __init__(self):
        self.dtype = None
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        data = np.array(data)
        self.dtype = data.dtype
        dims = data.ndim
        axis = [i for i in range(dims - 1)]
        axis = tuple(axis)
        self.mean = np.mean(data, axis)
        self.std = np.std(data, axis)
        return self.transform(data)

    def transform(self, data):
        assert self.mean is not None
        assert self.std is not None
        data = (data - self.mean) / self.std
        return np.nan_to_num(data)

    def inverse_transform(self, data):
        data: np.ndarray = data.numpy() if hasattr(data, 'numpy') else data
        assert self.mean is not None
        assert self.std is not None
        assert self.dtype is not None
        data = data * self.std + self.mean
        return data.astype(self.dtype)


class MrScaler:
    def __init__(self):
        self.dtype = None
        self.range = None
        self.median = None

    def fit_transform(self, data):
        data = data if isinstance(data, np.ndarray) else np.array(data)
        dims = data.ndim
        axis = tuple([i for i in range(1, dims - 1)])
        max, min, self.median = [f(data, axis=axis, keepdims=True) for f in (np.max, np.min, np.median)]
        self.range = max - min
        self.dtype = data.dtype
        return self.transform(data)

    def transform(self, data):
        for attr in (self.range, self.median):
            assert attr is not None
        data = (data - self.median) / self.range
        return np.nan_to_num(data)

    def inverse_transform(self, data):
        data = data.numpy() if hasattr(data, 'numpy') else data
        for attr in (self.range, self.median, self.dtype):
            assert attr is not None
        data = data * self.range + self.median
        return data.astype(self.dtype)


class StandardScaler:
    """
    第一个维度为batch, 最后一个维度为通道数.
    """

    def __init__(self):
        self.dtype = None
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        data = np.array(data)
        self.dtype = data.dtype
        dims = data.ndim
        axis = [i for i in range(1, dims - 1)]
        axis = tuple(axis)
        self.mean = np.mean(data, axis, keepdims=True)
        self.std = np.std(data, axis, keepdims=True)
        return self.transform(data)

    def transform(self, data):
        assert self.mean is not None
        assert self.std is not None
        data = (data - self.mean) / self.std
        return np.nan_to_num(data)

    def inverse_transform(self, data):
        data: np.ndarray = data.numpy() if hasattr(data, 'numpy') else data
        assert self.mean is not None
        assert self.std is not None
        assert self.dtype is not None
        assert data.ndim == self.mean.ndim
        data = data * self.std + self.mean
        return data.astype(self.dtype)


class FixedScaler:
    def __init__(self, fixed=255.):
        self.fixed = fixed
        self.dtype = None

    def fit_transform(self, data):
        self.dtype = data.dtype
        return self.transform(data)

    def transform(self, data):
        return data / self.fixed

    def inverse_transform(self, data):
        assert self.dtype is not None
        data = data.numpy() if hasattr(data, 'numpy') else data
        data *= self.fixed
        return data.astype(self.dtype)


class M3Scaler:
    def __init__(self):
        self.dtype = None
        self.max = None
        self.min = None
        self.mean = None

    def fit_transform(self, data):
        data = data if isinstance(data, np.ndarray) else np.array(data)
        dims = data.ndim
        axis = tuple([i for i in range(1, dims - 1)])
        self.max, self.min, self.mean = [f(data, axis=axis, keepdims=True) for f in (np.max, np.min, np.mean)]
        self.dtype = data.dtype
        return self.transform(data)

    def transform(self, data):
        for attr in (self.max, self.min, self.mean):
            assert attr is not None
        data = (data - self.mean) / (self.max - self.min)
        return np.nan_to_num(data)

    def inverse_transform(self, data):
        data = data.numpy() if hasattr(data, 'numpy') else data
        for attr in (self.max, self.min, self.mean, self.dtype):
            assert attr is not None
        data = data * (self.max - self.min) + self.mean
        return data.astype(self.dtype)


class MixtureScaler:
    def __init__(self, last=MrScaler, first=StandardScaler):
        self.last = last()
        self.first = first()

    def fit_transform(self, data):
        data = self.first.fit_transform(data)
        return self.last.fit_transform(data)

    def transform(self, data):
        data = self.first.transform(data)
        return self.last.transform(data)

    def inverse_transform(self, data):
        data = self.last.inverse_transform(data)
        return self.first.inverse_transform(data)


if __name__ == '__main__':
    # x = np.full([17, 64, 64, 3], 255)
    x = np.random.randint(256, size=[17, 64, 64, 3])
    print(x)
    scaler = FsScaler()
    scaled_x = scaler.fit_transform(x)
    print(scaled_x)
    inverse_x = scaler.inverse_transform(scaled_x)
    print(inverse_x)
