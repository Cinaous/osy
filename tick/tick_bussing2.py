import numpy as np
import pandas as pd
import tushare as ts

pro = ts.pro_api('52381b0ccf073115eb53141e54cac8d36d705b36d9b19bb9eb5fc80b')


def stock_data(ts_code, start_date='20180101'):
    data: pd.DataFrame = pro.daily(ts_code=ts_code, start_date=start_date)
    data = data[::-1]
    feature = data.iloc[:, 2:]
    return feature, data


def stock_train_data(data, idx, seq_num=12, batch=32, seq=5):
    batch_data, batch_label = [], []
    for start_idx in np.random.choice(idx, batch, False):
        end_idx = start_idx + seq_num
        one_data = data[start_idx:end_idx]
        one_label = data[end_idx:end_idx + seq, [3]]
        batch_data.append(one_data)
        batch_label.append(one_label)
    batch_data = np.array(batch_data)
    batch_data, (data_mean, data_std) = normalize(batch_data)
    return batch_data, (np.array(batch_label) - data_mean) / data_std


def normalize(data):
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    data = (data - data_mean) / data_std
    return data, (data_mean[..., [3]], data_std[..., [3]])


class Dataset:
    def __init__(self, ts_code, batch=32, seq_num=12, seq=5, percent=3, start_date='20180101'):
        data, _ = stock_data(ts_code, start_date=start_date)
        self.source_data = np.array(data)
        self.index = data.index
        self.batch = batch
        self.seq_num = seq_num
        self.seq = seq
        self.idx = np.arange(len(data) - seq_num - seq - 1)
        self.train_idx = self.idx[self.idx % percent != 0]
        self.test_idx = self.idx[self.idx % percent == 0]

    def __iter__(self):
        return self

    def __next__(self):
        return stock_train_data(self.source_data, self.train_idx, self.seq_num, self.batch, self.seq)

    def load_test_data(self, batch=1):
        return stock_train_data(self.source_data, self.test_idx, self.seq_num, batch, self.seq)

    def load_predict_data(self):
        data = np.array([self.source_data[-self.seq_num:]])
        return normalize(data)


if __name__ == '__main__':
    dataset = Dataset('000001.SZ')
    x_test, y_test = dataset.load_test_data()
    print(x_test.shape)
    print(y_test.shape)
    for x_train, y_train in dataset:
        print(x_train.shape, y_train.shape)
