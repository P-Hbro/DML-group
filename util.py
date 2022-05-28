import numpy as np


def save_log(log_path, loss_list, acc, time):
    data_dict = {
        'loss': loss_list,
        'acc': acc,
        'time': time
    }
    np.save(log_path, data_dict)


def load_log(log_path):
    data = np.load(log_path, allow_pickle=True)
    data_dict = data.item()
    return data_dict['loss'], data_dict['acc'], data_dict['time']
