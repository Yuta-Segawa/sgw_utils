import numpy as np
from keras.utils.np_utils import to_categorical

def convert_to_onehot(labels):
    nb_classes = labels.max()+1
    return to_categorical(labels, nb_classes)

def split_into_batches(data, batch_size=100):
    if len(data) <= batch_size:
        return data
    else:
        return np.array_split(data, range(batch_size, len(data), batch_size))

def make_set_xy(loaded_npy, label_type='sklearn'):

    labels = np.int32([ np.ones(len(class_f)) * idx for idx, class_f in enumerate(loaded_npy) ])

    x = np.vstack(loaded_npy)
    y = np.hstack(labels)

    if label_type == 'keras':
        y = convert_to_onehot(y)

    return x, y