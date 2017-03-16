import numpy as np
from keras.utils.np_utils import to_categorical

def convert_to_onehot(labels):
    """Convert label array to one-hot vector by Keras function 'to_categorical'. 

    :param labels: Numpy integer array representing label indices. 
    :return: One-hot labels. 
    """
    nb_classes = labels.max()+1
    return to_categorical(labels, nb_classes)

def split_into_batches(data, batch_size=100):
    """Split input data into some batches. 

    :param data: Target data in numpy array. 
    :param batch_size: Size of batches. This works based on numpy function 'array_split'. 
    :return: List of Batches. 
    """
    if len(data) <= batch_size:
        return data
    else:
        return np.array_split(data, range(batch_size, len(data), batch_size))

def make_set_xy(loaded_npy, label_type='sklearn'):
    """Arange numpy features to the format used for inputting to classifiers. 

    :param loaded_npy: Numpy features in shape of (classes_num, samples_num, dimensionality). 
    :param label_type: Label type of platform where you will use returned set: 

        - sklearn: For scikit-learn platform like [0, 0, ..., 1, 1]. 
        - keras: For Keras platform (one-hot representation) like [[1,0], [1,0], ..., [0,1], [0,1]]. 

    :return: Tuple of (<vertically stacked features>, <corresponding labels>). 
    """
    labels = np.int32([ np.ones(len(class_f)) * idx for idx, class_f in enumerate(loaded_npy) ])

    x = np.vstack(loaded_npy)
    y = np.hstack(labels)

    if label_type == 'keras':
        y = convert_to_onehot(y)

    return x, y