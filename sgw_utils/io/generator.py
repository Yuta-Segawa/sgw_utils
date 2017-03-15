import os, glob, cv2, random
import numpy as np
from keras.preprocessing import image
from multiprocessing import Process, Queue, Pool
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input


# this script includes 2 types of image preparation modules:
# 1. load all images at one time and feed images from RAM
# 2. load batch images from directory and feed the images

##############
### common ###
##############

def convert_to_onehot(labels):
    one_hot_vectors = np.float32([ np.zeros((len(class_label), len(labels))) for class_label in labels ])
    for class_label, v in zip(labels, one_hot_vectors):
        v[np.arange(len(v)), class_label.astype(int)] = 1
    return one_hot_vectors.astype(float)

####################################
### for local pattern prediction ###
####################################

def crop_imgs(img, patch_size=32, return_locs=False):
    import itertools as it
    tl_x_locs = np.arange(0, img.shape[1], patch_size)
    tl_y_locs = np.arange(0, img.shape[0], patch_size)
    tl_locs = np.array(list(it.product(tl_y_locs, tl_x_locs)))
    br_locs = tl_locs + np.array([patch_size, patch_size])
    mirrored_img = np.hstack([np.vstack([img, img[::-1]]), np.vstack([img[::, ::-1], img[::-1, ::-1]])])
    if return_locs:
        return [mirrored_img[tl[0]:br[0], tl[1]:br[1]] for tl, br in zip(tl_locs, br_locs)], zip(tl_locs, br_locs)
    else:
        return [mirrored_img[tl[0]:br[0], tl[1]:br[1]] for tl, br in zip(tl_locs, br_locs)]


##############
### type 1 ###
##############
def resize_and_transpose(img_fn, shape):
    return image.img_to_array(image.load_img(img_fn, target_size=shape)).astype(float)
def mp_func(args):
    return resize_and_transpose(args[0], args[1])

# for training adn evaluation
def load_images_from_dirs(basedir, classnames, 
    image_shape=(256, 256), extension="jpg", max_smps=-1, mp_nums=1):
    # only_labels: A flag to avoid to resizing images by reading only labels
    # max_smps: Number of images per a class which should be loaded. -1 means load maximum number as possible

    # get filenames from directories of the classes
    dirnames = [ os.path.join(basedir, cls) for cls in classnames 
                if os.path.exists(os.path.join(basedir, cls)) ]
    if not len(dirnames) == len(classnames):
        print("[I]Too few classes are found (expected %d classes, but %d). " % (len(classnames), len(dirnames)))
        return None
    img_filenames = [ sorted(glob.glob(os.path.join(dirname, "*.%s" % extension))) 
                    for dirname in dirnames ]
    # display info
    for idx, (found_dir, fns) in enumerate(zip(dirnames, img_filenames)):
        print("[I]Found %d images in %s. labeled them as %d. " % (len(fns), found_dir, idx))
        if max_smps != -1: 
            max_smps = min(len(fns), max_smps) 

    # load all images
    print("[I]Resize with shape %s" % str(image_shape))
    mp_status = 'enabled with %d threads' % mp_nums if not mp_nums == 1 else 'disable'
    print("[I]( Multi-Processing: %s)" % mp_status)

    loaded_imgs = []
    for i, cls_img_fns in enumerate(img_filenames):

        if max_smps != -1:
            random.shuffle(cls_img_fns)
            cls_img_fns = cls_img_fns[0:max_smps]
            
        print("[I]Resizing %d images belonging to the class %d. " % (len(cls_img_fns), i))

        if mp_nums > 1:
            p = Pool(mp_nums)
            loaded_imgs.append(np.array(p.map(mp_func, zip(cls_img_fns, [image_shape]*len(cls_img_fns))), float))
            p.close()
        else:
            loaded_imgs.append(np.array([mp_func(args) for args in zip(cls_img_fns, [image_shape]*len(cls_img_fns))], float))

    # make one-hot vectors as training labels
    labels = [ np.array([idx] * len(imgs)).astype(int) for idx, imgs in enumerate(img_filenames) ]
    if max_smps != -1:
        labels = [ l[0:max_smps] for l in labels ]
    one_hot_vectors = [  np.zeros((len(class_label), len(labels))) for class_label in labels ]
    for class_label, v in zip(labels, one_hot_vectors):
        v[np.arange(len(v)), class_label] = 1

    print("[I]Loaded all images and labels. ")

    return loaded_imgs, one_hot_vectors

# for prediction
def load_images_in_dir(dirname, 
    image_shape=(256, 256), extension="jpg", max_smps=-1, mp_nums=1):
    # only_labels: A flag to avoid to resizing images by reading only labels
    # max_smps: Number of images per a class which should be loaded. -1 means load maximum number as possible

    # get filenames from directories of the classes
    img_filenames = sorted(glob.glob(os.path.join(dirname, "*.%s" % extension)))

    # display info
    print("[I]Found %d images in %s. " % (len(img_filenames), dirname))
    if max_smps != -1: 
        max_smps = min(len(img_filenames), max_smps) 

    loaded_imgs = []
    if max_smps != -1:
        random.shuffle(img_filenames)
        img_filenames = img_filenames[0:max_smps]

    print("[I]Use %d images. " % len(img_filenames))

    # load all images
    if image_shape is None:
        print("[I]Not any resizing. ")
    else: 
        print("[I]Resizing with shape %s..." % str(image_shape))
    mp_status = 'enabled with %d threads' % mp_nums if not mp_nums == 1 else 'disable'
    print("[I]( Multi-Processing: %s)" % mp_status)


    if mp_nums > 1:
        p = Pool(mp_nums)
        loaded_imgs = np.array(p.map(mp_func, zip(img_filenames, [image_shape]*len(img_filenames)))).astype(float)
        p.close()
    else:
        loaded_imgs = np.array([mp_func(args) for args in zip(img_filenames, [image_shape]*len(img_filenames))]).astype(float)

    return loaded_imgs

def preprocess_on_images(images, type='inception'):

    # determinant preprocessing
    print("[I]Applying preprocessing. ")
    print("[I](Preprocessing type: %s)" % type)
    if type == 'inception':
        images /= 255.0
        images -= 0.5
        images *= 2.0
    elif type  == 'vgg':
        images = preprocess_input(images)
    elif type == 'disable':
        pass
    else: 
        print("[E]Invalid preprocessing type", type)
        return None
    return images

def generator_preparation(images, labels, 
    batch_size=10, type='inception', shuffle=False, 
    save_image_prefix=None):

    # transfromation in the preprocessing
    datagen = ImageDataGenerator()

    images = preprocess_on_images(images, type)

    # fit the images to generator in order to image preprocessing
    # datagen.fit(images)
    generator = datagen.flow(X=images, y=labels, 
        batch_size=batch_size, shuffle=shuffle, 
        save_to_dir=save_image_prefix)

    return generator

def train_val_gen_preparation(images, labels, 
    batch_size=10, split_rate=0.1, type='inception', shuffle=False, 
    save_image_prefix=None):
    
    # transfromation in the preprocessing
    datagen = ImageDataGenerator()

    images = preprocess_on_images(images, type)

    zipped = zip(images, labels)
    random.shuffle(zipped)
    X, y = list(zip(*zipped))

    nb_val_smp = int(float(len(images))*split_rate)
    train_X = np.array(X[:-nb_val_smp])
    val_X   = np.array(X[nb_val_smp:])
    train_y = np.array(y[:-nb_val_smp])
    val_y   = np.array(y[nb_val_smp:])

    train_gen = datagen.flow(X=train_X, y=train_y, 
        batch_size=batch_size, shuffle=shuffle, 
        save_to_dir=save_image_prefix)
    val_gen = datagen.flow(X=val_X, y=val_y, 
        batch_size=batch_size, shuffle=shuffle, 
        save_to_dir=save_image_prefix)

    return train_gen, val_gen


##############
### type 2 ###
##############
class ImageNetDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.rescale:

            if self.rescale == 'inception':
                x /= 255.0
                x -= 0.5
                x *= 2.0

            elif self.rescale == 'vgg':
                x = np.squeeze(preprocess_input(np.expand_dims(x, axis=0)))
            else:
                x *= self.rescale

        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

class DataIterator(ImageDataGenerator):

    def random_transform(self, x):
        # just returns the value without any transformations 
        return x

from keras.preprocessing.image import NumpyArrayIterator

class NumpyArrayIterator_LargeChannelAdapted(NumpyArrayIterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.x = np.asarray(x, dtype=K.floatx())
        # if self.x.ndim != 4:
        #     raise ValueError('Input data in `NumpyArrayIterator` '
        #                      'should have rank 4. You passed an array '
        #                      'with shape', self.x.shape)
        # channels_axis = 3 if dim_ordering == 'tf' else 1
        # if self.x.shape[channels_axis] not in {1, 3, 4}:
        #     raise ValueError('NumpyArrayIterator is set to use the '
        #                      'dimension ordering convention "' + dim_ordering + '" '
        #                      '(channels on axis ' + str(channels_axis) + '), i.e. expected '
        #                      'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
        #                      'However, it was passed an array with shape ' + str(self.x.shape) +
        #                      ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

def generator_preparation_from_dirs(dirpath, target_size, classes=None, 
    batch_size=10, shuffle=False, preprocessing_type='inception',
    save_image_prefix=None, class_separation=False, validation_include=True):
    
    # Note: class_separation should be True only when not using ground truth labels i.e. in prediction. 


    # what 'classes' is None is meaning a process on only one directory directly
    if classes is None:
        classes = dirpath.rstrip("/").split("/")[-1]
        dirpath = "/".join(dirpath.rstrip("/").split("/")[:-1])

    # modulate object type due to 'classes' input for flow_from_directory
    if not type(classes) == list:
        classes = [classes]

    # transfromation in the preprocessing
    datagen = None
    if preprocessing_type == 'inception':
        datagen = ImageNetDataGenerator(rescale='inception')
    elif preprocessing_type == 'vgg':
        datagen = ImageNetDataGenerator(rescale='vgg')
    elif preprocessing_type == 'disable':
        datagen = ImageDataGenerator()
    else: 
        print("[E]Invalid preprocessing type", type)
        quit()

    print "[I]Flow images from: "
    for idx, c in enumerate(classes):
        print "%d: %s" % (idx, c)


    if class_separation == False:
        generator = datagen.flow_from_directory(
            directory=dirpath, target_size=target_size, classes=classes,
            batch_size=batch_size, shuffle=shuffle, 
            save_to_dir=save_image_prefix)

        return generator

    else:

        if validation_include:
            train_gens =  [ datagen.flow_from_directory(
                directory=os.path.join(dirpath, "train"), target_size=target_size, classes=[c],
                batch_size=batch_size, shuffle=shuffle, 
                save_to_dir=save_image_prefix)
                for c in classes ]
            val_gens =  [ datagen.flow_from_directory(
                directory=os.path.join(dirpath, "validation"), target_size=target_size, classes=[c],
                batch_size=batch_size, shuffle=shuffle, 
                save_to_dir=save_image_prefix)
                for c in classes ]
            return train_gens, val_gens

        else:
            gens =  [ datagen.flow_from_directory(
                directory=dirpath, target_size=target_size, classes=[c],
                batch_size=batch_size, shuffle=shuffle, 
                save_to_dir=save_image_prefix)
                for c in classes ]
            return gens
