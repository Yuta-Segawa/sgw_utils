import os, glob, cv2, random
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import NumpyArrayIterator
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

class ImageNetDataGenerator(ImageDataGenerator):
    """Overloaded class on Keras 'ImageDataGenerator', 
    which is aiming at including preprocess for 'inception' and 'vgg'. 
    
    self.rescale works as a kerword of preprocessing type like: 

        - In case self.rescale = 'inception', preprocess on image in the same manner as GoogLeNet. 
        - In case self.rescale = 'vgg', preprocess on image in the same manner as VGG16. 

    """
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
    """Overloaded class on ImageDataGenerator for avoiding from any data augmentations. 
    """

    def random_transform(self, x):
        # just returns the value without any transformations 
        return x


class NumpyArrayIterator_LargeChannelAdapted(NumpyArrayIterator):
    """Overloaded class on NumpyArrayIterator in order to let further data input even not formatted as images.   
        by making a comment-out for exceptions on the input dimensionality. 
    """

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

def generator_preparation(images, labels, 
    batch_size=100, preprocessing_type='inception', shuffle=False, 
    save_image_prefix=None):
    """Create generator from images in a single directory. 

    :param images: Images to be stacked on a quene of a generator. 
    :param labels: Corresponidng labels. 

        - If you use on Keras platform, you have to convert the label to one-hot format in advance. 
    :param batch_size: Batch size when feeding from generator. 
    :param type: Preprocessing type of images. See the documentation of 'image' module for the details.
    :param shuffle: Flag to shuffle set of image and label before constructing a generator. 
    :param save_image_prefix: See 'Image Preprocessing' documentation in Keras because of Keras-based implementation. 
    :retrun: Generator object based on Keras. 
    """

    # transfromation in the preprocessing
    datagen = ImageDataGenerator()

    images = preprocess_on_images(images, type)

    # fit the images to generator in order to image preprocessing
    # datagen.fit(images)
    generator = datagen.flow(X=images, y=labels, 
        batch_size=batch_size, shuffle=shuffle, 
        save_to_dir=save_image_prefix)

    return generator



def generator_preparation_from_dirs(dirpath, target_size, classes=None, 
    batch_size=100, shuffle=False, preprocessing_type='inception',
    save_image_prefix=None, class_separation=False, validation_include=True):
    """Crate generators from at least one directories including images. 

    :param dirpath: Base directory containing sub-directories. 
    :param target_size: Shape for resizing images. None works as keep the original shape.
    :param class: List of the sub-directories' naems arranged in order of class labels. None works as to load images from 'dirpath' directory (ingore any sub-directories). 
    :param batch_size: Batch size when feeding from generator. 
    :param preprocessing_type: Preprocessing type of images. See the documentation of 'image' module for the details.
    :param shuffle: Flag to shuffle set of image and label before constructing a generator. 
    :param save_image_prefix: See 'Image Preprocessing' documentation in Keras because of Keras-based implementation. 
    :param class_separation: Flag to separate generators into classes. This flag is usually used in except training phase. Consequently, returned objects are arranged as list of generators for each class. 
    :param validation_include: Flag to load images from subdirectories under each of 'train' and 'validation' directories. This flag is usually used only in training phase. 
    :retrun: Generato object based on Keras. 
   """    


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
