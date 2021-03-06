import os, glob, cv2, random
import numpy as np
from keras.preprocessing import image
from multiprocessing import Process, Queue, Pool
from keras.preprocessing.image import ImageDataGenerator
import itertools as it
from keras.applications.imagenet_utils import preprocess_input
from sgw_utils.features import gabor, fisher

def get_crop_locs(shape, patch_size=32):
    """Get locations where patches are cropped in a source image. 

    :param shape: Shape of source image you suppose. 
    :param patch_size: The edge length of cropped patches. 
    :return: Cropping location as list of ((tl_y tl_x), (br_y, br_x)). 
    """
    tl_x_locs = np.arange(0, shape[1], patch_size)
    tl_y_locs = np.arange(0, shape[0], patch_size)
    tl_locs = np.array(list(it.product(tl_y_locs, tl_x_locs)))
    br_locs = tl_locs + np.array([patch_size, patch_size])
    return zip(tl_locs, br_locs)

def get_crop_masks(shape, patch_size=32):
    """Get masks for cropping regions in a source image. 

    :param shape: Shape of source image you suppose. 
    :param patch_size: The edge length of cropped patches. 
    :return: List of the masks of numpy array. 
    """
    locs = get_crop_locs(shape, patch_size)
    loc_masks = []
    for l in locs: 
        mask = np.zeros(shape[:2]).astype(bool)
        mask[l[0][0]:l[1][0], l[0][1]:l[1][1]] = 1
        loc_masks.append(mask)
    return loc_masks

def crop_imgs(img, patch_size=32):
    """Crop patches from the source image along the scan lines. 
    
    :param img: Source image. 
    :param patch_size: The edge length of cropped patches. 
    :return: List of cropped patches. 
    """
    locs = get_crop_locs(shape, patch_size)
    mirrored_img = np.hstack([np.vstack([img, img[::-1]]), np.vstack([img[::, ::-1], img[::-1, ::-1]])])
    return [mirrored_img[tl[0]:br[0], tl[1]:br[1]] for tl, br in locs]

def resize_and_transpose(img_fn, shape):
    return image.img_to_array(image.load_img(img_fn, target_size=shape)).astype(float)
def mp_func(args):
    return resize_and_transpose(args[0], args[1])

def load_images_from_dirs(basedir, classnames, 
    image_shape=(299, 299), extension="*", max_smps=-1, max_threads=1, vervosity=1):
    """Load images from sub-directories and pre-resize them. 

    :param basedir: Base directory containing sub-directories including images to be loaded. 
    :param classnames: List of the sub-directories' naems arranged in order of class labels.  
    :param image_shape: Shape of loaded images for resizing. 
    :param extension: Suffix of images to be loaded. 
    :param max_smps: Limitation of images to be loaded. Default of '-1' means the entire images will be loaded.  
    :param max_threads: Max threads for multiprocessing. Default of '1' means the multiprocessing will be disable. 
    :param bool vervosity: Vervosity on displaying status. 
    :return: Loaded and resized images as list of numpy array in shape of (samples_num, image_shape[0], image_shape[1]). 
    """

    # get filenames from directories of the classes
    dirnames = [ os.path.join(basedir, cls) for cls in classnames 
                if os.path.exists(os.path.join(basedir, cls)) ]
    if not len(dirnames) == len(classnames):
        print("[E]Too few classes are found (expected %d classes, but %d). " % (len(classnames), len(dirnames)))
        return None
    img_filenames = [ sorted(glob.glob(os.path.join(dirname, "*.%s" % extension))) 
                    for dirname in dirnames ]
    # display info
    for idx, (found_dir, fns) in enumerate(zip(dirnames, img_filenames)):
        if vervosity:
            print("[I]Found %d images in %s. labeled them as %d. " % (len(fns), found_dir, idx))
        if max_smps != -1: 
            max_smps = min(len(fns), max_smps) 

    # load all images
    mp_status = 'enabled with %d threads' % max_threads if not max_threads == 1 else 'disable'
    if vervosity:
        print("[I]Resize with shape %s" % str(image_shape))
        print("[I]( Multi-Processing: %s)" % mp_status)

    loaded_imgs = []
    for i, cls_img_fns in enumerate(img_filenames):

        if max_smps != -1:
            random.shuffle(cls_img_fns)
            cls_img_fns = cls_img_fns[0:max_smps]

        if vervosity: 
            print("[I]Resizing %d images belonging to the class %d. " % (len(cls_img_fns), i))

        if max_threads > 1:
            p = Pool(max_threads)
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
    if vervosity:
        print("[I]Loaded all images and labels. ")

    return loaded_imgs, one_hot_vectors

# for prediction
def load_images_in_dir(dirname, 
    image_shape=(299, 299), extension="*", max_smps=-1, max_threads=1, vervosity=1):
    """Load images from a single directory and pre-resize them. 

    :param dirname: Directory including images to be loaded. 
    :param image_shape: Shape of loaded images for resizing. 
    :param extension: Suffix of images to be loaded. 
    :param max_smps: Limitation of images to be loaded. Default of '-1' means the entire images will be loaded.  
    :param max_threads: Max threads for multiprocessing. Default of '1' means the multiprocessing will be disable. 
    :param bool vervosity: Vervosity on displaying status. 
    :return: Loaded and resized images as numpy array in shape of (classes_num, samples_num, image_shape[0], image_shape[1]). 
    """

    # get filenames from directories of the classes
    img_filenames = sorted(glob.glob(os.path.join(dirname, "*.%s" % extension)))

    # display info
    if vervosity:
        print("[I]Found %d images in %s. " % (len(img_filenames), dirname))
    if max_smps != -1: 
        max_smps = min(len(img_filenames), max_smps) 

    loaded_imgs = []
    if max_smps != -1:
        random.shuffle(img_filenames)
        img_filenames = img_filenames[0:max_smps]

    if vervosity:
        print("[I]Use %d images. " % len(img_filenames))

    # load all images
    if vervosity:
        if image_shape is None:
            print("[I]Not any resizing. ")
        else: 
            print("[I]Resizing with shape %s..." % str(image_shape))
    mp_status = 'enabled with %d threads' % max_threads if not max_threads == 1 else 'disable'
    if vervosity:
        print("[I]( Multi-Processing: %s)" % mp_status)


    if max_threads > 1:
        p = Pool(max_threads)
        loaded_imgs = np.array(p.map(mp_func, zip(img_filenames, [image_shape]*len(img_filenames)))).astype(float)
        p.close()
    else:
        loaded_imgs = np.array([mp_func(args) for args in zip(img_filenames, [image_shape]*len(img_filenames))]).astype(float)

    return loaded_imgs

def preprocess_on_images(images, type='inception', vervosity=1):
    """Preprocess on the images by several types of method. 

    :param images: Target images. Numpy array of multiple images like (sample_num, height, width) is also allowed to be this input. 
    :param type: Preprocessing type. You can choose from 'inception', 'vgg', and 'disable': 

            - inception: Apply GoogLeNet based method that makes values supressed within (-1.0, 1.0). 
            - vgg: Apply VGG based method that subtracts specific values from image values in RBG respectively. This has been implemented on Keras. 
            - disable: Not apply any methods of preprocessing. 

    :param bool vervosity: Vervosity on displaying status. 
    :return: Loaded and resized images as numpy array in shape of (samples_num, image_shape[0], image_shape[1]). 
    """

    # determinant preprocessing
    if vervosity:
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

def feature_select_switcher(feature_keyword, 
    image_dir=None, output_dir="./", identifier=None, extension="jpg", max_threads=1, load_GMM_dir=None):
    """Helper to easily determine which feature to be returned. 
    
    :param feature_keyword: Keyword for feature data which will be returned finally: 

        - If a path to a file of features is given, this will return loaded numpy array. 
        - If a type of feature and image_dir are given, this will return calculated features with the type. 
            - Now 'gabor' and 'fisher' are available as feature types. 
        - If an unrecognized string or None is given, this will return None. 
    
    :param image_dir: Directory including images for calculating features. This will be ignored when feature file is already existing. See fisher and gabor module for more detail. 
    :param output_dir: Output directory for calculated feature file. This will be also ignored when feature file is already existing. 
    :param identifier: Identifier of file name of features. This will be used as like "<identifier>_<feature_keyword>_features.npy"
    :param extension: Suffix of images to be loaded. 
    :param max_threads: Max threads for multiprocessing. Default of '1' means the multiprocessing will be disable. 
    :param load_GMM_dir: (ignored except fisher feature)Directory of GMM paramterers which were calcluated at training time in advance. In general this flag should work when evaluation. 
    :return: Calculated or loaded features as numpy array. 
    """

    return_features = None
 
    # 1) load from file if the option has a valid path
    if os.path.exists(feature_keyword):
        print "[I]Load from the feature file: %s" % feature_keyword
        return_features = np.load(feature_keyword)

    # 2) obtain features by calculation or loading from an existing file
    available_feature_types = ['gabor', 'fisher']
    if feature_keyword in available_feature_types:
        name_elms = [ elm for elm in [identifier, feature_keyword, "features.npy"] if elm is not None]
        feature_filename = os.path.join(output_dir, "_".join(name_elms))
    
        # calculate features if the option has a valid feature type
        if feature_keyword == "gabor":
            print "[I]Feature type: Gabor"
            return_features = gabor.gabor_features(image_dir, output_dir, 
                extension=extension, file=feature_filename, max_threads=max_threads) \
                if not os.path.exists(feature_filename) else gabor.load_gabor_features(feature_filename)
        elif feature_keyword == 'fisher':
            print "[I]Feature type: Fisher"
            gmm = fisher.generate_gmm(image_dir, output_dir) \
                if not load_GMM_dir else fisher.load_gmm(load_GMM_dir)
            return_features = fisher.fisher_features(image_dir, output_dir, gmm, 
                extension=extension, file=feature_filename) \
                if not os.path.exists(feature_filename) else fisher.load_fisher_features(feature_filename)

    # 3) exception
    if return_features is None:
        print "[E]Couldn't obtain features. "
        raise IOError

    return return_features