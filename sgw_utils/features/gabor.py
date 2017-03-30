#Author: Jacob Gildenblat, 2014
#License: you may use this for whatever you like 
import os, sys, glob, argparse
import numpy as np
import math, cv2
import time
from sklearn import svm
from sklearn.externals import joblib
from scipy import signal
from skimage.filters import gabor_kernel
from multiprocessing import Pool
import itertools as it

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d' , "--dir", dest='dir', help="Directory with images" , default='.')
    parser.add_argument('-o' , "--output_dir", dest='output_dir', help="Output Directory. " , default='.')
    parser.add_argument("-g" , "--loadgf", dest='loadgf', help="Load Gabor features from a file. ", action = 'store_true', default = False)
    parser.add_argument('-e' , "--extension", dest='extension', help="Extension of images. " , default='jpg')

    return parser.parse_args()

# def get_gabor_map(shape, rot_degs=range(0, 180, 45)):

#     # determine parameters for DFT Gabor distribution
#     center_freq = 2.0**int(np.floor(np.log2(shape[0]))) * np.sqrt(2.0) * 0.25
#     gauss_mean = np.array([center_freq, 0.0])

#     freq_bandwidth = 1.0
#     angle_deg_bandwidth = 45.0
#     var_u = center_freq / np.sqrt(2) * (2.0**freq_bandwidth - 1.0) / (2.0**freq_bandwidth + 1.0)
#     var_v = center_freq / np.sqrt(2) * np.tan(np.radians(angle_deg_bandwidth / 2.0))
#     gauss_var = np.float32([var_u, var_v])

#     norm_term = 1.0 / ( 4.0 * np.pi**2 * np.prod(gauss_var) )

#     gabor_maps = []
#     for deg in rot_degs:
#         theta = np.radians(deg)
#         c, s = np.cos(theta), np.sin(theta)
#         R = np.matrix('{} {}; {} {}'.format(c, -s, s, c)).astype(float)

#         gabor_map = np.zeros(shape)
#         for u in range(0, shape[1]):
#             for v in range(0, shape[0]):
#                 inv_loc_uv = np.dot(R.T, np.float32([u, v]))

#                 # loc_mean = np.array([u_0, 0.0])
#                 shift_loc_minus = (inv_loc_uv - gauss_mean) / gauss_var
#                 shift_loc_plus  = (inv_loc_uv + gauss_mean) / gauss_var
#                 exp_term1 = np.exp(-0.5 * np.sum(np.power(shift_loc_minus,2)))
#                 exp_term2 = np.exp(-0.5 * np.sum(np.power(shift_loc_plus,2)))

#                 gabor_map[u, v] = norm_term * exp_term1 * exp_term2

#         gabor_maps += [gabor_map]

#     return gabor_maps

def normalize_with_tanh(array):
    return np.tanh(array)

def calc_texture_energy(array, w_size=5):

    summation_filter = np.ones([w_size, w_size]) / float(w_size**2)

    conved = np.float32([ signal.fftconvolve(patch, summation_filter, mode='same') for patch in array] )
    squated = conved * conved

    return squated

def preprocessing_on_batch(batch):
    preprocessed = batch / 255.0
    preprocessed -= 0.5
    preprocessed *= 2.0
    return preprocessed

def gabor_descriptor(args):

    # sample_batch, gabor_maps = args
    sample_batch, gabor_kernels = args
    # preprocessed_sample_batch = preprocessing_on_batch(sample_batch)

    # convolve_paris = np.array(list(it.product(preprocessed_sample_batch, gabor_kernels)))
    convolve_paris = np.array(list(it.product(sample_batch, gabor_kernels)))

    filtered_sample_batch = np.float32([ signal.fftconvolve(in1, in2, mode='same')
                                for in1, in2 in convolve_paris ])

    total_energy = filtered_sample_batch.reshape((len(sample_batch), -1, filtered_sample_batch.shape[-2], filtered_sample_batch.shape[-1]))
    total_energy = total_energy.sum(axis=1)
    
    # texture_energy = calc_texture_energy(total_energy)

    # normalized_sample_batch = normalize_with_tanh(texture_energy)
    normalized_sample_batch = normalize_with_tanh(total_energy)


    # # descrive gabor filter application
    # dft_imgs = np.fft.fft2(sample_batch)
    # dft_imgs = np.fft.fftshift(dft_imgs)

    # gabored_dft_imgs = np.float32([ np.multiply(dft_imgs, rotated_gabor_map) for rotated_gabor_map in gabor_maps ])
    # # swap so as to be (samples, directions, height, width)
    # gabored_dft_imgs = np.swapaxes(gabored_dft_imgs, 0, 1)

    # idft_imgs = np.fft.ifftshift(gabored_dft_imgs)

    # normalized = nonlinear_normalize(idft_imgs)

    # texture_energy = np.float32([ np.float32([calc_texture_energy(img) for img in img_for_directions]).flatten() 
    #                     for img_for_directions in  normalized ])

    # return texture_energy
    normalized_sample_batch = normalized_sample_batch.reshape(len(sample_batch), -1)
    # total_energy = total_energy.reshape(len(sample_batch), -1)
    return total_energy


def get_gabor_features_from_folder(folder, extension="jpg", max_threads=1, batch_size=100):

    files = sorted(glob.glob(folder + "/*.%s" % extension))
    print 'Calculate Gabor features for %d images. ' % len(files)
    images = [cv2.imread(fn, 0) for fn in files]
    # image_batches = np.expand_dims(images, axis=0)
    split_num = len(files) / batch_size
    if split_num > 0:
        image_batches = np.array_split(images, split_num)
        print 'Splitted %d batches respectively including %d images. ' % (len(image_batches), len(image_batches[0]))

    # gabor_maps = get_gabor_map(images[0].shape[:2], range(0, 180, 45))

    # func_args = zip(image_batches, [gabor_maps] * len(image_batches))

    # filter bank for:
    # u = 0.2, 0.3, and 0.5
    # theta = 0, 45, 90, 135
    frequencies = np.float32([0.2, 0.3, 0.5])
    degs = np.arange(0, 180, 45)*np.pi/180.0
    gabor_bank_params = list(it.product(frequencies, degs))
    gabor_kernels = [ gabor_kernel(frequency=freq, 
        theta=deg, bandwidth=1, 
        sigma_x=None, sigma_y=None, n_stds=3, offset=0).real 
        for freq, deg in gabor_bank_params ]

    func_args = zip(image_batches, [gabor_kernels] * len(image_batches))

    if max_threads == 1:
        return np.vstack([gabor_descriptor(args) for args in func_args])
    else:
        p = Pool(max_threads)
        return np.vstack(p.map(gabor_descriptor, func_args))


def get_gabor_features_from_folder_in_patchscale(folder, patch_size=32, extension="jpg", max_threads=1, batch_size=100):
    files = sorted(glob.glob(folder + "/*.%s" % extension))
    print 'Calculate Gabor features for %d images. ' % len(files)

    gabor_kernels = [ gabor_kernel(frequency=0.5, 
        theta=deg, bandwidth=1, 
        sigma_x=None, sigma_y=None, n_stds=3, offset=0).real 
        for deg in np.arange(0, 180, 45)*np.pi/180.0 ]

    features = []
    for frame_idx, file in enumerate(files):

        print "Processing on frame %d..." % (frame_idx+1)

        images = crop_imgs(cv2.imread(file, 0), patch_size=patch_size)
        image_batches = np.expand_dims(images, axis=0)
        split_num = len(images) / batch_size
        if split_num > 0:
            image_batches = np.array_split(np.uint8(images), split_num)
            print 'Splitted %d batches respectively including %d images. ' % (len(image_batches), len(image_batches[0]))

        func_args = zip(image_batches, [gabor_kernels] * len(image_batches))

        if max_threads == 1:
            frame_features = np.vstack([gabor_descriptor(*func_arg) for func_arg in func_args])
        else:
            from multiprocessing import Pool
            p = Pool(max_threads)
            frame_features = np.vstack(p.map(gabor_descriptor, func_args))
        features += frame_features

    return np.float32(features)

def gabor_features_in_patchscale(input_floder, working_folder, 
    patch_size=32, extension="jpg", file="gabor_features_for_evaluations.npy", max_threads=1):
    """Calcluate Gabor features on patches cropped from images along scan line. 

    :param input_folder: Path to folder over sub directories containing images. 
    :param working_folder: Path to folder where the model parameters are saved in. 
    :param patch_size: Size of patch width and height, which is used for cropping patches from images. 
    :param extension: Suffix of images. 
    :param file: Filename of features. This is used when saving. 
    :param max_threads: Max threads for multiprocessing. 
    :return: Calculated fisher features as numpy array in shape of (classes_num, smaples_num, dimensionality). 
    """


    # Get subdirectories
    folders = sorted(glob.glob(input_floder + "/*"))

    print "Found %d classes. " % len(folders)
    features = [get_gabor_features_from_folder_in_patchscale(f, extension=extension, patch_size=patch_size, max_threads=max_threads) 
            for f in folders]

    output_feature_filename = os.path.join(working_folder, file)
    np.save(output_feature_filename, features.values())
    print "Saved calculated features as %s. " % output_feature_filename

    return features

def load_gabor_features(file = "gabor_features.npy"):
	print("Read file '%s' in the workspace to get gabor features." % file)
	features = np.load( file )
	return features
# For training: find directories of 2 classes and calculate features in whole image region
def gabor_features(input_folder, working_folder, extension="jpg", file="gabor_features.npy", max_threads=1):
    """Calcluate Gabor features on images. 

    :param input_folder: Path to folder over sub directories containing images. 
    :param working_folder: Path to folder where the model parameters are saved in. 
    :param extension: Suffix of images. 
    :param file: Filename of features. This is used when saving. 
    :param max_threads: Max threads for multiprocessing. 
    :return: Calculated fisher features as numpy array in shape of (classes_num, smaples_num, dimensionality). 
    """

    # Get subdirectories
    folders = sorted(glob.glob(input_folder + "/*"))
    print folders

    fmt = 'Images in {} will be labeled as {}. '
    for label, folder in enumerate(folders):
        print fmt.format(folder, label)

    print "Found %d class. " % len(folders)
    features = [get_gabor_features_from_folder(f, 
                    extension=extension, max_threads=max_threads) for f in folders]

    output_feature_filename = os.path.join(working_folder, file)
    np.save(output_feature_filename, features)
    print "Saved calculated features as %s. " % output_feature_filename

    return features


if __name__ == "__main__":
    args = get_args()
    gabor_features = gabor_features(args.dir, args.output_dir, extension=args.extension, file="gabor_features.npy") if not args.loadgf else load_gabor_features(args.output_dir)
    #TBD, split the features into training and validation
    classifier = train(gabor_features)
    rate = success_rate(classifier, gabor_features)
    print("Success rate is", rate)
    save_svm(classifier, args.output_dir + "/svm.pkl")
