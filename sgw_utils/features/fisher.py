#Author: Jacob Gildenblat, 2014
#License: you may use this for whatever you like 
import os, sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
import dsift_extractor as dsift

def dictionary(descriptors, N):
	em = cv2.EM(N)
	em.train(descriptors)

	return np.float32(em.getMat("means")), \
		np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]

def image_descriptors_original(file):
	img = cv2.imread(file, 0)
	img = cv2.resize(img, (256, 256))
	_ , descriptors = cv2.SIFT().detectAndCompute(img, None)
	return descriptors

"""
my function (segawa)
"""
def image_descriptors(file, dense_steps=30):
	img = cv2.imread(file, 0)
	img = cv2.resize(img, (256, 256))
	_ , descriptors = dsift.denseSIFT(img, dense_steps)
	return descriptors


def folder_descriptors_original(folder):
	files = glob.glob(folder + "/*.jpg")
	print("Calculating descriptos. Number of images is", len(files))
	return np.concatenate([image_descriptors(file) for file in files])

"""
my function (segawa)
"""
def folder_descriptors(folder, dense_steps=30, extension="jpg"):
	files = sorted(glob.glob(folder + "/*.%s" % extension))
	print("Calculating descriptors. Number of images is %d. " % len(files))
	return np.concatenate([image_descriptors(file, dense_steps) for file in files])

def likelihood_moment(x, ytk, moment):	
	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
	return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
	gaussians, s0, s1,s2 = {}, {}, {}, {}
	samples = zip(range(0, len(samples)), samples)
	
	g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
	for index, x in samples:
		gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

	for k in range(0, len(weights)):
		s0[k], s1[k], s2[k] = 0, 0, 0
		for index, x in samples:
			probabilities = np.multiply(gaussians[index], weights)
			probabilities = probabilities / np.sum(probabilities)
			s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
			s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
			s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

	return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
	return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
	T = samples.shape[0]
	covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
	a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
	b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
	c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
	fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
	fv = normalize(fv)
	return fv

def generate_gmm(input_folder, working_folder, N=5, dense_steps=30, extension="jpg"):
	"""Fit a GMM(Gaussian mixture model) from dense SIFT descriptors and save its parameters. 

	:param input_folder: Path to folder containing images. 
	:param working_folder: Path to folder where the model parameters are saved in. 
	:param N: Mixed number of the model. 
	:param dense_steps: Dense steps of keypoints to be used for SIFT feature extraction. 
	:param extension: Suffix of images. 
	:return: GMM parameters which are means, covariances, and mixture weights. 
	"""

	words = np.concatenate([folder_descriptors(folder, dense_steps, extension) for folder in sorted(glob.glob(input_folder + '/*'))])
	print("Training GMM of size", N)
	means, covs, weights = dictionary(words, N)
	#Throw away gaussians with weights that are too small:
	th = 1.0 / N
	means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

	np.save(working_folder + "/means.gmm.npy", means)
	np.save(working_folder + "/covs.gmm.npy", covs)
	np.save(working_folder + "/weights.gmm.npy", weights)
	return means, covs, weights

def load_gmm(working_folder):
	"""Load GMM parameters which were calculated at training time. 

	:param working_folder: Path to folder including 'means.gmm', 'covs.gmm', and 'weights.gmm'. 
	:return: GMM parameters which are means, covariances, and mixture weights. 
	"""
	print("Load GMM parameters from %s. " % working_folder)
	means   = np.load(working_folder + "/means.gmm.npy")
	covs    = np.load(working_folder + "/covs.gmm.npy")
	weights = np.load(working_folder + "/weights.gmm.npy")
	return means, covs, weights

def get_fisher_vectors_from_folder_original(folder, gmm):
	files = glob.glob(folder + "/*.jpg")
	return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])

def get_fisher_vectors_from_folder(folder, gmm, dense_steps = 30, extension="jpg"):
	files = sorted(glob.glob(folder + "/*.%s"%extension))
	return np.float32([fisher_vector(image_descriptors(file, dense_steps), *gmm) for file in files])

def fisher_features(input_folder, working_folder, gmm, dense_steps = 30, extension="jpg", file="fisher_features.npy"):
	"""Calcluate fisher features on images with GMM parameters estimated in advance. 

	This is usually called by feature_select_switcher. 

	:param input_folder: Path to folder containing sud directories containing images. If this folder has no directories but images, the images are directly loaded with label 0.
	:param working_folder: Path to folder where the model parameters are saved in. 
	:param gmm: GMM model as a parameter tuple of (means, covariances, mixture weights). 
	:param dense_steps: Dense steps of keypoints to be used for SIFT feature extraction. 
	:param extension: Suffix of images. 
	:param file: Filename of features. This is used when saving. 
	:return: Calculated fisher features as numpy array in shape of (classes_num, smaples_num, dimensionality). 
	"""
	folders = [ d for d in sorted(glob.glob(input_folder + "/*")) if not '.%s' % extension in d ]
	if len(folders) == 0:
		folders = [input_folder]

	features = np.array([get_fisher_vectors_from_folder(f, gmm, dense_steps, extension=extension) for f in folders])

	if working_folder:
		fmt = 'Images in {} labeled as {}. '
		for input_folder, label in zip(folders, range(0, len(folders))):
			print fmt.format(input_folder, label)

		np.save(os.path.join(working_folder, file), features)

	return np.squeeze(features)
	
def load_fisher_features(file = "fisher_features.npy"):
	print("Read file '%s' in the workspace to get fisher features." % file)
	features = np.load( file )
	return features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d' , "--dir", help="Directory with images" , default='.')
    parser.add_argument("-g" , "--loadgmm" , help="Load Gmm dictionary", action = 'store_true', default = False)
    parser.add_argument('-n' , "--number", help="Number of words in dictionary" , default=5, type=int)
    args = parser.parse_args()
    return args
if __name__ == "__main__":
	args = get_args()
	working_folder = args.dir

	gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(working_folder, working_folder, args.number)
	fisher_features = fisher_features(working_folder, gmm)
	#TBD, split the features into training and validation
	classifier = train(gmm, fisher_features)
	rate = success_rate(classifier, fisher_features)
	print("Success rate is", rate)
