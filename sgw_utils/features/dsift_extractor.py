import sys, os
import numpy as np
import cv2

def denseSIFT(img, steps):
    """Output SIFT descriptors on dense keypoints lined up in a grid.

    :param img: Source image. 
    :param step: Dense steps of keypoints. 
    :return: Keypoints and SIFT descriptors on them. 
    """

    # get all dense keypoints
    # keypoints = get_denseKeypoints(img, max_dims)

    # descript with SIFT for all keypoints

    dense_detector = cv2.FeatureDetector_create("Dense")

    # set parameters for dense point detector
    dense_detector.setDouble("initFeatureScale", 4)
    dense_detector.setInt("initXyStep", steps)
    dense_detector.setInt("initImgBound", int((steps+1)/2) )

    # get dense points as keypoints
    kp = dense_detector.detect(img)

    # Note: requires libopencv-nonfree-dev, python-opencv, libopencv-dev
    sift_extractor = cv2.DescriptorExtractor_create("SIFT")

    # extract SIFT descriptor for each keypoints
    kp, des = sift_extractor.compute(img, kp)

    return kp, des


if __name__ == "__main__":
    opt, args = OptParser()

    img = cv2.imread(opt.inPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = denseSIFT(gray, opt.steps)