import sys, os
import numpy as np
import cv2

"""
option parser
"""
def OptParser():

    from optparse import OptionParser
    from optparse import OptionGroup

    # description
    psr = OptionParser(version="0.1", 
                       description="Description: This module provides an image feature of dense SIFT based fisher vector. "
                                    +"This can be used as main module for a test, not only as a imported library.")

    # essencial options
    essencial_grp = OptionGroup(psr, "Essensial")
    essencial_grp.add_option("-i", "--input", dest="inPath", default=None, type="string",
                    help = "[string]Input image path which must be specified.")
    psr.add_option_group(essencial_grp)

    # arbitary options
    arbitary_grp = OptionGroup(psr, "Arbitary")
#    arbitary_grp.add_option("--output", dest="outPath", default=None, type="string",
#                    help = "[string]Output file path. if no specified, output file will be written on the current directory. ")
    arbitary_grp.add_option("--dense_steps", dest="steps", default=30, type="int",
                    help = "[int]Steps for dense sampling on an input image. ")
    psr.add_option_group(arbitary_grp)

    # flags
    flags_grp = OptionGroup(psr, "Flags")
    flags_grp.add_option("-s", "--show_params", dest="flag_SHOWPARAMS", default=False, action="store_true", 
                    help = "A flag to check the specified parameters. ")
    psr.add_option_group(flags_grp)

    # parse options and arguments
    (opt, args) = psr.parse_args()

    # check essencial options
    if opt.inPath == None:
        psr.print_help()
        quit()

    # check whether input file exists
    if not os.path.exists(opt.inPath):
        print ("%s is not found. Try again." % opt.inPath)
        quit()

    # show parameter if the flag is True
    if opt.flag_SHOWPARAMS:
        print "inPath: \n  "+str(opt.inPath)
        print "max_dims: \n  "+str(opt.max_dims)

    return opt, args

"""
quickshow
"""
def quickshow(img):
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", np.uint8(img))
    cv2.waitKey(0)

def denseSIFT(img, steps):
    """
    Output SIFT vector for each dense sampled keypoints.
    """

    # get all dense keypoints
    # keypoints = get_denseKeypoints(img, max_dims)

    # descript with SIFT for all keypoints

    dense_detector = cv2.FeatureDetector_create("Dense")

    # set parameters for dense point detector
    dense_detector.setDouble("initFeatureScale", 4)
    dense_detector.setInt("initXyStep", steps)
    dense_detector.setInt("initImgBound", int((steps+1)/2) )
    """
    f = '{} ({}): {}'
    for param in dense_detector.getParams():
        type_ = dense_detector.paramType(param)
        if type_ == cv2.PARAM_BOOLEAN:
            print f.format(param, 'boolean', dense_detector.getBool(param))
        elif type_ == cv2.PARAM_INT:
            print f.format(param, 'int', dense_detector.getInt(param))
        elif type_ == cv2.PARAM_REAL:
            print f.format(param, 'real', dense_detector.getDouble(param))
        else:
            print param
    """

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



"""
def draw_rect_for_dense_points(img, steps_along_height, steps_along_width):

    points_tl = [ (x, y) 
                    for y in xrange(0, img.shape[0], steps_along_height) 
                    for x in xrange(0, img.shape[1], steps_along_width) ]

    points_br = [ (x+2, y+2) for x, y in points_tl ]

    for tl, br in zip(points_tl, points_br):
       cv2.rectangle(img, tl, br, (0, 0, 255) )

    quickshow(img)


def get_denseKeypoints(img, max_dims):
    # calculate dense sampling step from returned max_dims
    sift_dims = 128
    aspect_ratio = float(img.shape[0]) / float(img.shape[1])
    print "aspect_ratio: "+str(aspect_ratio)

    possible_points = int(max_dims / sift_dims)
    print "possible_points: "+str(possible_points)
    points_along_height = int(possible_points * aspect_ratio)
    points_along_width = possible_points - points_along_height
    print "points_along_height: "+str(points_along_height)
    print "points_along_width: "+str(points_along_width)

    steps_along_height = int(img.shape[0] / points_along_height)
    steps_along_width =  int(img.shape[1] / points_along_width)

    print "steps_along_height: "+str(steps_along_height)
    print "steps_along_width: "+str(steps_along_width)

    # top left of each 4x4 rectangulars
    keypoints_tl = [ (x, y) 
                    for y in xrange(0, img.shape[0], steps_along_height) 
                    for x in xrange(0, img.shape[1], steps_along_width) ]

    return keypoints_tl
"""