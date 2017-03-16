import os

def check_and_creator(dirpath):
    """Create directory if it is not exisiting. 

    :param dirpath: Directory path. 
    """
    if not os.path.exists(dirpath):
        print "[W]Not found '%s'. So newly created. " % dirpath
        os.makedirs(dirpath)