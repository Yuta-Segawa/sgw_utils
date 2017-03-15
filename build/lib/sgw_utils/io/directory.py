import os

def check_and_creator(dirpath):
    if not os.path.exists(dirpath):
        print "[W]Not found '%s'. So newly created. " % dirpath
        os.makedirs(dirpath)