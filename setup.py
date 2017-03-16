# from distutils.core import setup
from setuptools import setup, find_packages
from sgw_utils import __author__, __version__, __license__
setup(name='sgw_utils',
      version=__version__,
      description='My package for experiments. ',
      license=__license__,
      url='https://github.com/Yuta-Segawa/sgw_utils.git',
      author='__author__',
      author_email='kawalab.14.segawa@gmail.com',
      packages=find_packages()
     )