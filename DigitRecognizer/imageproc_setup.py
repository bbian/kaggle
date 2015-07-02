from distutils.core import setup, Extension
import numpy

# define the extension module
imageproc_np = Extension('imageproc_np', sources=['imageproc_np.c'],
                          include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[imageproc_np])

