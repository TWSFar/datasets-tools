import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [Extension("nms_gpu", ["nms_gpu.pyx"], extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]),
              Extension("bbox_gpu", ["bbox_gpu.pyx"], extra_compile_args=["-Wno-cpp", "-Wno-unused-function"])]


setup(
    name="gpu_functions",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
