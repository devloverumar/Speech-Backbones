""" from https://github.com/jaywalnut310/glow-tts """

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'monotonic_align',
    ext_modules = cythonize("core.pyx", compiler_directives={'language_level': "3"}, annotate=True, verbose=True),
    include_dirs=[numpy.get_include()]

)
