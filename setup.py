import os
import sys
import setuptools

import numpy.distutils.core

# helful links:
# https://stackoverflow.com/questions/31043774/customize-location-of-so-file-generated-by-cython
# https://docs.scipy.org/doc/numpy/reference/distutils.html
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compilation
# https://docs.scipy.org/doc/numpy/f2py/distutils.html
# https://stackoverflow.com/questions/22404060/fortran-cython-workflow

# =================================== #
# 1. Compile the pure Fortran modules #
# =================================== #
os.system("gfortran aquacrop/native/types.f90 -c -o aquacrop/native/types.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran aquacrop/native/soil_evaporation.f90 -c -o aquacrop/native/soil_evaporation.o -O3 -fPIC -fbounds-check -mtune=native")

# =================================== #
# 2. Compile the f2py wrappers        #
# =================================== #

# TODO: check rabascus package to see how to link with OpenMP

f90_fnames = [
    'soil_evaporation_w.f90'
    ]

f90_paths = []
for fname in f90_fnames:
    f90_paths.append( 'aquacrop/native/' + fname )

f90_flags = ["-fPIC", "-O3", "-fbounds-check", "-mtune=native"]

# make Extension object which links with the pure Fortran modules compiled above
ext1 = numpy.distutils.core.Extension(
    name = 'aquacrop_fc',
    sources = f90_paths,
    extra_f90_compile_args = f90_flags,
    # extra_link_args = omp_lib,
    extra_link_args=['aquacrop/native/types.o','aquacrop/native/soil_evaporation.o']
    )

# =================================== #
# 3. run setup                        #
# =================================== #

numpy.distutils.core.setup(
    name='aquacrop',
    version=0.1,
    description='Python implementation of FAO AquaCrop',
    url='https://github.com/simonmoulds/aquacrop',
    author='Simon Moulds',
    author_email='sim.moulds@gmail.com',
    license='GPL',
    packages=['aquacrop'],
    ext_modules = [ext1],
    python_requires = '>=3.7.*',
    zip_safe=False)

# =================================== #
# 4. Clean up                         #
# =================================== #

os.system("rm -rf aquacrop/native/types.o")
os.system("rm -rf aquacrop/native/soil_evaporation.o")
# os.system("rm -rf aquacrop/native/types.mod")
# os.system("rm -rf aquacrop/native/soil_evaporation.mod")

# not used:

# rabascus:

# numpy.distutils.core.setup(
#     install_requires = ['quantities', 'scipy', 'h5py'],
#     name = 'rabacus',
#     version = '0.9.5',
#     description = description,
#     long_description = long_description,
#     url = 'https://bitbucket.org/galtay/rabacus',
#     download_url = 'https://pypi.python.org/pypi/rabacus',
#     license = 'Free BSD',
#     platforms = 'linux',
#     author = 'Gabriel Altay',
#     author_email = 'gabriel.altay@gmail.com',
#     classifiers = [
#         "Programming Language :: Python",
#         "Programming Language :: Fortran",
#         "License :: OSI Approved :: BSD License",
#         "Operating System :: POSIX :: Linux",
#         "Topic :: Scientific/Engineering :: Astronomy",
#         "Topic :: Scientific/Engineering :: Physics",
#         "Intended Audience :: Science/Research",
#         "Development Status :: 4 - Beta",
#         "Topic :: Education",
#         "Natural Language :: English",
#         ],
#     packages = setuptools.find_packages(),
#     package_data = {'': ['*.f90','*.out','*.dat','*.minimum']},
#     ext_modules = [ext1],
# )
