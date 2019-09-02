import os
import sys
import setuptools

import numpy.distutils.core

f90_fnames = [
    'soil_evaporation.f90'
    ]

f90_paths = []
for fname in f90_fnames:
    f90_paths.append( 'aquacrop/f2py/' + fname )

# # rabascus:
# f90_flags = ["-fopenmp", "-fPIC", "-O3", "-fbounds-check", "-mtune=native"]
f90_flags = ["-fPIC", "-O3", "-fbounds-check", "-mtune=native"]

ext1 = numpy.distutils.core.Extension(
    name = 'aquacrop_fc',
    sources = f90_paths,
    extra_f90_compile_args = f90_flags# ,
    # extra_link_args = omp_lib,
    )

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
