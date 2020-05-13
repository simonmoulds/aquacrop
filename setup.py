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

os.system("rm -rf ./*.so")
os.system("rm -rf ./build")
os.system("rm -rf ./*.egg-info")

# =================================== #
# 1. Compile the pure Fortran modules #
# =================================== #
os.chdir("aquacrop/native")
os.system("gfortran types.f90 -c -o types.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran soil_evaporation.f90 -c -o soil_evaporation.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran temperature_stress.f90 -c -o temperature_stress.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran biomass_accumulation.f90 -c -o biomass_accumulation.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran water_stress.f90 -c -o water_stress.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran canopy_cover.f90 -c -o canopy_cover.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran capillary_rise.f90 -c -o capillary_rise.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran check_gw_table.f90 -c -o check_gw_table.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran drainage.f90 -c -o drainage.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran germination.f90 -c -o germination.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran gdd.f90 -c -o gdd.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran growth_stage.f90 -c -o growth_stage.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran harvest_index.f90 -c -o harvest_index.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran infiltration.f90 -c -o infiltration.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran inflow.f90 -c -o inflow.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran pre_irr.f90 -c -o pre_irr.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran rainfall_partition.f90 -c -o rainfall_partition.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran root_dev.f90 -c -o root_dev.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran root_zone_water.f90 -c -o root_zone_water.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran transpiration.f90 -c -o transpiration.o -O3 -fPIC -fbounds-check -mtune=native")
os.system("gfortran crop_parameters.f90 -c -o crop_parameters.o -O3 -fPIC -fbounds-check -mtune=native")
os.chdir("../..")

# =================================== #
# 2. Compile the f2py wrappers        #
# =================================== #

# TODO: check rabascus package to see how to link with OpenMP

f90_fnames = [
    'types.f90',
    'soil_evaporation_w.f90',
    'biomass_accumulation_w.f90',
    'water_stress_w.f90',
    'canopy_cover_w.f90',
    'capillary_rise_w.f90',
    'check_gw_table_w.f90',
    'drainage_w.f90',
    'germination_w.f90',
    'gdd_w.f90',
    'growth_stage_w.f90',
    'harvest_index_w.f90',
    'infiltration_w.f90',
    'inflow_w.f90',
    'pre_irr_w.f90',
    'rainfall_partition_w.f90',
    'root_dev_w.f90',
    'root_zone_water_w.f90',
    'transpiration_w.f90',
    'crop_parameters_w.f90'
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
    extra_link_args=['aquacrop/native/soil_evaporation.o','aquacrop/native/temperature_stress.o','aquacrop/native/biomass_accumulation.o','aquacrop/native/water_stress.o','aquacrop/native/canopy_cover.o','aquacrop/native/capillary_rise.o','aquacrop/native/check_gw_table.o','aquacrop/native/drainage.o','aquacrop/native/germination.o','aquacrop/native/gdd.o','aquacrop/native/growth_stage.o','aquacrop/native/harvest_index.o','aquacrop/native/infiltration.o','aquacrop/native/inflow.o','aquacrop/native/pre_irr.o','aquacrop/native/rainfall_partition.o','aquacrop/native/root_dev.o','aquacrop/native/root_zone_water.o','aquacrop/native/transpiration.o','aquacrop/native/crop_parameters.o']
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
    install_requires=[
        'click'
    ],
    entry_points='''
        [console_scripts]
        aquacrop=aquacrop.cli.aquacrop:cli
    ''',
    ext_modules = [ext1],
    python_requires = '>=3.7.*',
    zip_safe=False)

# =================================== #
# 4. Clean up                         #
# =================================== #

os.system("rm -rf aquacrop/native/*.o")
os.system("rm -rf aquacrop/native/*.mod")

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
