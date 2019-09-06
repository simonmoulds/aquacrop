from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# This line only needed if building with NumPy in Cython file.
from numpy import get_include
from os import system

# compile the fortran modules without linking
fortran_mod_comp = 'gfortran soil_evaporation.f90 -c -o soil_evaporation.o -O3 -fPIC'
print(fortran_mod_comp)
system(fortran_mod_comp)
shared_obj_comp = 'gfortran surf_evap_interface.f90 -c -o surf_evap_interface.o -O3 -fPIC'
print(shared_obj_comp)
system(shared_obj_comp)

ext_modules = [Extension(# module name:
                         'surf_evap',
                         # 'test.pygfunc',
                         # source file:
                         ['surf_evap.pyx'],
                         # other compile args for gcc
                         extra_compile_args=['-fPIC', '-O3'],
                         # other files to link to
                         extra_link_args=['soil_evaporation.o', 'surf_evap_interface.o'])]

setup(name = 'surf_evap',
      cmdclass = {'build_ext': build_ext},
      # Needed if building with NumPy.
      # This includes the NumPy headers when compiling.
      include_dirs = [get_include()],
      ext_modules = ext_modules)
