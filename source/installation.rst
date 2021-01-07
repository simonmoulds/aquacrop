Install
=======

.. currentmodule:: aquacrop

AquaCrop-Python is written in Python 3 and Fortran in a Linux environment. There are two ways of installing the package.

Python package
~~~~~~~~~~~~~~

Assuming you have a Linux machine the easiest way to install AquaCrop-Python is using the Python package manager. As for most Python applications we recommend using a virtual environment tool such as Anaconda.

.. code-block:: console

   $ conda create --name aquacrop python=3.8
   $ conda activate aquacrop
   $ python -m pip install -e git+https://github.com/simonmoulds/aquacrop.git#egg=aquacrop

Docker image
~~~~~~~~~~~~

Alternatively the software is available as a Docker image (EXPERIMENTAL). 
