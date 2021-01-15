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

Alternatively the software is available as a Docker image, which means it can be used without needing to install any prerequisites. Assuming you have Docker installed on your system, simply run:

.. code-block:: console

   $ docker pull simonmoulds/aquacrop:latest

Inside the Docker container `aquacrop` is run from the directory `/app`. This means that to run the model on your local machine you must map the local data directory to this container folder. Assuming your config file `my-config.toml` is located in your current working directory, you could run:

.. code-block:: console

   $ docker run -v $PWD:/app aquacrop /app/my-config.toml

If you're using the Docker container to run AquaCrop-Python then you must take care when specifying the location of input files in the configuration files.
