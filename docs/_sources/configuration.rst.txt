Configure
=========

.. currentmodule:: aquacrop

AquaCrop-Python reads the configuration settings from a user-supplied TOML configuration file. A template configuration file can be generated using the following command:

.. code-block:: console
		
   $ aquacrop-config-template my-template-config.toml

The configuration file is divided into multiple sections. Detailed guidance about the requirements of each section can be found by following the links below.

.. toctree::
   :maxdepth: 1
      
   model_grid
   clock
   pseudo_coords
   meteorological_inputs
   groundwater
   crop_parameters
   irrigation_parameters
   irrigation_schedule
   field_parameters
   soil_profile
   soil_hydraulic_parameters
   soil_parameters
   reporting   

