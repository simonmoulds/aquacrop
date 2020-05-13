The Configuration File
======================

.. currentmodule:: aquacrop

AquaCrop-Python reads the various settings from a user-supplied configuration file based on the structure found in Microsoft Windows INI files. The basic structure of a configuration file is as follows:

.. code-block:: python
		
    [SECTION_ONE]
    option = value

    [SECTION_TWO]
    option = value


..
   The sections and required options are listed below.
    
..
   ``[FILE_PATHS]``
   ~~~~~~~~~~~~~~~~

   ``PathIn``
       Path to input files.

   ``PathOut``
       Path to location of model output.

   ``[MODEL_GRID]``
   ~~~~~~~~~~~~~~~~

   ``mask`` : ``str``
       The name of the region mask netCDF. This must be provided if running a two-dimensional simulation.

   ``[PSEUDO_COORDS]``
   ~~~~~~~~~~~~~~~~~~~

   ``crop`` : ``int``
       The number of crops in the simulation.

   ``farm`` : ``int``
       The number of farms in the simulation.

   ``[CLOCK]``
   ~~~~~~~~~~~

   ``startTime`` : ``str``
       The start time of the simulation, in 'YYYY-MM-DD'

   ``endTime``   : ``str``
       The end time of the simulation, in 'YYYY-MM-DD'

   ``[INITIAL_WATER_CONTENT]``
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ``initialConditionType`` : ``str``
       Specify the way in which the initial condition should be set:

       :``FILE``: Read initial condition from a file
       :``FILE``: TODO
       :``FILE``: TODO

   ``filename`` : ``str``
       If ``initialConditionType`` is ``FILE`` this option specifies the name of the file from which the values can be read

   ``initialConditionInterpMethod`` : ``str``
       TODO

   ``initialConditionDepthVarName`` : ``str``
       TODO

   ``[NETCDF_ATTRIBUTES]``
   ~~~~~~~~~~~~~~~~~~~~~~~

   This section can include any user-specified attributes to be added to the output netCDF files.

   ``[PRECIPITATION]``
   ~~~~~~~~~~~~~~~~~~~
   ``filename`` : ``str``
       Path to netCDF file containing precipitation.

   ``varname`` : ``str``
       NetCDF variable name.

   ``[TAVG]``
   ~~~~~~~~~~
   ``filename`` : ``str``
       Path to netCDF file containing precipitation.

   ``varname`` : ``str``
       NetCDF variable name.

   ``[TMIN]``
   ~~~~~~~~~~

   ``filename`` : ``str``
       Path to netCDF file containing precipitation.

   ``varname`` : ``str``
       NetCDF variable name.

   ``[ETREF]``
   ~~~~~~~~~~~

   ``filename`` : ``str``
       Path to netCDF file containing precipitation.

   ``varname`` : ``str``
       NetCDF variable name.

   ``[CARBON_DIOXIDE]``
   ~~~~~~~~~~~~~~~~~~~~
   ``filename`` : ``str``
       Path to netCDF file containing precipitation.

   ``varname`` : ``str``
       NetCDF variable name.

   ``[WATER_TABLE]``
   ~~~~~~~~~~~~~~~~~~~
   ``WaterTable`` : ``str``
       TODO

   ``VariableWaterTable`` : ``str``
       TODO

   ``groundwaterVarName`` : ``str``
       TODO

   ``groundwaterInputDir`` : ``str``
       TODO

   ``DailyGroundwaterNC`` : ``str``
       TODO

   ``groundwaterInputFile`` : ``str``
       TODO

   ``[CROP_PARAMETERS]``
   ~~~~~~~~~~~~~~~~~~~~~
   ``filename``
       TODO

   ``CalendarType``
       TODO

   ``SwitchGDD``
       TODO

   ``GDDmethod``
       TODO

   ``daily_total``
       TODO

   ``year_max``
       TODO

   ``[IRRIGATION_PARAMETERS]``
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ``filename``
       TODO

   ``varname``
       TODO   

   ``[IRRIGATION_SCHEDULE]``
   ~~~~~~~~~~~~~~~~~~~~~~~~~

   ``filename``
       TODO

   ``varname``
       TODO

   ``[FIELD_PARAMETERS]``
   ~~~~~~~~~~~~~~~~~~~~~~

   ``filename``
       TODO

   ``varname``
       TODO

   ``[SOIL_HYDRAULIC_PARAMETERS]``
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ``filename``
       Path to netCDF file containing soil hydraulic parameters


   ``[SOIL_PROFILE]``
   ~~~~~~~~~~~~~~~~~~

   ``dzLayer``
       TODO

   ``dzComp``
       TODO

   ``[SOIL_PARAMETERS]``
   ~~~~~~~~~~~~~~~~~~~~~

   ``filename``
       TODO

   ``adjustReadilyAvailableWater``
       TODO

   ``adjustCurveNumber``
       TODO

   ``[REPORTING]``
   ~~~~~~~~~~~~~~~
