Configure
=========

.. currentmodule:: aquacrop

AquaCrop-Python reads the various settings from a user-supplied TOML configuration file. The basic structure of a configuration file is as follows:

.. code-block:: python

    [MODEL_GRID]
    mask = "aquacrop_landmask_pt.nc"
    is_1d = true
    mask_varname = "landmask"
    xy_dimname = "space"

    [PSEUDO_COORDS]
    crop = [ 2,]
    farm = 1

    [CLOCK]
    start_time = 2019-01-01T00:00:00
    end_time = 2019-12-31T00:00:00

    [INITIAL_WATER_CONTENT]
    type = "percent"
    property = ""
    percent = 50.0
    filename = ""
    is_1d = false
    xy_dimname = ""
    interp_method = ""

    [PRECIPITATION]
    filename = "TAHMO_2019_meteo.nc"
    varname = "prate"
    is_1d = true
    xy_dimname = "space"

    [TAVG]
    filename = "TAHMO_2019_meteo.nc"
    varname = "tavg"
    is_1d = true
    xy_dimname = "space"

    [TMIN]
    filename = "TAHMO_2019_meteo.nc"
    varname = "tmin"
    is_1d = true
    xy_dimname = "space"

    [TMAX]
    filename = "TAHMO_2019_meteo.nc"
    varname = "tmax"
    is_1d = true
    xy_dimname = "space"

    [LWDOWN]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [SP]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [SH]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [RHMAX]
    filename = "AgMERRA_1981_2010_rhstmax.nc"
    varname = "rhstmax"
    is_1d = true
    xy_dimname = "space"

    [RHMIN]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [RHMEAN]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [SWDOWN]
    filename = "AgMERRA_1981_2010_srad.nc"
    varname = "srad"
    is_1d = true
    xy_dimname = "space"

    [WIND]
    filename = "AgMERRA_1981_2010_wndspd.nc"
    varname = "wndspd"
    is_1d = true
    xy_dimname = "space"

    [ETREF]
    preprocess = false
    method = "Hargreaves"
    daily_total = "ETref"
    filename = "TAHMO_2019_meteo.nc"
    varname = "etref"
    is_1d = true
    xy_dimname = "space"

    [CARBON_DIOXIDE]
    filename = "aquacrop_co2_conc_pt.nc"
    varname = "co2"
    is_1d = true
    xy_dimname = "space"

    [WATER_TABLE]
    water_table = false
    dynamic = false
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""
    directory = ""
    coupled = false
    time_lag = 0
    max_wait_time = 0
    wait_interval = 0

    [CROP_PARAMETERS]
    filename = "aquacrop_crop_parms_pt.nc"
    varname = ""
    is_1d = true
    xy_dimname = "space"
    crop_id = [ 2,]
    calendar_type = 2
    switch_gdd = true
    gdd_method = 3
    planting_day = [ "121",]
    harvest_day = [ "304",]

    [IRRIGATION_MANAGEMENT]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [FIELD_MANAGEMENT]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""

    [SOIL_PROFILE]
    dzLayer = [ 0.0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0,]
    dzComp = [ 0.1, 0.35, 1.0, 3.0,]

    [SOIL_HYDRAULIC_PARAMETERS]
    filename = "aquacrop_soil_parms_pt.nc"
    k_sat_varname = "k_sat"
    th_sat_varname = "th_sat"
    th_fc_varname = "th_fc"
    th_wilt_varname = "th_wilt"
    is_1d = true
    xy_dimname = "space"

    [SOIL_PARAMETERS]
    filename = ""
    varname = ""
    is_1d = false
    xy_dimname = ""
    adjust_raw = true
    adjust_cn = true

    [REPORTING]
    report = true
    daily_total = [ "CC", "B",]
    year_max = [ "Y",]
    
    [NETCDF_ATTRIBUTES]
    institution = ""
    title = ""
    description = ""

.. toctree::
   :maxdepth: 2
   :hidden:
      
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

