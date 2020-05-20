#!/usr/bin/env python
# -*- coding: utf-8 -*-

netcdf_shortname = {}
netcdf_standard_name = {}
netcdf_long_name = {}
netcdf_dimensions = {}
netcdf_units = {}
netcdf_calendar = {}
netcdf_description = {}
comment = {}
latex_symbol = {}

netcdf_shortname['time'] = 'time'
netcdf_standard_name['time'] = 'time'
netcdf_dimensions['time'] = ('time',)
netcdf_units['time'] = 'Days since 1901-01-01'
netcdf_long_name['time'] = 'Days since 1901-01-01'
netcdf_calendar['time'] = 'standard'
netcdf_description['time'] = None
comment['time'] = None
latex_symbol['time'] = None

netcdf_shortname['lat'] = 'lat'
netcdf_standard_name['lat'] = 'latitude'
netcdf_dimensions['lat'] = ('lat',)
netcdf_units['lat'] = 'degrees_north'
netcdf_long_name['lat'] = 'latitude'
netcdf_description['lat'] = None
comment['lat'] = None
latex_symbol['lat'] = None

netcdf_shortname['space'] = 'space'
netcdf_standard_name['space'] = 'space'
netcdf_dimensions['space'] = ('space',)
netcdf_units['space'] = '1'
netcdf_long_name['space'] = 'space'
netcdf_description['space'] = None
comment['space'] = None
latex_symbol['space'] = None

netcdf_shortname['lon'] = 'lon'
netcdf_standard_name['lon'] = 'longitude'
netcdf_dimensions['lon'] = ('lon',)
netcdf_units['lon'] = 'degrees_east'
netcdf_long_name['lon'] = 'longitude'
netcdf_description['lon'] = None
comment['lon'] = None
latex_symbol['lon'] = None

netcdf_shortname['crop'] = 'crop'
netcdf_standard_name['crop'] = 'crop'
netcdf_dimensions['crop'] = ('crop',)
netcdf_units['crop'] = '1'
netcdf_long_name['crop'] = 'crop'
netcdf_description['crop'] = None
comment['crop'] = None
latex_symbol['crop'] = None

netcdf_shortname['farm'] = 'farm'
netcdf_standard_name['farm'] = 'farm'
netcdf_dimensions['farm'] = ('farm',)
netcdf_units['farm'] = '1'
netcdf_long_name['farm'] = 'farm'
netcdf_description['farm'] = None
comment['farm'] = None
latex_symbol['farm'] = None

aquacrop_dimension_name = 'depth'
netcdf_shortname[aquacrop_dimension_name] = 'depth'
netcdf_standard_name[aquacrop_dimension_name] = 'depth'
netcdf_dimensions[aquacrop_dimension_name] = ('depth',)
netcdf_units[aquacrop_dimension_name] = '1'
netcdf_long_name[aquacrop_dimension_name] = 'depth'
netcdf_description[aquacrop_dimension_name] = None
comment[aquacrop_dimension_name] = None
latex_symbol[aquacrop_dimension_name] = None

aquacrop_variable_name = 'precipitation'
netcdf_shortname[aquacrop_variable_name] = 'precipitation'
netcdf_standard_name[aquacrop_variable_name] = 'precipitation'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name] = 'm'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'th'
netcdf_shortname[aquacrop_variable_name] = 'water_content'
netcdf_standard_name[aquacrop_variable_name] = 'water_content'
netcdf_dimensions[aquacrop_variable_name] = ('time', 'depth')
netcdf_units[aquacrop_variable_name] = 'm3 m-3'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'zGW'
netcdf_shortname[aquacrop_variable_name] = 'groundwater_depth'
netcdf_standard_name[aquacrop_variable_name] = 'groundwater_depth'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name] = 'm'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'direct_runoff'
netcdf_shortname[aquacrop_variable_name] = 'surface_runoff'
netcdf_standard_name[aquacrop_variable_name] = 'surface_runoff'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name] = 'm3'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'ETact'
netcdf_shortname[aquacrop_variable_name] = 'actual_evapotranspiration'
netcdf_standard_name[aquacrop_variable_name] = 'actual_evapotranspiration'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name] = 'm3'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'deep_percolation'
netcdf_shortname[aquacrop_variable_name] = 'recharge'
netcdf_standard_name[aquacrop_variable_name] = 'recharge'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name] = 'm3'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'CC'
netcdf_shortname[aquacrop_variable_name] = 'canopy_cover'
netcdf_standard_name[aquacrop_variable_name] = 'canopy_cover'
netcdf_dimensions[aquacrop_variable_name] = ('farm', 'crop', 'time')
netcdf_units[aquacrop_variable_name] = '1'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None

aquacrop_variable_name = 'thRZ_Act'
netcdf_shortname[aquacrop_variable_name] = 'th_rz'
netcdf_standard_name[aquacrop_variable_name] = 'root_zone_water_content'
netcdf_dimensions[aquacrop_variable_name] = ('farm', 'crop', 'time')
netcdf_units[aquacrop_variable_name] = 'm3 m-3'
netcdf_long_name[aquacrop_variable_name] = None
netcdf_description[aquacrop_variable_name] = None
comment[aquacrop_variable_name] = None
latex_symbol[aquacrop_variable_name] = None
