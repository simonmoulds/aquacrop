#!/usr/bin/env python
# -*- coding: utf-8 -*-

netcdf_shortname = {}
netcdf_standard_name = {}
netcdf_long_name = {}
netcdf_dimensions = {}
netcdf_units = {}
# netcdf_monthly_total_unit = {} 
# netcdf_yearly_total_unit  = {}
netcdf_calendar = {}
netcdf_description = {}
comment = {}
latex_symbol = {}

netcdf_shortname['time'] = 'time'
netcdf_standard_name['time'] = 'time'
netcdf_dimensions['time'] = ('time',)
netcdf_units['time']       = 'Days since 1901-01-01'
netcdf_long_name['time']  = 'Days since 1901-01-01'
netcdf_calendar['time']   = 'standard'
netcdf_description['time']       = None
comment['time']           = None
latex_symbol['time']      = None

netcdf_shortname['lat'] = 'lat'
netcdf_standard_name['lat'] = 'latitude'
netcdf_dimensions['lat'] = ('lat',)
netcdf_units['lat']       = 'degrees_north'
netcdf_long_name['lat']  = 'latitude'
netcdf_description['lat']       = None
comment['lat']           = None
latex_symbol['lat']      = None

netcdf_shortname['lon'] = 'lon'
netcdf_standard_name['lon'] = 'longitude'
netcdf_dimensions['lon'] = ('lon',)
netcdf_units['lon']       = 'degrees_east'
netcdf_long_name['lon']  = 'longitude'
netcdf_description['lon']       = None
comment['lon']           = None
latex_symbol['lon']      = None

netcdf_shortname['space'] = 'space'
netcdf_standard_name['space'] = 'space'
netcdf_dimensions['space'] = ('space',)
netcdf_units['space'] = '1'
netcdf_long_name['space'] = 'space'
netcdf_description['space'] = None
comment['space'] = None
latex_symbol['space'] = None

aquacrop_variable_name = 'ETref'
netcdf_shortname[aquacrop_variable_name] = 'etref'
netcdf_standard_name[aquacrop_variable_name] = 'etref'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name]       = 'kg m-2 day-1'
netcdf_long_name[aquacrop_variable_name]  = 'reference crop evapotranspiration'
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'tmin'
netcdf_shortname[aquacrop_variable_name] = 'tmin'
netcdf_standard_name[aquacrop_variable_name] = 'air_temperature'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = 'minimum daily air temperature'
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'tmax'
netcdf_shortname[aquacrop_variable_name] = 'tmax'
netcdf_standard_name[aquacrop_variable_name] = 'air_temperature'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = 'maximum daily air temperature'
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'tmean'
netcdf_shortname[aquacrop_variable_name] = 'tmean'
netcdf_standard_name[aquacrop_variable_name] = 'air_temperature'
netcdf_dimensions[aquacrop_variable_name] = ('time',)
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = 'mean daily air temperature'
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

# aquacrop_variable_name = 'TrAct'
# netcdf_shortname[aquacrop_variable_name] = 'transpiration'
# netcdf_standard_name[aquacrop_variable_name] = 'transpiration'
# netcdf_dimensions[aquacrop_variable_name] = ('crop','time','lat','lon')
# netcdf_units[aquacrop_variable_name]       = '1e-3 m'
# netcdf_long_name[aquacrop_variable_name]  = None
# netcdf_description[aquacrop_variable_name]       = None
# comment[aquacrop_variable_name]           = None
# latex_symbol[aquacrop_variable_name]      = None

# aquacrop_variable_name = 'Tpot'
# netcdf_shortname[aquacrop_variable_name] = 'potential_transpiration'
# netcdf_standard_name[aquacrop_variable_name] = 'potential_transpiration'
# netcdf_dimensions[aquacrop_variable_name] = ('crop','time','lat','lon')
# netcdf_units[aquacrop_variable_name]       = '1e-3 m'
# netcdf_long_name[aquacrop_variable_name]  = None
# netcdf_description[aquacrop_variable_name]       = None
# comment[aquacrop_variable_name]           = None
# latex_symbol[aquacrop_variable_name]      = None
