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

netcdf_shortname['crop'] = 'crop'
netcdf_standard_name['crop'] = 'crop'
netcdf_dimensions['crop'] = ('crop',)
netcdf_units['crop']       = '1'
netcdf_long_name['crop']  = 'crop'
netcdf_description['crop']       = None
comment['crop']           = None
latex_symbol['crop']      = None

netcdf_shortname['farm'] = 'farm'
netcdf_standard_name['farm'] = 'farm'
netcdf_dimensions['farm'] = ('farm',)
netcdf_units['farm']       = '1'
netcdf_long_name['farm']  = 'farm'
netcdf_description['farm']       = None
comment['farm']           = None
latex_symbol['farm']      = None

aquacrop_dimension_name = 'depth'
netcdf_shortname[aquacrop_dimension_name] = 'depth'
netcdf_standard_name[aquacrop_dimension_name] = 'depth'
netcdf_dimensions[aquacrop_dimension_name] = ('depth',)
netcdf_units[aquacrop_dimension_name]       = '1'
netcdf_long_name[aquacrop_dimension_name]  = 'depth'
netcdf_description[aquacrop_dimension_name]       = None
comment[aquacrop_dimension_name]           = None
latex_symbol[aquacrop_dimension_name]      = None

aquacrop_variable_name = 'precipitation'
netcdf_shortname[aquacrop_variable_name] = 'precipitation'
netcdf_standard_name[aquacrop_variable_name] = 'precipitation'
netcdf_dimensions[aquacrop_variable_name] = ('time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'm'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'th'
netcdf_shortname[aquacrop_variable_name] = 'water_content'
netcdf_standard_name[aquacrop_variable_name] = 'water_content'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','depth','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'm3 m-3'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Wr'
netcdf_shortname[aquacrop_variable_name] = 'root_zone_water_content'
netcdf_standard_name[aquacrop_variable_name] = 'root_zone_water_content'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'zGW'
netcdf_shortname[aquacrop_variable_name] = 'groundwater_depth'
netcdf_standard_name[aquacrop_variable_name] = 'groundwater_depth'
netcdf_dimensions[aquacrop_variable_name] = ('time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'm'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'SurfaceStorage'
netcdf_shortname[aquacrop_variable_name] = 'surface_layer_water_content'
netcdf_standard_name[aquacrop_variable_name] = 'surface_layer_water_content'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Irr'
netcdf_shortname[aquacrop_variable_name] = 'irrigation_depth'
netcdf_standard_name[aquacrop_variable_name] = 'irrigation_depth'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'IrrCum'
netcdf_shortname[aquacrop_variable_name] = 'cumulative_irrigation'
netcdf_standard_name[aquacrop_variable_name] = 'cumulative_irrigation'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'IrrNetCum'
netcdf_shortname[aquacrop_variable_name] = 'cumulative_net_irrigation'
netcdf_standard_name[aquacrop_variable_name] = 'cumulative_net_irrigation'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Infl'
netcdf_shortname[aquacrop_variable_name] = 'infiltration'
netcdf_standard_name[aquacrop_variable_name] = 'infiltration'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Runoff'
netcdf_shortname[aquacrop_variable_name] = 'surface_runoff'
netcdf_standard_name[aquacrop_variable_name] = 'surface_runoff'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'DeepPerc'
netcdf_shortname[aquacrop_variable_name] = 'deep_percolation'
netcdf_standard_name[aquacrop_variable_name] = 'deep_percolation'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Recharge'
netcdf_shortname[aquacrop_variable_name] = 'recharge'
netcdf_standard_name[aquacrop_variable_name] = 'recharge'
netcdf_dimensions[aquacrop_variable_name] = ('time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'm3'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'CrTot'
netcdf_shortname[aquacrop_variable_name] = 'capillary_rise'
netcdf_standard_name[aquacrop_variable_name] = 'capillary_rise'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'GwIn'
netcdf_shortname[aquacrop_variable_name] = 'groundwater_inflow'
netcdf_standard_name[aquacrop_variable_name] = 'groundwater_inflow'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'EsAct'
netcdf_shortname[aquacrop_variable_name] = 'evaporation'
netcdf_standard_name[aquacrop_variable_name] = 'evaporation'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Epot'
netcdf_shortname[aquacrop_variable_name] = 'potential_evaporation'
netcdf_standard_name[aquacrop_variable_name] = 'potential_evaporation'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'TrAct'
netcdf_shortname[aquacrop_variable_name] = 'transpiration'
netcdf_standard_name[aquacrop_variable_name] = 'transpiration'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Tpot'
netcdf_shortname[aquacrop_variable_name] = 'potential_transpiration'
netcdf_standard_name[aquacrop_variable_name] = 'potential_transpiration'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 m'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

# crop growth
# ###########

aquacrop_variable_name = 'GDD'
netcdf_shortname[aquacrop_variable_name] = 'growing_degree_days'
netcdf_standard_name[aquacrop_variable_name] = 'growing_degree_days'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'days'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'GDDcum'
netcdf_shortname[aquacrop_variable_name] = 'cumulative_growing_degree_days'
netcdf_standard_name[aquacrop_variable_name] = 'cumulative_growing_degree_days'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'days'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Zroot'
netcdf_shortname[aquacrop_variable_name] = 'root_depth'
netcdf_standard_name[aquacrop_variable_name] = 'root_depth'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'm'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'CC'
netcdf_shortname[aquacrop_variable_name] = 'canopy_cover'
netcdf_standard_name[aquacrop_variable_name] = 'canopy_cover'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'CC_NS'
netcdf_shortname[aquacrop_variable_name] = 'reference_canopy_cover'
netcdf_standard_name[aquacrop_variable_name] = 'reference_canopy_cover'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'B'
netcdf_shortname[aquacrop_variable_name] = 'biomass'
netcdf_standard_name[aquacrop_variable_name] = 'biomass'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 kg m-2'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'B_NS'
netcdf_shortname[aquacrop_variable_name] = 'reference_biomass'
netcdf_standard_name[aquacrop_variable_name] = 'reference_biomass'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 kg m-2'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'HI'
netcdf_shortname[aquacrop_variable_name] = 'harvest_index'
netcdf_standard_name[aquacrop_variable_name] = 'harvest_index'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'HIadj'
netcdf_shortname[aquacrop_variable_name] = 'adjusted_harvest_index'
netcdf_standard_name[aquacrop_variable_name] = 'adjusted_harvest_index'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Y'
netcdf_shortname[aquacrop_variable_name] = 'crop_yield'
netcdf_standard_name[aquacrop_variable_name] = 'crop_yield'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 kg 1e4 m-2'  # tonne/hectare
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'Production'
netcdf_shortname[aquacrop_variable_name] = 'crop_production'
netcdf_standard_name[aquacrop_variable_name] = 'crop_production'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = '1e-3 kg'  # tonne
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None

aquacrop_variable_name = 'thRZ_Act'
netcdf_shortname[aquacrop_variable_name] = 'th_rz'
netcdf_standard_name[aquacrop_variable_name] = 'root_zone_water_content'
netcdf_dimensions[aquacrop_variable_name] = ('farm','crop','time','lat','lon')
netcdf_units[aquacrop_variable_name]       = 'm3 m-3'
netcdf_long_name[aquacrop_variable_name]  = None
netcdf_description[aquacrop_variable_name]       = None
comment[aquacrop_variable_name]           = None
latex_symbol[aquacrop_variable_name]      = None
