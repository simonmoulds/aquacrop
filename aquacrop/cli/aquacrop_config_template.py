#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import os
import sys
import toml

@click.command()
# @click.option('--debug/--no-debug', 'debug', default=False)
# @click.option('-o', '--outputdir', 'outputdir', default='.', type=click.Path())
# @click.option('--monte-carlo', 'montecarlo', default=False)
# @click.option('--enkf', 'kalmanfilter', default=False)
@click.argument('config', default='config-template.toml', type=click.Path())
def cli(config):
    config_dict = {
        'MODEL_GRID' : {
            'mask'          : 'landmask.nc',
            'is_1d'         : True,
            'mask_varname'  : 'mask',
            'xy_dimname'    : 'space',
        },
        'PSEUDO_COORDS' : {
            'crop'          : [1,],
            'farm'          : [1,]
        },
        'CLOCK' : {
            'start_time'    : '2019-01-01T00:00:00',
            'end_time'      : '2019-12-31T00:00:00'
        },
        '[INITIAL_WATER_CONTENT]' : {
            'type'          : 'percent',
            'property'      : '',
            'percent'       : 50.0,
            'filename'      : '',
            'is_1d'         : False,
            'xy_dimname'    : '',
            'interp_method' : '',
        },
        '[PRECIPITATION]' : {
            'filename'      : 'prate.nc',
            'varname'       : 'prate',
            'is_1d'         : True,
            'xy_dimname'    : 'space',
        },
        '[TAVG]' : {
            'filename'      : 'tavg.nc',
            'varname'       : 'tavg',
            'is_1d'         : True,
            'xy_dimname'    : 'space',
        },
        '[TMIN]' : {
            'filename'      : 'tmin.nc',
            'varname'       : 'tmin',
            'is_1d'         : True,
            'xy_dimname'    : 'space',
        },
        '[TMAX]' : {
            'filename'      : 'tmax.nc',
            'varname'       : 'tmax',
            'is_1d'         : True,
            'xy_dimname'    : 'space',
        },
        '[LWDOWN]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[SP]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[SH]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[RHMAX]' : {
            'filename'      : 'rhstmax.nc',
            'varname'       : 'rhstmax',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[RHMIN]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[RHMEAN]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[SWDOWN]' : {
            'filename'      : 'srad.nc',
            'varname'       : 'srad',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[WIND]' : {
            'filename'      : 'wndspd.nc',
            'varname'       : 'wndspd',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[ETREF]' : {
            'preprocess'    : False,
            'method'        : 'Hargreaves',
            'daily_total'   : 'ETref',
            'filename'      : 'etref.nc',
            'varname'       : 'etref',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[CARBON_DIOXIDE]' : {
            'filename'      : 'conc.nc',
            'varname'       : 'conc',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[WATER_TABLE]' : {
            'water_table'   : False,
            'dynamic'       : False,
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
            'directory'     : '',
            'coupled'       : False,
            'time_lag'      : 0,
            'max_wait_time' : 0,
            'wait_interval' : 0
        },
        '[CROP_PARAMETERS]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : 'space',
            'crop_id'       : [ 2,],
            'calendar_type' : 2,
            'switch_gdd'    : True,
            'gdd_method'    : 3,
            'planting_day'  : [ 121,],
            'harvest_day'   : [ 304,]
        },
        '[IRRIGATION_MANAGEMENT]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },        
        '[FIELD_MANAGEMENT]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
        },
        '[SOIL_PROFILE]' : {
            'dzLayer'       : [ 0.0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0,],
            'dzComp'        : [ 0.1, 0.35, 1.0, 3.0,]
        },            
        '[SOIL_HYDRAULIC_PARAMETERS]' : {
            'filename'      : 'soilparms.nc',
            'k_sat_varname' : 'k_sat',
            'th_sat_varname': 'th_sat',
            'th_fc_varname' : 'th_fc',
            'th_wilt_varname': 'th_wilt',
            'is_1d'         : True,
            'xy_dimname'    : 'space'
        },
        '[SOIL_PARAMETERS]' : {
            'filename'      : '',
            'varname'       : '',
            'is_1d'         : True,
            'xy_dimname'    : '',
            'adjust_raw'    : True,
            'adjust_cn'     : True
        },
        '[REPORTING]' : {
            'report'        : True,
            'daily_total'   : [ 'CC', 'B',],
            'year_max'      : [ 'Y',]
        },
        '[NETCDF_ATTRIBUTES]' : {
            'institution'   : '',
            'title'         : '',
            'description'   : ''
        }
    }    
    with open(config, 'w') as configfile:
        toml.dump(config_dict, configfile)
    
