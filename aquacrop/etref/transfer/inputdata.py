#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hm.api import open_hmdataarray
from hm.input import HmInputData

# class HmInputData(object):
#     def __init__(
#             self,
#             model,
#             filename,
#             nc_varname,
#             model_varname
#     ):
#         self.model = model
#         self.filename = filename
#         self.nc_varname = nc_varname
#         self.model_varname = model_varname        

#     def initial(self):
#         self.read()
        
#     def read(self):
#         vars(self.model)[self.model_varname] = open_hmdataarray(
#             self.filename,
#             self.nc_varname,
#             self.model.domain
#         )
        
#     def update(self):
#         vars(self.model)[self.model_varname].select(
#             time=self.model.time.curr_time, method='nearest'
#         )
        
#     def dynamic(self):
#         self.update()    

class Wind(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.WIND['filename']
        self.nc_varname = model.config.WIND['varname']
        self.is_1d = model.config.WIND['is_1d']
        self.xy_dimname = model.config.WIND['xy_dimname']
        self.model_varname = 'wind'
        
    def update(self, method='nearest'):
        super()
        z = 10. # height of wind speed variable (10 meters above surface)
        # TODO: pass a function to open_hmdataarray?
        self.model.wind_2m = self.model.wind.values * 4.87 / (np.log(67.8 * z - 5.42))
        
class LongwaveRadiation(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.LWDOWN['filename']
        self.nc_varname = model.config.LWDOWN['varname']
        self.is_1d = model.config.LWDOWN['is_1d']
        self.xy_dimname = model.config.LWDOWN['xy_dimname']
        self.model_varname = 'longwave_radiation'
        
class ShortwaveRadiation(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.SWDOWN['filename']
        self.nc_varname = model.config.SWDOWN['varname']
        self.is_1d = model.config.SWDOWN['is_1d']
        self.xy_dimname = model.config.SWDOWN['xy_dimname']
        self.model_varname = 'shortwave_radiation'
    
class MaxRelativeHumidity(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.RHMAX['filename']
        self.nc_varname = model.config.RHMAX['varname']
        self.is_1d = model.config.RHMAX['is_1d']
        self.xy_dimname = model.config.RHMAX['xy_dimname']
        self.model_varname = 'max_relative_humidity'
    
class MinRelativeHumidity(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.RHMIN['filename']
        self.nc_varname = model.config.RHMIN['varname']
        self.is_1d = model.config.RHMIN['is_1d']
        self.xy_dimname = model.config.RHMIN['xy_dimname']
        self.model_varname = 'min_relative_humidity'
        
class MeanRelativeHumidity(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.RHMEAN['filename']
        self.nc_varname = model.config.RHMEAN['varname']
        self.is_1d = model.config.RHMEAN['is_1d']
        self.xy_dimname = model.config.RHMEAN['xy_dimname']
        self.model_varname = 'mean_relative_humidity'

class RelativeHumidity(object):
    def __init__(self, model):
        self.max_relative_humidity_module = MaxRelativeHumidity(model)
        self.min_relative_humidity_module = MinRelativeHumidity(model)
        self.mean_relative_humidity_module = MeanRelativeHumidity(model)

    def initial(self):
        self.max_relative_humidity_module.initial()
        self.min_relative_humidity_module.initial()
        self.mean_relative_humidity_module.initial()

    def dynamic(self):
        self.max_relative_humidity_module.dynamic()
        self.min_relative_humidity_module.dynamic()
        self.mean_relative_humidity_module.dynamic()
        
class MaxTemperature(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.TMAX['filename']
        self.nc_varname = model.config.TMAX['varname']
        self.is_1d = model.config.TMAX['is_1d']
        self.xy_dimname = model.config.TMAX['xy_dimname']
        self.model_varname = 'tmax'
        # self.offset = model.config.TMAX['offset']
        # self.factor = model.config.TMAX['factor']
        
    def update(self, method='nearest'):
        super()
        # self.model.tmax.values += self.offset
        # self.model.tmax.values *= self.factor

class MinTemperature(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.TMIN['filename']
        self.nc_varname = model.config.TMIN['varname']
        self.is_1d = model.config.TMIN['is_1d']
        self.xy_dimname = model.config.TMIN['xy_dimname']
        self.model_varname = 'tmin'
        # self.offset = model.config.TMIN['offset']
        # self.factor = model.config.TMIN['factor']
        
    def update(self, method='nearest'):
        super()
        # self.model.tmin.values += self.offset
        # self.model.tmin.values *= self.factor
        
class MeanTemperature(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.TAVG['filename']
        self.nc_varname = model.config.TAVG['varname']
        self.is_1d = model.config.TAVG['is_1d']
        self.xy_dimname = model.config.TAVG['xy_dimname']
        self.model_varname = 'tmean'
        # self.offset = model.config.TAVG['offset']
        # self.factor = model.config.TAVG['factor']
        
    def update(self, method='nearest'):
        super()
        # self.model.tmean.values += self.offset
        # self.model.tmean.values *= self.factor

class Temperature(object):
    def __init__(self, model):
        self.max_temperature_module = MaxTemperature(model)
        self.min_temperature_module = MinTemperature(model)
        self.mean_temperature_module = MeanTemperature(model)

    def initial(self):
        self.max_temperature_module.initial()
        self.min_temperature_module.initial()
        self.mean_temperature_module.initial()

    def dynamic(self):
        self.max_temperature_module.dynamic()
        self.min_temperature_module.dynamic()
        self.mean_temperature_module.dynamic()
        
class SpecificHumidity(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.SH['filename']
        self.nc_varname = model.config.SH['varname']
        self.is_1d = model.config.SH['is_1d']
        self.xy_dimname = model.config.SH['xy_dimname']
        self.model_varname = 'specific_humidity'

class SurfacePressure(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.SP['filename']
        self.nc_varname = model.config.SP['varname']
        self.is_1d = model.config.SP['is_1d']
        self.xy_dimname = model.config.SP['xy_dimname']
        self.model_varname = 'surface_pressure'
