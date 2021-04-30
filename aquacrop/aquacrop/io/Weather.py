#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hm.api import open_hmdataarray
from hm.input import HmInputData


class MaxTemperature(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.TMAX['filename']
        self.nc_varname = model.config.TMAX['varname']
        self.is_1d = model.config.TMAX['is_1d']
        self.xy_dimname = model.config.TMAX['xy_dimname']
        self.model_varname = 'tmax'


class MinTemperature(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.TMIN['filename']
        self.nc_varname = model.config.TMIN['varname']
        self.is_1d = model.config.TMIN['is_1d']
        self.xy_dimname = model.config.TMIN['xy_dimname']
        self.model_varname = 'tmin'


class MeanTemperature(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.TAVG['filename']
        self.nc_varname = \
            model.config.TAVG['varname']
        self.is_1d = model.config.TAVG['is_1d']
        self.xy_dimname = model.config.TAVG['xy_dimname']
        self.model_varname = 'tmean'


class Precipitation(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.PRECIPITATION['filename']
        self.nc_varname = model.config.PRECIPITATION['varname']
        self.is_1d = model.config.PRECIPITATION['is_1d']
        self.xy_dimname = model.config.PRECIPITATION['xy_dimname']
        self.model_varname = 'prec'


class ETref(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = model.config.ETREF['filename']
        self.nc_varname = model.config.ETREF['varname']
        self.is_1d = model.config.ETREF['is_1d']
        self.xy_dimname = model.config.ETREF['xy_dimname']
        self.model_varname = 'etref'


class Weather(object):
    def __init__(self, model):
        self.model = model
        self.max_temperature_module = MaxTemperature(model)
        self.min_temperature_module = MinTemperature(model)
        self.prec_module = Precipitation(model)
        self.etref_module = ETref(model)

    def initial(self):
        self.max_temperature_module.initial()
        self.min_temperature_module.initial()
        self.prec_module.initial()
        self.etref_module.initial()

        self.model.Tmax = self.model.tmax.values
        self.model.Tmin = self.model.tmin.values
        self.model.P = self.model.prec.values
        self.model.ETref = self.model.etref.values

    def dynamic(self):
        self.max_temperature_module.dynamic()
        self.min_temperature_module.dynamic()
        self.prec_module.dynamic()
        self.etref_module.dynamic()
