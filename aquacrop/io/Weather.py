#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import string
import numpy as np
import xarray as xr

from hm import file_handling
from hm.Messages import ModelError, ModelFileError, ModelWarning

import logging
logger = logging.getLogger(__name__)

class Weather(object):

    def __init__(self, Weather_variable):
        self._configuration = Weather_variable._configuration
        self._modelTime = Weather_variable._modelTime
        self.cloneMapFileName = Weather_variable.cloneMapFileName
        # self.cloneMap = Weather_variable.cloneMap
        self.landmask = Weather_variable.landmask

    def initial(self):
        self.set_input_filenames()
        self.set_nc_variable_names()
        self.set_weather_conversion_factors()

    def set_input_filenames(self):
        self.preFileNC = self._configuration.WEATHER['precipitationNC']
        self.minDailyTemperatureNC = self._configuration.WEATHER['minDailyTemperatureNC']
        self.maxDailyTemperatureNC = self._configuration.WEATHER['maxDailyTemperatureNC']
        self.etpFileNC = self._configuration.WEATHER['refEvapotranspirationNC']        
        self.check_input_filenames()

    def check_input_filenames(self):
        self.check_format_args([
            self.preFileNC,
            self.minDailyTemperatureNC,
            self.maxDailyTemperatureNC,
            self.etpFileNC
            ])
        
    def check_format_args(self, filenames):
        for filename in filenames:
            format_args = file_handling.get_format_args(filename)
            format_args_ok = file_handling.check_format_args_ok(format_args, ['day','month','year'])
            if len(format_args) > 0:
                if not format_args_ok:
                    msg = 'Filename ' + filename + ' contains invalid format arguments: only day, month and year are allowable'
                    raise ModelError(msg)
                
    def set_nc_variable_names(self):
        self.preVarName = self._configuration.WEATHER['precipitationVarName']
        self.tminVarName = self._configuration.WEATHER['minDailyTemperatureVarName']
        self.tmaxVarName = self._configuration.WEATHER['maxDailyTemperatureVarName']
        self.refEvapotranspirationVarName = self._configuration.WEATHER['refEvapotranspirationVarName']
        self.check_nc_variable_names()
        
    def check_nc_variable_names(self):
        filenames = [self.preFileNC, self.minDailyTemperatureNC, self.maxDailyTemperatureNC, self.etpFileNC]
        variable_names = [self.preVarName, self.tminVarName, self.tmaxVarName, self.refEvapotranspirationVarName]
        day, month, year = (self._modelTime.startTime.day, self._modelTime.startTime.month, self._modelTime.year)        
        result = []
        msg = []
        for filename,variable in zip(filenames,variable_names):
            variable_in_nc = file_handling.check_if_nc_has_variable(filename.format(day=day, month=month, year=year), variable)
            result.append(variable_in_nc)
            if not variable_in_nc:
                msg.append('File ' + str(filename) + ' does not contain variable ' + str(variable) + '\n')

        if not all(result):
            msg = '\n'.join(msg)
            raise ModelError(msg)
        
    def set_weather_conversion_factors(self):
        self.preConst = 0.0
        self.preFactor = 1.0
        self.tminConst = 0.0
        self.tminFactor = 1.0
        self.tmaxConst = 0.0
        self.tmaxFactor = 1.0
        self.etrefConst = 0.0
        self.etrefFactor = 1.0
        if 'precipitationConstant' in self._configuration.WEATHER:
            self.preConst = np.float64(self._configuration.WEATHER['precipitationConstant'])
        if 'precipitationFactor' in self._configuration.WEATHER:
            self.preFactor = np.float64(self._configuration.WEATHER['precipitationFactor'])
        if 'minDailyTemperatureConstant' in self._configuration.WEATHER:
            self.tminConst = np.float64(self._configuration.WEATHER['minDailyTemperatureConstant'])
        if 'minDailyTemperatureFactor' in self._configuration.WEATHER:
            self.tminFactor = np.float64(self._configuration.WEATHER['minDailyTemperatureFactor'])
        if 'maxDailyTemperatureConstant' in self._configuration.WEATHER:
            self.tmaxConst = np.float64(self._configuration.WEATHER['maxDailyTemperatureConstant'])
        if 'maxDailyTemperatureFactor' in self._configuration.WEATHER:
            self.tmaxFactor = np.float64(self._configuration.WEATHER['maxDailyTemperatureFactor'])
        if 'ETpotConstant' in self._configuration.WEATHER:
            self.etrefConst = np.float64(self._configuration.WEATHER['refEvapotranspirationConstant'])
        if 'ETpotFactor' in self._configuration.WEATHER:
            self.etrefFactor = np.float64(self._configuration.WEATHER['refEvapotranspirationFactor'])
        
    def adjust_precipitation_input_data(self):
        self.precipitation = self.preConst + self.preFactor * self.precipitation
        self.precipitation = np.maximum(0.0, self.precipitation)
        self.precipitation[np.isnan(self.precipitation)] = 0.0
        self.precipitation = np.floor(self.precipitation * 100000.)/100000.

    def read_precipitation_data(self):
        method_for_time_index = None        
        # fn = self.preFileNC.format(day=self._modelTime.currTime.day, month=self._modelTime.currTime.month, year=self._modelTime.currTime.year)
        # ds = xr.open_dataset(fn)
        # ar = ds['precipitation'].sel(time=self._modelTime.fulldate)
        # print(ar.values)
        self.precipitation = file_handling.netcdf_to_array(
            self.preFileNC.format(
                day=self._modelTime.currTime.day,
                month=self._modelTime.currTime.month,
                year=self._modelTime.currTime.year),
            self.preVarName,
            str(self._modelTime.fulldate),
            useDoy = method_for_time_index,
            cloneMapFileName = self.cloneMapFileName,
            LatitudeLongitude = True
        )[self.landmask][None,None,:]
        print(self.preVarName)
        print(self.precipitation)
        self.adjust_precipitation_input_data()

    def adjust_temperature_data(self):
        self.tmin = self.tminConst + self.tminFactor * self.tmin
        self.tmax = self.tmaxConst + self.tmaxFactor * self.tmax
        self.tmin = np.round(self.tmin * 1000.) / 1000.
        self.tmax = np.round(self.tmax * 1000.) / 1000.
        
    def read_temperature_data(self):
        method_for_time_index = None
        self.tmin = file_handling.netcdf_to_array(
            self.minDailyTemperatureNC.format(
                day=self._modelTime.currTime.day,
                month=self._modelTime.currTime.month,
                year=self._modelTime.currTime.year),
            self.tminVarName,
            str(self._modelTime.fulldate),
            useDoy = method_for_time_index,
            cloneMapFileName = self.cloneMapFileName,
            LatitudeLongitude = True)[self.landmask][None,None,:]

        self.tmax = file_handling.netcdf_to_array(
            self.maxDailyTemperatureNC.format(
                day=self._modelTime.currTime.day,
                month=self._modelTime.currTime.month,
                year=self._modelTime.currTime.year),
            self.tmaxVarName,
            str(self._modelTime.fulldate),
            useDoy = method_for_time_index,
            cloneMapFileName = self.cloneMapFileName,
            LatitudeLongitude = True)[self.landmask][None,None,:]

        self.adjust_temperature_data()

    def adjust_reference_ET_data(self):
        self.referencePotET = self.etrefConst + self.etrefFactor * self.referencePotET

    def read_reference_ET_data(self):
        method_for_time_index = None
        self.referencePotET = file_handling.netcdf_to_array(
            self.etpFileNC.format(
                day=self._modelTime.currTime.day,
                month=self._modelTime.currTime.month,
                year=self._modelTime.currTime.year),
            self.refEvapotranspirationVarName,
            str(self._modelTime.fulldate), 
            useDoy = method_for_time_index,
            cloneMapFileName=self.cloneMapFileName,
            LatitudeLongitude = True)[self.landmask][None,None,:]
        self.adjust_reference_ET_data()
        
    def read_reference_EW_data(self):
        # **TODO**
        self.EWref = self.referencePotET.copy()
        
    def dynamic(self):
        self.read_precipitation_data()
        self.read_temperature_data()
        self.read_reference_ET_data()
        self.read_reference_EW_data()  # for open water evaporation

