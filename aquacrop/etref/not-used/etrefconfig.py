#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os
import sys
import string
import pandas

from hm.config import Configuration
from hm import utils

import time
import datetime
import shutil
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ETRefConfiguration(Configuration):
    
    def set_config(self, system_arguments = None):
        super(ETRefConfiguration, self).set_config(
            system_arguments = system_arguments
        )
        self.set_reference_ET_estimation_method()
        self.set_meteo_input_files()
        self.set_reporting_options()
        
    def set_reference_ET_estimation_method(self):
        if 'method' not in list(self.ET_METHOD.keys()):
            logger.error('method not provided')
        elif self.ET_METHOD['method'] == "None":
            logger.error('method cannot be None')
        else:
            method_selection = [x.strip() for x in self.ET_METHOD['method'].split(",")]
            allowable_methods = ['PenmanMonteith','PriestleyTaylor','Hargreaves']
            methods_ok = all([method in allowable_methods for method in method_selection])
            if not methods_ok:
                logger.error('method must be one or more of PenmanMonteith, PriestleyTaylor, Hargreaves')
            else:
                self.RefETMethodSelection = method_selection

    def set_meteo_input_files(self):
        self.set_meteo_variable_names()
        # self.repair_meteo_scale_factor()
        # self.repair_meteo_offset()
        self.set_meteo_input_options()

    def set_meteo_variable_names(self):
        pass
        # """Function which checks that for each input file key, there 
        # are corresponding keys which specify the variable name and
        # time dimension name in the netCDF file
        # """
        # msg = ''
        # missing_varname = False
        # missing_timedimname = False
        # # for filename, varname, timedimname in zip(self.filename_dict.filename, self.filename_dict.varname, self.filename_dict.timedimname):
        # for filename, varname in zip(self.filename_dict.filename, self.filename_dict.varname):
        #     filename_in_config = filename in list(self.WEATHER.keys())
        #     if filename_in_config:
        #         filename_is_not_none = not self.WEATHER[filename] == "None"
        #         if filename_is_not_none:
        #             if not varname in list(self.WEATHER.keys()):
        #                 missing_varname = True
        #                 msg += 'Configuration does not specify ' + varname + ' \n'
        #             # if not timedimname in list(self.WEATHER.keys()):
        #             #     missing_timedimname = True
        #             #     msg += 'Configuration does not specify ' + varname + ' \n'
                    
        #     if not filename_in_config:
        #         self.WEATHER[filename] = "None"
        #         self.WEATHER[varname] = "None"
        #         # self.WEATHER[timedimname] = "None"
                        
        # if missing_varname or missing_timedimname:
        #     logger.error(msg)

    # def repair_meteo_scale_factor(self):
    #     for scalefactor in self.filename_dict.scalefactor:
    #         scalefactor_in_config = scalefactor in list(self.WEATHER.keys())
    #         if scalefactor_in_config:
    #             self.WEATHER[scalefactor] = np.float64(self.WEATHER[scalefactor])
    #         else:
    #             self.WEATHER[scalefactor] = 1.
                
    # def repair_meteo_offset(self):
    #     for offset in self.filename_dict.offset:
    #         offset_in_config = offset in list(self.WEATHER.keys())
    #         if offset_in_config:
    #             self.WEATHER[offset] = np.float64(self.WEATHER[offset])
    #         else:
    #             self.WEATHER[offset] = 0.
    def set_meteo_input_options(self):
        self.check_format_args()
        self.set_has_subdaily_temperature()
        self.set_has_mean_daily_temperature()
        self.set_has_min_daily_temperature()
        self.set_has_max_daily_temperature()
        self.set_has_specific_humidity()
        self.set_has_mean_relative_humidity()
        self.set_has_min_relative_humidity()
        self.set_has_max_relative_humidity()
        
    def check_format_args(self):
        pass
                                    
    def set_has_subdaily_temperature(self):
        if 'compute_daily_min_max_temperature_from_subdaily_data' not in list(self.WEATHER.keys()):
            self.WEATHER['compute_daily_min_max_temperature_from_subdaily_data'] = "False"

    def set_has_mean_daily_temperature(self):
        self.has_mean_daily_temperature = False
        if self.TMEAN['filename'] != "":
            self.has_mean_daily_temperature = True

    def set_has_max_daily_temperature(self):
        self.has_max_daily_temperature = False
        if self.TMAX['filename'] != "":
            self.has_max_daily_temperature = True

    def set_has_min_daily_temperature(self):
        self.has_min_daily_temperature = False
        if self.TMIN['filename'] != "":
            self.has_min_daily_temperature = True

    def set_has_specific_humidity(self):
        self.has_specific_humidity = False
        if self.SH['filename'] != "":
            self.has_specific_humidity = True
    
    def set_has_mean_relative_humidity(self):
        self.has_mean_relative_humidity = False
        if self.RHMEAN['filename'] != "":
            self.has_mean_relative_humidity = True

    def set_has_max_relative_humidity(self):
        self.has_max_relative_humidity = False
        if self.RHMAX['filename'] != "":
            self.has_max_relative_humidity = True

    def set_has_min_relative_humidity(self):
        self.has_min_relative_humidity = False
        if self.RHMIN['filename'] != "":
            self.has_min_relative_humidity = True
                    
    def set_reporting_options(self):
        if 'REPORTING' not in self.config_sections:
            self.REPORTING = {}

        if 'report' not in list(self.REPORTING.keys()):
            self.REPORTING['report'] = False
