#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

# from hm import file_handling

from hm.input import HmInputData

import logging
logger = logging.getLogger(__name__)


# class IrrigationSchedule(HmInputData):
#     def __init__(self, model):
#         self.model = model
#         self.filename = \
#             model.config.IRRIGATION_SCHEDULE['filename']
#         self.nc_varname = \
#             model.config.IRRIGATION_SCHEDULE['varname']
#         self.is_1d = model.config.IRRIGATION_SCHEDULE['is_1d']
#         self.xy_dimname = model.config.IRRIGATION_SCHEDULE['xy_dimname']
#         self.model_varname = 'IrrigationSchedule'


class IrrigationManagementParameters(object):

    def __init__(self, model):
        self.model = model
        # self.irrigation_schedule = IrrigationSchedule(model)

    def initial(self):
        self.model.irrMgmtParameterFileNC = self.model.config.IRRIGATION_MANAGEMENT['irrigationManagementNC']
        self.model.parameter_names = [
            'IrrMethod','IrrInterval',
            'SMT1','SMT2','SMT3','SMT4','MaxIrr',
            'AppEff','NetIrrSMT','WetSurf'
        ]        
        for param in self.model.parameter_names:
            if self.model.irrMgmtParameterFileNC is not None:
                try:
                    d = file_handling.netcdf_to_arrayWithoutTime(
                        self.model.irrMgmtParameterFileNC,
                        param,
                        cloneMapFileName=self.model.cloneMapFileName)
                    d = d[self.model.landmask_crop].reshape(self.model.nCrop,self.model.domain.nxy)
                except:
                    d = np.zeros((self.model.nCrop, self.model.domain.nxy))
            else:
                d = np.zeros((self.model.nCrop, self.model.domain.nxy))
                
            vars(self.model)[param] = np.broadcast_to(d, (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))

        # check if an irrigation schedule file is required
        if np.sum(self.model.IrrMethod == 3) > 0:
            if self.model._configuration.IRRIGATION_MANAGEMENT['irrigationScheduleNC'] != "None":
                self.model.irrScheduleFileNC = self.model._configuration.IRRIGATION_MANAGEMENT['irrigationScheduleNC']
            else:
                logger.error('IrrMethod equals 3 in some or all places, but irrScheduleNC is not set in configuration file')

        else:
            self.model.irrScheduleFileNC = None

        # TODO: put this somewhere more appropriate
        
        # self.irrigation_schedule.initial()
        self.model.IrrScheduled = np.zeros((self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
            
    def dynamic(self):
        pass
        # self.irrigation_schedule.dynamic()
