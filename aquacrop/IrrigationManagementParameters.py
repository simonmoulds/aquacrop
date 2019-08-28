#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm import file_handling

class IrrigationManagementParameters(object):

    def __init__(self, IrrigationManagementParameters_variable):
        self.var = IrrigationManagementParameters_variable

    def initial(self):
        self.var.irrMgmtParameterFileNC = self.var._configuration.IRRIGATION_MANAGEMENT['irrigationManagementNC']
        self.var.parameter_names = [
            'IrrMethod','IrrInterval',
            'SMT1','SMT2','SMT3','SMT4','MaxIrr',
            'AppEff','NetIrrSMT','WetSurf'
        ]        
        for param in self.var.parameter_names:
            if self.var.irrMgmtParameterFileNC is not None:
                try:
                    d = file_handling.netcdf_to_arrayWithoutTime(
                        self.var.irrMgmtParameterFileNC,
                        param,
                        cloneMapFileName=self.var.cloneMapFileName)
                    d = d[self.var.landmask_crop].reshape(self.var.nCrop,self.var.nCell)
                except:
                    d = np.zeros((self.var.nCrop, self.var.nCell))
            else:
                d = np.zeros((self.var.nCrop, self.var.nCell))
                
            vars(self.var)[param] = np.broadcast_to(d, (self.var.nFarm, self.var.nCrop, self.var.nCell))

        # check if an irrigation schedule file is required
        if np.sum(self.var.IrrMethod == 3) > 0:
            if self.var._configuration.IRRIGATION_MANAGEMENT['irrigationScheduleNC'] != "None":
                self.var.irrScheduleFileNC = self.var._configuration.IRRIGATION_MANAGEMENT['irrigationScheduleNC']
            else:
                logger.error('IrrMethod equals 3 in some or all places, but irrScheduleNC is not set in configuration file')

        else:
            self.var.irrScheduleFileNC = None

    def dynamic(self):
        pass
