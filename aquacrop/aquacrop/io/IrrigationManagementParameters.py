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
        # self.config = self.model.config.IRRIGATION_MANAGEMENT
        # self.irrigation_schedule = IrrigationSchedule(model)

    def initial(self):

        int_irr_params = [
            'IrrMethod', 'IrrInterval'
        ]
        flt_irr_params = [
            'SMT1','SMT2','SMT3','SMT4','MaxIrr',
            'AppEff','NetIrrSMT','WetSurf'
        ]
        # self.model.irrig_parameters_to_read = int_irr_params + flt_irr_params        
        irr_params_to_read = int_irr_params + flt_irr_params        
        # self.model.irrig_parameters_to_read = [
        #     'IrrMethod','IrrInterval',
        #     'SMT1','SMT2','SMT3','SMT4','MaxIrr',
        #     'AppEff','NetIrrSMT','WetSurf'
        # ]
        
        for param in irr_params_to_read:

            if param in int_irr_params:
                datatype = np.int32
            else:
                datatype = np.float64
                
            if param in self.model.config.IRRIGATION_MANAGEMENT.keys():
                # 1 - Try to read from config file
                # TODO: what dimensions should it have?
                parameter_values = np.array(self.model.config.IRRIGATION_MANAGEMENT[param]) 
                if (len(parameter_values) == 1) | (len(parameter_values) == self.model.nFarm):                        
                    vars(self.model)[param] = np.require(
                        np.broadcast_to(
                            parameter_values,
                            (self.model.nFarm,
                             self.model.nCrop,
                             self.model.domain.nxy)
                        ),
                        requirements=['A','O','W','F']
                    )
                else:
                    raise ValueError(
                        "Error reading parameter " + param
                        + " from configuration file: length"
                        + " of parameter list must equal number"
                        + " of farms in simulation"
                    )

            else:        
                # 2 - Try to read from netCDF file
                try:
                    arr = open_hmdataarray(  
                        self.model.config.IRRIGATION_MANAGEMENT['filename'],
                        param,
                        self.model.domain,
                        self.model.config.IRRIGATION_MANAGEMENT['is_1d'],
                        self.model.config.IRRIGATION_MANAGEMENT['xy_dimname'],
                    )
                    vars(self.model)[param] = np.require(
                        np.broadcast_to(
                            arr.values,
                            (self.model.nFarm, self.model.nCrop, self.model.domain.nxy)
                        ),
                        dtype=datatype,
                        requirements=['A','O','W','F']
                    )
                    # arr = open_hmdataarray(
                    #     self.config['irrigationManagementNC'],
                    #     param,
                    #     self.model.domain
                    # )
                    # vars(self.model)[param] = np.require(
                    #     np.broadcast_to(
                    #         arr.values,
                    #         (self.model.nFarm,
                    #          self.model.nCrop,
                    #          self.model.domain.nxy)
                    #     ),
                    #     requirements=['A','O','W','F']
                    # )                    
                    
                # 3 - Set to zero
                except:
                    vars(self.model)[param] = np.require(
                        np.zeros((
                            self.model.nFarm,
                            self.model.nCrop,
                            self.model.domain.nxy
                        )),
                        requirements=['A','O','W','F']
                    )
            
            # if self.model.irrMgmtParameterFileNC is not None:
            #     try:
            #         d = file_handling.netcdf_to_arrayWithoutTime(
            #             self.model.irrMgmtParameterFileNC,
            #             param,
            #             cloneMapFileName=self.model.cloneMapFileName)
            #         d = d[self.model.landmask_crop].reshape(self.model.nCrop,self.model.domain.nxy)
            #     except:
            #         d = np.zeros((self.model.nCrop, self.model.domain.nxy))
            # else:
            #     d = np.zeros((self.model.nCrop, self.model.domain.nxy))
                
            # vars(self.model)[param] = np.broadcast_to(d, (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))

        # # check if an irrigation schedule file is required
        # if np.any(self.model.IrrMethod == 3) > 0:
            
        #     if self.config['irrigationScheduleNC'] != None:
        #         self.model.irrScheduleFileNC = self.model._configuration.IRRIGATION_MANAGEMENT['irrigationScheduleNC']
        #     else:
        #         logger.error('IrrMethod equals 3 in some or all places, but irrScheduleNC is not set in configuration file')

        # else:
        #     self.model.irrScheduleFileNC = None
        
        # TODO: put this somewhere more appropriate        
        # self.irrigation_schedule.initial()
        self.model.IrrScheduled = np.require(
            np.zeros((self.model.nFarm, self.model.nCrop, self.model.domain.nxy)),
            dtype=np.float64,
            requirements=['A','O','W','F']
        )        
            
    def dynamic(self):
        pass
        # self.irrigation_schedule.dynamic()
