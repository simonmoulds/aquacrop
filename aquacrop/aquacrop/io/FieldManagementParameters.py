#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

class FieldManagementParameters(object):
    def __init__(self, model):
        self.model = model
        # self.config = self.model.config.FIELD_MANAGEMENT

    def initial(self):
        self.read_field_mgmt_parameters()
        
    def read_field_mgmt_parameters(self):
        int_field_mgmt_params = [
            'Mulches', 'Bunds'
        ]
        flt_field_mgmt_params = [
            'MulchPctGS','MulchPctOS',
            'fMulch','zBund','BundWater'
        ]
        field_mgmt_params = int_field_mgmt_params + flt_field_mgmt_params
        for param in field_mgmt_params:

            if param in int_field_mgmt_params:
                datatype = np.int32
            else:
                datatype = np.float64
            
            if param in self.model.config.IRRIGATION_MANAGEMENT.keys():
                # 1 - Try to read from config file
                    
                # should have length equal to number of farms (?)                
                parameter_values = np.array(self.model.config.IRRIGATION_MANAGEMENT[param]) 
                if (len(parameter_values) == 1) | (len(parameter_values) == self.model.nFarm):                        
                    vars(self.model)[param] = np.require(
                        np.broadcast_to(
                            parameter_values,
                            (self.model.nFarm,
                             self.model.nCrop,
                             self.model.domain.nxy)
                        ),
                        dtype=datatype,
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
                        self.model.config.FIELD_MANAGEMENT['filename'],
                        param,
                        self.model.domain,
                        self.model.config.FIELD_MANAGEMENT['is_1d'],
                        self.model.config.FIELD_MANAGEMENT['xy_dimname'],
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
                    #     self.config['fieldManagementNC'],
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
                    #     dtype=datatype,
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
                        dtype=datatype,
                        requirements=['A','O','W','F']
                    )
                        
    def dynamic(self):
        pass

