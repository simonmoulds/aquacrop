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
        self.config = self.model.config.FIELD_MANAGEMENT

    def initial(self):
        self.read_field_mgmt_parameters()
        
    def read_field_mgmt_parameters(self):
        field_mgmt_parameter_names = [
            'Mulches','MulchPctGS','MulchPctOS',
            'fMulch','Bunds','zBund','BundWater'
        ]
        for param in field_mgmt_parameter_names:            
            if param in self.config.keys():
                # should have length equal to number of farms
                parameter_values = np.array(self.config[param]) 
                if (len(parameter_values) == 1) | (len(parameter_values) == self.model.nFarm):                        
                    vars(self.model)[param] = np.broadcast_to(
                        parameter_values,
                        (self.model.nFarm,
                         self.model.nCrop,
                         self.model.domain.nxy)
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
                        self.config['fieldManagementNC'],
                        param,
                        self.model.domain
                    )
                    vars(self.model)[param] = np.broadcast_to(
                        arr.values,
                        (self.model.nFarm,
                         self.model.nCrop,
                         self.model.domain.nxy)
                    )
                    
                # 3 - Set to zero
                except:
                    vars(self.model)[param] = np.zeros((
                        self.model.nFarm,
                        self.model.nCrop,
                        self.model.domain.nxy
                    ))                         
                        
    def dynamic(self):
        pass

