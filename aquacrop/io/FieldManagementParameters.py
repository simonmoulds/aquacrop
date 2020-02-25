#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

# from hm import file_handling

class FieldManagementParameters(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        self.read_field_mgmt_parameters()
        
    def read_field_mgmt_parameters(self):
        field_mgmt_parameter_names = [
            'Mulches','MulchPctGS','MulchPctOS',
            'fMulch','Bunds','zBund','BundWater'
        ]
        for param in field_mgmt_parameter_names:
            if self.model.config.FIELD_MANAGEMENT['fieldManagementNC'] is not None:
                try:
                    d = hm.open_hmdatarray(
                        self.model.config.FIELD_MANAGEMENT['fieldManagementNC'],
                        param,
                        self.model.domain
                    )
                    # d = file_handling.netcdf_to_arrayWithoutTime(
                    #     self.model.config.FIELD_MANAGEMENT['fieldManagementNC'],
                    #     param,
                    #     cloneMapFileName=self.model.cloneMapFileName
                    # )
                    # d = d[self.model.landmask_crop].reshape(self.model.nCrop,self.model.domain.nxy)
                except:
                    d = np.zeros((self.model.nCrop, self.model.domain.nxy))
            else:
                d = np.zeros((self.model.nCrop, self.model.domain.nxy))                
            vars(self.model)[param] = np.broadcast_to(d, (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))

    def dynamic(self):
        pass

# def read_params(fn):
#     with open(fn) as f:
#         content = f.read().splitlines()
#     # remove commented lines
#     content = [x for x in content if re.search('^(?!%).*', x)]
#     content = [re.split('\s*:\s*', x) for x in content]
#     params = {}
#     for x in content:
#         if len(x) > 1:
#             nm = x[0]
#             val = x[1]
#             params[nm] = val
#     return params
    
# def FieldManagementParametersPoint(FieldManagementParameters):
    
#     def read_field_mgmt_parameters(self):
#         field_mgmt_parameter_names = [
#             'Mulches','MulchPctGS','MulchPctOS',
#             'fMulch','Bunds','zBund','BundWater'
#         ]
#         field_mgmt_parameter_values = read_params(self.model._configuration.FIELD_MANAGEMENT['soilParameterFile'])
        
#         for param in field_mgmt_parameters:
#             read_from_file = (param in field_mgmt_parameter_values.keys())
#             if read_from_file:
#                 d = field_mgmt_parameter_values[param]
#                 d = np.broadcast_to(d[None,None,:], (self.model.nFarm,self.model.nCrop, self.model.domain.nxy))
#                 vars(self.model)[param] = d.copy()                
#             else:
#                 vars(self.model)[param] = np.full(
#                     (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
#                     0
#                 )
