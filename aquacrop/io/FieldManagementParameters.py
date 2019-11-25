#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm import file_handling

class FieldManagementParameters(object):
    def __init__(self, FieldManagementParameters_variable):
        self.var = FieldManagementParameters_variable

    def initial(self):
        self.read_field_mgmt_parameters()
        
    def read_field_mgmt_parameters(self):
        field_mgmt_parameter_names = [
            'Mulches','MulchPctGS','MulchPctOS',
            'fMulch','Bunds','zBund','BundWater'
        ]
        for param in field_mgmt_parameter_names:
            if self.var._configuration.FIELD_MANAGEMENT['fieldManagementNC'] is not None:
                try:
                    d = file_handling.netcdf_to_arrayWithoutTime(
                        self.var._configuration.FIELD_MANAGEMENT['fieldManagementNC'],
                        param,
                        cloneMapFileName=self.var.cloneMapFileName
                    )
                    d = d[self.var.landmask_crop].reshape(self.var.nCrop,self.var.nCell)                    
                except:
                    d = np.zeros((self.var.nCrop, self.var.nCell))
            else:
                d = np.zeros((self.var.nCrop, self.var.nCell))
                
            vars(self.var)[param] = np.broadcast_to(d, (self.var.nFarm, self.var.nCrop, self.var.nCell))

    def dynamic(self):
        pass

def read_params(fn):
    with open(fn) as f:
        content = f.read().splitlines()
    # remove commented lines
    content = [x for x in content if re.search('^(?!%).*', x)]
    content = [re.split('\s*:\s*', x) for x in content]
    params = {}
    for x in content:
        if len(x) > 1:
            nm = x[0]
            val = x[1]
            params[nm] = val
    return params
    
def FieldManagementParametersPoint(FieldManagementParameters):
    
    def read_field_mgmt_parameters(self):
        field_mgmt_parameter_names = [
            'Mulches','MulchPctGS','MulchPctOS',
            'fMulch','Bunds','zBund','BundWater'
        ]
        field_mgmt_parameter_values = read_params(self.var._configuration.FIELD_MANAGEMENT['soilParameterFile'])
        
        for param in field_mgmt_parameters:
            read_from_file = (param in field_mgmt_parameter_values.keys())
            if read_from_file:
                d = field_mgmt_parameter_values[param]
                d = np.broadcast_to(d[None,None,:], (self.var.nFarm,self.var.nCrop, self.var.nCell))
                vars(self.var)[param] = d.copy()                
            else:
                vars(self.var)[param] = np.full(
                    (self.var.nFarm, self.var.nCrop, self.var.nCell),
                    0
                )
