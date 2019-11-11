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
        # TODO: similar to crop parameters, use a database of default values (i.e. 0 in most cases)
        self.var.fieldMgmtParameterFileNC = self.var._configuration.FIELD_MANAGEMENT['fieldManagementNC']
        self.var.parameter_names = [
            'Mulches','MulchPctGS','MulchPctOS',
            'fMulch','Bunds','zBund','BundWater'
        ]
        for param in self.var.parameter_names:
            if self.var.fieldMgmtParameterFileNC is not None:
                try:
                    d = file_handling.netcdf_to_arrayWithoutTime(
                        self.var.fieldMgmtParameterFileNC,
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
