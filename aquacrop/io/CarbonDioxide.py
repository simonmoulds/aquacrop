#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from hm.input import HmInputData

import logging
logger = logging.getLogger(__name__)

refconc = 369.41

class CarbonDioxide(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = \
            self.model.config.CARBON_DIOXIDE['carbonDioxideNC']
        self.nc_varname = \
            self.model.config.CARBON_DIOXIDE['carbonDioxideVarName']
        self.model_varname = 'conc'
        
# class CarbonDioxide(object):
#     def __init__(self, CarbonDioxide_variable):
#         self.var = CarbonDioxide_variable

#     def initial(self):
#         self.var.co2FileNC = self.var._configuration.CARBON_DIOXIDE['carbonDioxideNC']
#         self.var.co2VarName = 'co2' 
#         if 'co2VarName' in self.var._configuration.CARBON_DIOXIDE:
#             self.var.co2VarName = self.var._configuration.CARBON_DIOXIDE['co2VarName']
#         self.var.co2_set_per_year  = False
#         self.var.conc = np.ones((self.var.nCell))
#         # one option would be to simply use xarray directly at this point: load the dataset, extract relevant points, etc.
        
#     def dynamic(self):        
#         if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:
#             date = '%04i-%02i-%02i' %(self.var._modelTime.year, 1, 1)
#             conc = file_handling.netcdf_to_array(
#                 self.var.co2FileNC,
#                 self.var.co2VarName,
#                 date,
#                 useDoy = None,
#                 cloneMapFileName = self.var.cloneMapFileName,
#                 LatitudeLongitude = True,
#                 # oneDimensional = True
#             )
#             self.var.conc = conc[self.var.landmask]
