#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from hm import file_handling

import logging
logger = logging.getLogger(__name__)

# see https://gis.stackexchange.com/a/225239 for extracting data based on points

class CarbonDioxide(object):
    def __init__(self, CarbonDioxide_variable):
        self.var = CarbonDioxide_variable
        self.var.RefConc = 369.41        

    def initial(self):
        self.var.co2FileNC = self.var._configuration.CARBON_DIOXIDE['carbonDioxideNC']
        self.var.co2VarName = 'co2' 
        if 'co2VarName' in self.var._configuration.CARBON_DIOXIDE:
            self.var.co2VarName = self.var._configuration.CARBON_DIOXIDE['co2VarName']
        self.var.co2_set_per_year  = False
        self.var.conc = np.ones((self.var.nCell))
        
    def dynamic(self):        
        if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:
            date = '%04i-%02i-%02i' %(self.var._modelTime.year, 1, 1)
            conc = file_handling.netcdf_to_array(
                self.var.co2FileNC,
                self.var.co2VarName,
                date,
                useDoy = None,
                cloneMapFileName = self.var.cloneMapFileName,
                LatitudeLongitude = True,
                # oneDimensional = True
            )
            self.var.conc = conc[self.var.landmask]

# TODO: use this in conversion tool
# class CarbonDioxidePoint(CarbonDioxide):
        
#     def initial(self):
#         d = read_csv(
#             self.var._configuration.CARBON_DIOXIDE['carbonDioxideNC'],
#             delimiter='\s+|\t',
#             header=None,
#             names=['Year','CO2'],
#             skiprows=0,
#             comment="%",
#             engine='python'
#         )        
#         years = d['Year'].values
#         data = d['CO2'].values
#         f = interpolate.interp1d(years, data, kind="linear")
#         newyears = np.arange(years.min(), years.max() + 1)
#         newdata = f(newyears)        
#         self.var.carbonDioxideData = pd.DataFrame({'CO2' : newdata}, index=newyears)
#         self.var.co2_set_per_year  = False
#         self.var.conc = np.ones((self.var.nCell))

#     def dynamic(self):        
#         if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:
#             self.var.conc = self.var.carbonDioxideData['CO2'][self.var._modelTime.year]
