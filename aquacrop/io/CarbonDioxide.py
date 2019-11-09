#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hm import file_handling

import logging
logger = logging.getLogger(__name__)

class CarbonDioxide(object):

    def __init__(self, CarbonDioxide_variable):
        self.var = CarbonDioxide_variable

    def initial(self):
        self.var.RefConc = 369.41        
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
                LatitudeLongitude = True
            )
            self.var.conc = conc[self.var.landmask]
