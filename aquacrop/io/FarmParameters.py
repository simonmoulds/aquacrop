#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm import file_handling

class FarmParametersPoint(object):
    def __init__(self, FarmParameters_variable):
        self.var = FarmParameters_variable
        self.var.nFarm = 1
        self.var.AnnualChangeInFarmArea = False
        
    def initial(self):
        self.var.FarmArea = 1
        
    def dynamic(self):
        pass
    
class FarmParametersGrid(object):

    def __init__(self, FarmParameters_variable):
        self.var = FarmParameters_variable
        self.var.nFarm = int(self.var._configuration.FARM_PARAMETERS['nFarm'])
        self.var.AnnualChangeInFarmArea = bool(int(self.var._configuration.FARM_PARAMETERS['AnnualChangeInFarmArea']))

    def initial(self):
        pass

    def read_farm_area(self, date = None):
        if date is None:
            farm_area = file_handling.netcdf_to_arrayWithoutTime(
                self.var._configuration.FARM_PARAMETERS['farmAreaNC'],
                self.var._configuration.FARM_PARAMETERS['farmAreaVarName'],
                cloneMapFileName = self.var.cloneMapFileName)            
        else:
            farm_area = file_handling.netcdf_to_array(
                self.var._configuration.FARM_PARAMETERS['farmAreaNC'],
                self.var._configuration.FARM_PARAMETERS['farmAreaVarName'],
                date,
                useDoy = None,
                cloneMapFileName = self.var.cloneMapFileName,
                LatitudeLongitude = True)

        farm_area = np.reshape(
            farm_area[self.var.landmask_farm],
            (self.var.nFarm, self.var.nCell))
        farm_area = np.broadcast_to(
            farm_area[:,None,:],
            (self.var.nFarm,
             self.var.nCrop,
             self.var.nCell)).copy()
        return farm_area
        
    def set_farm_area(self):
        self.var.landmask_farm = np.broadcast_to(
            self.var.landmask[None,:,:],
            (self.var.nFarm, self.var.nLat, self.var.nLon))
        
        if self.var.AnnualChangeInFarmArea:
            if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:
                date = '%04i-%02i-%02i' % (self.var._modelTime.year, 1, 1)
                self.var.FarmAreaNew = self.read_farm_area(date = date)
                    
            if np.any(self.var.GrowingSeasonDayOne):
                self.var.FarmArea[self.var.GrowingSeasonDayOne] = self.var.FarmAreaNew[self.var.GrowingSeasonDayOne]
                
        else:
            if self.var._modelTime.timeStepPCR == 1:
                if not self.var._configuration.FARM_PARAMETERS['farmAreaNC'] == "None":
                    self.var.FarmArea = self.read_farm_area(date = None)
                else:
                    self.var.FarmArea = np.ones(
                        (self.var.nFarm,
                         self.var.nCrop,
                         self.var.nCell))
        
    def dynamic(self):
        self.set_farm_area()
