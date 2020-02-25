#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

# from hm import file_handling

# class FarmParametersPoint(object):
#     def __init__(self, FarmParameters_variable):
#         self.var = FarmParameters_variable
#         self.var.nFarm = 1
#         self.var.AnnualChangeInFarmArea = False
        
#     def initial(self):
#         self.var.FarmArea = 1
        
#     def dynamic(self):
#         pass
    
class FarmParameters(object):

    def __init__(self, model):
        self.model = model
        self.model.nFarm = self.model.config.FARM_PARAMETERS['nFarm']
        self.model.AnnualChangeInFarmArea = self.model.config.FARM_PARAMETERS['AnnualChangeInFarmArea']

    def initial(self):
        pass

    # def read_farm_area(self, date = None):
    #     farm_area = open_hmdataarray(
    #         self.model.config.FARM_PARAMETERS['farmAreaNC'],
    #         self.model.config.FARM_PARAMETERS['farmAreaVarName'],
    #         self.model.domain
    #     )
    #     # if date is None:
    #     #     farm_area = file_handling.netcdf_to_arrayWithoutTime(
    #     #         self.model._configuration.FARM_PARAMETERS['farmAreaNC'],
    #     #         self.model._configuration.FARM_PARAMETERS['farmAreaVarName'],
    #     #         cloneMapFileName = self.model.cloneMapFileName)            
    #     # else:
    #     #     farm_area = file_handling.netcdf_to_array(
    #     #         self.model._configuration.FARM_PARAMETERS['farmAreaNC'],
    #     #         self.model._configuration.FARM_PARAMETERS['farmAreaVarName'],
    #     #         date,
    #     #         useDoy = None,
    #     #         cloneMapFileName = self.model.cloneMapFileName,
    #     #         LatitudeLongitude = True)

    #     # farm_area = np.reshape(
    #     #     farm_area[self.model.landmask_farm],
    #     #     (self.model.nFarm, self.model.domain.nxy))
    #     # farm_area = np.broadcast_to(
    #     #     farm_area[:,None,:],
    #     #     (self.model.nFarm,
    #     #      self.model.nCrop,
    #     #      self.model.domain.nxy)).copy()
    #     # return farm_area
        
    def set_farm_area(self):
        # if self.model.AnnualChangeInFarmArea:
        #     if self.model._modelTime.timestep == 1 or self.model._modelTime.doy == 1:
        #         date = '%04i-%02i-%02i' % (self.model._modelTime.year, 1, 1)
        #         self.model.FarmAreaNew = self.read_farm_area(date = date)
                    
        #     if np.any(self.model.GrowingSeasonDayOne):
        #         self.model.FarmArea[self.model.GrowingSeasonDayOne] = self.model.FarmAreaNew[self.model.GrowingSeasonDayOne]
                
        # else:
        #     if self.model._modelTime.timestep == 1:
        #         if not self.model._configuration.FARM_PARAMETERS['farmAreaNC'] == "None":
        #             self.model.FarmArea = self.read_farm_area(date = None)
        #         else:
        #             self.model.FarmArea = np.ones(
        #                 (self.model.nFarm,
        #                  self.model.nCrop,
        #                  self.model.domain.nxy))
        self.model.FarmArea = np.ones(
            (self.model.nFarm,
             self.model.nCrop,
             self.model.domain.nxy))
        
    def dynamic(self):
        self.set_farm_area()
