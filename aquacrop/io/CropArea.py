#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

# from hm import file_handling

# class CropAreaPoint(object):
#     def __init__(self, CropArea_variable):
#         self.var = CropArea_variable

#     def initial(self):
#         self.var.CurrentCropArea = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
#         self.var.CroplandArea = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
#         self.var.CropArea = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
#         self.var.FarmCropArea = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        
#     def dynamic(self):
#         pass
class CropArea(object):
    def __init__(self, model):
        self.model = model
    def initial(self):
        self.model.CurrentCropArea = np.ones((self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        self.model.CroplandArea = np.ones((self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        self.model.CropArea = np.ones((self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        self.model.FarmCropArea = np.ones((self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        
    def dynamic(self):
        pass
        
# class CropAreaGrid(object):
    
#     def __init__(self, CropArea_variable):
#         self.var = CropArea_variable
#         self.var.AnnualChangeInCropArea = bool(int(self.var._configuration.CROP_PARAMETERS['AnnualChangeInCropArea']))
        
#     def initial(self):
#         pass
    
#     def read_cropland_area(self):
#         if self.var.AnnualChangeInCropArea:
#             if self.var._modelTime.timestep == 1 or self.var._modelTime.doy == 1:
#                 date = '%04i-%02i-%02i' % (self.var._modelTime.year, 1, 1)
#                 CroplandArea = file_handling.netcdf_to_array(
#                     self.var._configuration.CROP_PARAMETERS['croplandAreaNC'],
#                     self.var._configuration.CROP_PARAMETERS['croplandAreaVarName'],
#                     date,
#                     useDoy = None,
#                     cloneMapFileName = self.var.cloneMapFileName,
#                     LatitudeLongitude = True
#                 )
#                 self.var.CroplandArea = CroplandArea[self.var.landmask]
                
#         else:
#             if self.var._modelTime.timestep == 1:
#                 if not self.var._configuration.CROP_PARAMETERS['cropAreaNC'] == "None":
#                     CroplandArea = file_handling.netcdf_to_arrayWithoutTime(
#                         self.var._configuration.CROP_PARAMETERS['croplandAreaNC'],
#                         self.var._configuration.CROP_PARAMETERS['croplandAreaVarName'],
#                         cloneMapFileName = self.var.cloneMapFileName
#                     )
#                     self.var.CroplandArea = CroplandArea[self.var.landmask]
                    
#                 else:
#                     self.var.CroplandArea = np.ones((self.var.domain.nxy)) * self.var.nCrop
                    
#         self.var.CroplandArea = np.broadcast_to(
#             self.var.CroplandArea,
#             (self.var.nFarm,
#              self.var.nCrop,
#              self.var.domain.nxy)
#         )        
#         self.var.CroplandArea = self.var.CroplandArea.astype(np.float64)
        
#     def read_crop_area(self, date = None):
#         if date is None:
#             crop_area = file_handling.netcdf_to_arrayWithoutTime(
#                 self.var._configuration.CROP_PARAMETERS['cropAreaNC'],
#                 self.var._configuration.CROP_PARAMETERS['cropAreaVarName'],
#                 cloneMapFileName = self.var.cloneMapFileName,
#                 LatitudeLongitude = True
#             )
            
#         else:
#             crop_area = file_handling.netcdf_to_array(
#                 self.var._configuration.CROP_PARAMETERS['cropAreaNC'],
#                 self.var._configuration.CROP_PARAMETERS['cropAreaVarName'],
#                 date,
#                 useDoy = None,
#                 cloneMapFileName = self.var.cloneMapFileName,
#                 LatitudeLongitude = True
#             )
            
#         crop_area_has_farm_dimension = (
#             file_handling.check_if_nc_variable_has_dimension(
#                 self.var._configuration.CROP_PARAMETERS['cropAreaNC'],
#                 self.var._configuration.CROP_PARAMETERS['cropAreaVarName'],
#                 'farm'
#             )
#         )

#         if crop_area_has_farm_dimension:
#             crop_area = np.reshape(
#                 crop_area[self.var.landmask_farm_crop],
#                 (self.var.nFarm,
#                  self.var.nCrop,
#                  self.var.domain.nxy)
#             )
#         else:
#             crop_area = np.reshape(
#                 crop_area[self.var.landmask_crop],
#                 (self.var.nCrop,
#                  self.var.domain.nxy)
#             )
#             crop_area = np.broadcast_to(
#                 crop_area[None,:,:],
#                 (self.var.nFarm,
#                  self.var.nCrop,
#                  self.var.domain.nxy)
#             )
            
#         crop_area = crop_area.astype(np.float64)
#         return crop_area

#     def scale_crop_area(self):
#         """Function to scale crop area to match cropland area."""
#         pd = self.var.PlantingDate.copy()[0,...]
#         hd = self.var.HarvestDate.copy()[0,...]
#         hd[hd < pd] += 365
#         max_harvest_date = int(np.max(hd))
#         day_idx = (
#             np.arange(1, max_harvest_date + 1)[:,None,None]
#             * np.ones((self.var.nCrop, self.var.domain.nxy))[None,:,:]
#         )
#         growing_season_idx = ((day_idx >= pd) & (day_idx <= hd))
#         crop_area = self.var.CropArea[0,...]  # remove farm dimension
#         crop_area_daily = crop_area[None,...] * growing_season_idx  # get daily crop area
#         total_crop_area_daily = np.sum(crop_area_daily, axis=1)     # sum of all crops grown on a given day
#         max_crop_area = np.max(total_crop_area_daily, axis=0)       # get the max crop area considering all growing seasons
#         scale_factor = np.divide(
#             self.var.CroplandArea,
#             max_crop_area,
#             out=np.zeros_like(self.var.CroplandArea),
#             where=max_crop_area>0
#         )  # compute scale factor by dividing cropland area by max crop area
#         self.var.CropArea *= scale_factor
        
#         # # TEST:
#         # crop_area = self.var.CropArea[0,...]
#         # crop_area_daily = crop_area[None,...] * growing_season_idx
#         # total_crop_area_daily = np.sum(crop_area_daily, axis=1)
#         # max_crop_area = np.max(total_crop_area_daily, axis=0)

#     def set_crop_area(self):
#         """Function to read crop area"""
#         if self.var.AnnualChangeInCropArea:
#             if self.var._modelTime.timestep == 1 or self.var._modelTime.doy == 1:
#                 # In this case crop area is updated on the first day of each year,
#                 # hence in order to prevent the area under a specific crop changing
#                 # mid-season, it is necessary to introduce an intermediate variable
#                 # (i.e. CropAreaNew) and only update the area on the first day of
#                 # the growing season.
#                 date = '%04i-%02i-%02i' % (self.var._modelTime.year, 1, 1)
#                 self.var.CropAreaNew = self.read_crop_area(date = date)
                    
#             if np.any(self.var.GrowingSeasonDayOne):
#                 self.var.CropArea[self.var.GrowingSeasonDayOne] = self.var.CropAreaNew[self.var.GrowingSeasonDayOne]
#                 self.scale_crop_area()
                
#         else:
#             if self.var._modelTime.timestep == 1:
#                 if not self.var._configuration.CROP_PARAMETERS['cropAreaNC'] == "None":
#                     # If crop area doesn't change then there is no need for an
#                     # intermediate variable
#                     self.var.CropArea = self.read_crop_area(date = None)
#                 else:
#                     self.var.CropArea = np.ones(
#                         (self.var.nFarm,
#                          self.var.nCrop,
#                          self.var.domain.nxy)
#                     )
#                 self.scale_crop_area()

#     def compute_current_crop_area(self):
#         """Function to work out the relative area of current crops and 
#         fallow area, and divide fallow land proportionally between 
#         crops that are not currently grown. This is necessary because 
#         the program continues to compute the water balance for crops 
#         which are not currently grown).
#         """        
#         crop_area = self.var.CropArea * self.var.GrowingSeasonIndex
#         total_crop_area = np.sum(crop_area, axis=(0,1))

#         # Compute scale factor to represent the relative area of
#         # each crop not currently grown by dividing fallow area
#         # by total fallow area
#         crop_area_not_grown = (
#             self.var.CropArea
#             * np.logical_not(self.var.GrowingSeasonIndex)
#         )
#         total_crop_area_not_grown = np.sum(
#             crop_area_not_grown
#             * np.logical_not(self.var.GrowingSeasonIndex)
#         )
#         scale_factor = np.divide(
#             crop_area_not_grown,
#             total_crop_area_not_grown,
#             out=np.zeros_like(crop_area_not_grown),
#             where=total_crop_area_not_grown>0)
        
#         # Compute the area which remains fallow during the growing
#         # season, and scale according to the relative area of the
#         # crops *not* currently grown.
#         target_fallow_area = np.clip(self.var.CroplandArea - total_crop_area, 0, None)
#         fallow_area = target_fallow_area * scale_factor        
#         self.var.CurrentCropArea = self.var.CropArea.copy()        
#         self.var.CurrentCropArea[np.logical_not(self.var.GrowingSeasonIndex)] = (
#             (fallow_area[np.logical_not(self.var.GrowingSeasonIndex)])
#         )

#         # CurrentCropArea represents the crop area in the entire grid
#         # cell; it does not represent the area of the respective crops
#         # grown on the respective farms. Thus, create a farm scale factor
#         # to scale the current cropped area down to the level of individual
#         # farms.
#         farm_scale_factor = np.divide(
#             self.var.FarmArea,
#             self.var.CroplandArea,
#             out=np.zeros_like(self.var.CroplandArea),
#             where=self.var.CroplandArea>0
#         )

#         # The farm scale factor represents the fraction of total cropland
#         # found within each farm. Thus, multiplying the scale factor by
#         # CurrentCropArea provides an estimate of the area of each crop
#         # within a given farm.        
#         self.var.FarmCropArea = farm_scale_factor * self.var.CurrentCropArea
        
#     def dynamic(self):
#         self.read_cropland_area()
#         self.set_crop_area()
#         self.compute_current_crop_area()
