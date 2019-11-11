#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime

from hm import file_handling

from .SoilHydraulicParameters import *
from .SoilParameters import *
from .CropParameters import *
from .FarmParameters import *
from .IrrigationManagementParameters import *
from .FieldManagementParameters import *

class BaseClass(object):
    def __init__(self, var, configuration):
        self.var = var
        self.configuration = configuration
    
class CoverFraction(BaseClass):

    def __init__(self, var, configuration):
        super(CoverFraction, self).__init__(var, configuration)
        # self.dynamicLandCover = bool(int(self.var._configuration.LAND_COVER['dynamicLandCover']))
        # self.staticLandCoverYear = None
        # if not self.dynamicLandCover:
        #     self.staticLandCoverYear = int(self.var._configuration.LAND_COVER['staticLandCoverYear'])
        
    def initial(self):
        self.coverFractionNC = str(self.configuration['landCoverFractionNC'])
        self.coverFractionVarName = str(self.configuration['landCoverFractionVarName'])
        self.var.cover_fraction = np.zeros((self.var.nCell))
        self.update_cover_fraction()
        
    def update_cover_fraction(self):
        
        # # TODO: make flexible the day on which land cover is changed
        
        # start_of_model_run = (self.var._modelTime.timeStepPCR == 1)
        # start_of_year = (self.var._modelTime.doy == 1)
        # if self.dynamicLandCover:
        #     if start_of_model_run or start_of_year:
        #         date = datetime.datetime(self.var._modelTime.year, 1, 1, 0, 0, 0)
        #         cover_fraction = file_handling.netcdf_to_array(
        #             self.coverFractionNC.format(
        #                 day=self.var._modelTime.currTime.day,
        #                 month=self.var._modelTime.currTime.month,
        #                 year=self.var._modelTime.currTime.year),
        #             self.coverFractionVarName,
        #             date,
        #             # useDoy = method_for_time_index,
        #             # cloneMapAttributes = self.cloneMapAttributes,
        #             cloneMapFileName = self.var.cloneMapFileName,
        #             LatitudeLongitude = True)[self.var.landmask]
        #         cover_fraction = np.float64(cover_fraction)
        #         self.var.cover_fraction = cover_fraction
        # else:
        #     if start_of_model_run:
        #         date = datetime.datetime(self.staticLandCoverYear, 1, 1, 0, 0, 0)
        #         cover_fraction = file_handling.netcdf_to_array(
        #             self.coverFractionNC.format(
        #                 day=self.var._modelTime.currTime.day,
        #                 month=self.var._modelTime.currTime.month,
        #                 year=self.var._modelTime.currTime.year),
        #             self.coverFractionVarName,
        #             date,
        #             # useDoy = method_for_time_index,
        #             # cloneMapAttributes = self.cloneMapAttributes,
        #             cloneMapFileName = self.var.cloneMapFileName,
        #             LatitudeLongitude = True)[self.var.landmask]
        #         cover_fraction = np.float64(cover_fraction)
        #         self.var.cover_fraction = cover_fraction
        self.var.cover_fraction = np.ones_like(self.var.landmask)  # TEMPORARY
        
    def dynamic(self):
        self.update_cover_fraction()

class LandCoverParameters(object):
    def __init__(self, LandCoverParameters_variable, config_section_name):
        self.var = LandCoverParameters_variable
        self.configuration = getattr(
            self.var._configuration,
            config_section_name)

class AquaCropParameters(LandCoverParameters):
    def __init__(self, var, config_section_name):
        super(AquaCropParameters, self).__init__(var, config_section_name)
        self.soil_hydraulic_parameters_module = SoilHydraulicParameters(var, config_section_name)
        self.soil_parameters_module = SoilParameters(var, config_section_name)
        self.cover_fraction_module = CoverFraction(var, self.configuration)        
        self.farm_parameters_module = FarmParameters(var)
        self.crop_parameters_module = CropParameters(var)
        self.crop_area_module = CropArea(var)
        self.field_mgmt_parameters_module = FieldManagementParameters(var)
        self.irrigation_parameters_module = IrrigationManagementParameters(var)

    def initial(self):
        self.soil_hydraulic_parameters_module.initial()
        self.soil_parameters_module.initial()
        self.cover_fraction_module.initial()
        self.farm_parameters_module.initial()
        self.crop_parameters_module.initial()
        self.crop_area_module.initial()
        self.field_mgmt_parameters_module.initial()
        self.irrigation_parameters_module.initial()

    def dynamic(self):
        self.soil_hydraulic_parameters_module.dynamic()
        self.soil_parameters_module.dynamic()
        self.cover_fraction_module.dynamic()
        self.farm_parameters_module.dynamic()
        self.crop_parameters_module.dynamic()
        self.crop_area_module.dynamic()
        self.field_mgmt_parameters_module.dynamic()
        self.irrigation_parameters_module.dynamic()
                
