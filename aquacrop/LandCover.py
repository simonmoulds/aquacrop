#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm.reporting import Reporting

from .io.LandCoverParameters import AquaCropParameters
from .io.CarbonDioxide import CarbonDioxide
from .io.InitialCondition import InitialCondition
from .GrowingDegreeDay import GrowingDegreeDay
from .CheckGroundwaterTable import CheckGroundwaterTable
from .PreIrrigation import PreIrrigation
from .Drainage import Drainage
from .RainfallPartition import RainfallPartition
from .RootZoneWater import RootZoneWater
from .Irrigation import Irrigation
from .Infiltration import Infiltration
from .CapillaryRise import CapillaryRise
from .Germination import Germination
from .GrowthStage import GrowthStage
from .RootDevelopment import RootDevelopment
from .WaterStress import WaterStress
from .CanopyCover import CanopyCover
from .SoilEvaporation import SoilEvaporation
from .Transpiration import Transpiration
from .Evapotranspiration import Evapotranspiration
from .Inflow import Inflow
from .BiomassAccumulation import BiomassAccumulation
from .TemperatureStress import TemperatureStress
from .HarvestIndex import HarvestIndex, HarvestIndexAdjusted
from .CropYield import CropYield

from .io import variable_list_crop

class LandCover(object):
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.time = model.time
        self.domain = model.domain
        self.weather = model.weather_module
        self.groundwater = model.groundwater_module

    def initial(self):
        pass
    
    def dynamic(self):
        pass

class Cropland(LandCover):
    def __init__(self, model):
        super(Cropland, self).__init__(model)
        self.carbon_dioxide_module = CarbonDioxide(self)
        self.lc_parameters_module = AquaCropParameters(self)
        self.initial_condition_module = InitialCondition(self)
        self.gdd_module = GrowingDegreeDay(self)        
        self.check_groundwater_table_module = CheckGroundwaterTable(self)
        self.pre_irrigation_module = PreIrrigation(self)
        self.drainage_module = Drainage(self)
        self.rainfall_partition_module = RainfallPartition(self)
        self.root_zone_water_module = RootZoneWater(self)
        self.irrigation_module = Irrigation(self)
        self.infiltration_module = Infiltration(self)
        self.capillary_rise_module = CapillaryRise(self)
        self.germination_module = Germination(self)
        self.growth_stage_module = GrowthStage(self)
        self.root_development_module = RootDevelopment(self)
        self.water_stress_module = WaterStress(self)
        self.canopy_cover_module = CanopyCover(self)
        self.soil_evaporation_module = SoilEvaporation(self)
        self.transpiration_module = Transpiration(self)
        self.evapotranspiration_module = Evapotranspiration(self)
        self.inflow_module = Inflow(self)
        # merge the following five classes?
        self.HI_ref_current_day_module = HarvestIndex(self)
        self.biomass_accumulation_module = BiomassAccumulation(self)
        self.temperature_stress_module = TemperatureStress(self)
        self.harvest_index_module = HarvestIndexAdjusted(self)
        self.crop_yield_module = CropYield(self)
#         self.grid_cell_mean_module = GridCellMean(self)
    
    def initial(self):
        self.carbon_dioxide_module.initial()
        self.lc_parameters_module.initial()
        self.gdd_module.initial()
        self.initial_condition_module.initial()
        self.check_groundwater_table_module.initial()
        self.pre_irrigation_module.initial()
        self.drainage_module.initial()
        self.rainfall_partition_module.initial()
        self.root_zone_water_module.initial()
        self.irrigation_module.initial()
        self.infiltration_module.initial()
        self.capillary_rise_module.initial()
        self.germination_module.initial()
        self.growth_stage_module.initial()
        self.root_development_module.initial()
        self.water_stress_module.initial()
        self.canopy_cover_module.initial()
        self.soil_evaporation_module.initial()
        self.transpiration_module.initial()
        self.evapotranspiration_module.initial()
        self.inflow_module.initial()
        self.HI_ref_current_day_module.initial()
        self.biomass_accumulation_module.initial()
        self.temperature_stress_module.initial()
        self.harvest_index_module.initial()
        self.crop_yield_module.initial()
#         self.grid_cell_mean_module.initial()
        self.reporting_module = Reporting(self, variable_list_crop, 'CROP_PARAMETERS')
        self.reporting_module.initial()
        
    def dynamic(self):
        print('--------------------')
        print(self.model.time._timestep)
        # print(self.th[0,0,:,0])
        self.carbon_dioxide_module.dynamic(method='pad')
        self.lc_parameters_module.dynamic()
        self.gdd_module.dynamic()
        self.growth_stage_module.dynamic()
        self.initial_condition_module.dynamic()
        self.check_groundwater_table_module.dynamic()
        self.pre_irrigation_module.dynamic()
        # print('after pre irrigation: ', self.th[0,0,1,0])
        # print(self.th[0,0,1,0])
        self.drainage_module.dynamic()
        # print('after drainage: ', self.th[0,0,1,0])
        self.rainfall_partition_module.dynamic()
        # print('after rainfall partition: ', self.th[0,0,1,0])
        self.root_zone_water_module.dynamic()
        # print('after root zone water: ', self.th[0,0,1,0])
        self.irrigation_module.dynamic()
        # print('after irrigation: ', self.th[0,0,1,0])
        self.infiltration_module.dynamic()
        # print('after infiltration: ', self.th[0,0,1,0])
        self.capillary_rise_module.dynamic()
        # print('after cap rise: ', self.th[0,0,1,0])
        self.germination_module.dynamic()
        # self.growth_stage_module.dynamic() # TODO: compare against AquaCropOS - don't think this is needed
        # print('after germination: ', self.th[0,0,1,0])
        self.root_development_module.dynamic()
        # print('after root dev: ', self.th[0,0,1,0])
        self.root_zone_water_module.dynamic()
        # print('after root zone water: ', self.th[0,0,1,0])
        self.water_stress_module.dynamic(beta=True)
        # print('after water stress: ', self.th[0,0,1,0])
        self.canopy_cover_module.dynamic()
        # print('after canopy cover: ', self.th[0,0,1,0])
        # print(self.th[0,0,:,0])
        self.soil_evaporation_module.dynamic()
        # print('after soil evaporation: ', self.th[0,0,1,0])
        # print(self.th[0,0,:,0])
        self.root_zone_water_module.dynamic()
        # print('after root zone water: ', self.th[0,0,1,0])
        self.water_stress_module.dynamic(beta=True)
        # print('after water stress: ', self.th[0,0,1,0])
        self.transpiration_module.dynamic()
        # print('after transpiration: ', self.th[0,0,1,0])
        self.evapotranspiration_module.dynamic()
        self.inflow_module.dynamic()
        self.HI_ref_current_day_module.dynamic()
        self.temperature_stress_module.dynamic()
        self.biomass_accumulation_module.dynamic()
        self.root_zone_water_module.dynamic()
        self.water_stress_module.dynamic(beta=True)
        self.temperature_stress_module.dynamic()
        self.harvest_index_module.dynamic()
        self.crop_yield_module.dynamic()
        # print(self.th[0,0,1,0])
        # print(self.Y[0,0,0])
        self.root_zone_water_module.dynamic()
        self.reporting_module.dynamic()
        
