#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm import file_handling
from hm.Reporting import Reporting

from .io.LandCoverParameters import LandCoverParameters, AquaCropParameters
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
    def __init__(self, var, config_section_name):
        self.var = var
        self._configuration = var._configuration
        self._modelTime = var._modelTime
        self.cloneMapFileName = var.cloneMapFileName
        self.cloneMap = var.cloneMap
        self.landmask = var.landmask
        self.grid_cell_area = var.grid_cell_area
        self.dimensions = var.dimensions
        self.nLat, self.nLon, self.nCell = var.nLat, var.nLon, var.nCell
        self.nFarm, self.nCrop = 1, 1
        
        # attach weatherrological and groundwater data to land cover object
        self.weather = var.weather_module
        self.groundwater = var.groundwater_module
        # self.canal = var.canal_module

    def initial(self):
        pass
    
    def add_dimensions(self):
        """Function to add dimensions to model dimensions 
        object. This is necessary if the LandCover object 
        contains a reporting method."""
        self.dimensions['farm'] = np.arange(self.nFarm)
        self.dimensions['crop'] = np.arange(self.nCrop)

    def dynamic(self):
        pass

class Cropland(LandCover):
    def __init__(self, var, config_section_name):
        super(Cropland, self).__init__(
            var,
            config_section_name)
        
        self.carbon_dioxide_module = CarbonDioxide(self)
        self.lc_parameters_module = AquaCropParameters(self, config_section_name)
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

        # plan is to merge these five classes:
        self.HI_ref_current_day_module = HarvestIndex(self)
        # self.HI_ref_current_day_module = HIrefCurrentDay(self)
        self.biomass_accumulation_module = BiomassAccumulation(self)
        self.temperature_stress_module = TemperatureStress(self)
        self.harvest_index_module = HarvestIndexAdjusted(self)
        self.crop_yield_module = CropYield(self)
        
#         self.grid_cell_mean_module = GridCellMean(self)
        self.add_dimensions()
        
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
        self.reporting_module = Reporting(
            self,
            self._configuration.outNCDir,
            self._configuration.NETCDF_ATTRIBUTES,
            self._configuration.CROP_PARAMETERS,  # TODO: shouldn't have to specify this
            variable_list_crop,
            'cropland')
        
    def dynamic(self):
        self.carbon_dioxide_module.dynamic()
        self.lc_parameters_module.dynamic()
        self.gdd_module.dynamic()
        self.growth_stage_module.dynamic()
        self.initial_condition_module.dynamic()
        self.check_groundwater_table_module.dynamic()
        self.pre_irrigation_module.dynamic()
        self.drainage_module.dynamic()
        self.rainfall_partition_module.dynamic()
        self.root_zone_water_module.dynamic()
        self.irrigation_module.dynamic()
        self.infiltration_module.dynamic()
        self.capillary_rise_module.dynamic()
        self.germination_module.dynamic()
        # self.growth_stage_module.dynamic() # TODO: compare against AquaCropOS - don't think this is needed
        self.root_development_module.dynamic()
        self.root_zone_water_module.dynamic()
        self.water_stress_module.dynamic(beta=True)
        self.canopy_cover_module.dynamic()
        self.soil_evaporation_module.dynamic()
        self.root_zone_water_module.dynamic()
        self.water_stress_module.dynamic(beta=True)
        self.transpiration_module.dynamic()
        self.evapotranspiration_module.dynamic()
        self.inflow_module.dynamic()
        self.HI_ref_current_day_module.dynamic()
        self.temperature_stress_module.dynamic()  # PREVIOUSLY THIS WAS CALLED FROM BIOMASS_ACCUMULATION
        self.biomass_accumulation_module.dynamic()
        self.root_zone_water_module.dynamic()
        self.water_stress_module.dynamic(beta=True)
        self.temperature_stress_module.dynamic()
        self.harvest_index_module.dynamic()
        self.crop_yield_module.dynamic()
        self.root_zone_water_module.dynamic()
#         self.grid_cell_mean_module.dynamic()
        self.reporting_module.report()
    
# class NaturalVegetation(LandCover):
#     def __init__(self, var, config_section_name):
#         super(NaturalVegetation, self).__init__(
#             var,
#             config_section_name)

#         self.lc_parameters_module = NaturalVegetationParameters(self, config_section_name)
#         self.initial_condition_module = InitialConditionNaturalVegetation(self)
#         self.root_zone_water_module = RootZoneWaterNaturalVegetation(self)
#         self.snow_frost_module = SnowFrost(self)
#         self.evapotranspiration_module = Evapotranspiration(self)
#         self.interception_module = Interception(self)
#         self.infiltration_module = Infiltration(self)
#         self.capillary_rise_module = CapillaryRise(self)
#         self.drainage_module = Drainage(self)

#     def initial(self):
#         self.lc_parameters_module.initial()
#         self.initial_condition_module.initial()
#         self.root_zone_water_module.initial()
#         self.snow_frost_module.initial()
#         self.evapotranspiration_module.initial()
#         self.interception_module.initial()
#         self.infiltration_module.initial()
#         self.capillary_rise_module.initial()
#         self.drainage_module.initial()
        
#     def dynamic(self):
#         self.lc_parameters_module.dynamic()
#         self.initial_condition_module.dynamic()
#         self.root_zone_water_module.dynamic()
#         self.snow_frost_module.dynamic()
#         self.evapotranspiration_module.dynamic()
#         self.interception_module.dynamic()
#         self.infiltration_module.dynamic()
#         self.capillary_rise_module.dynamic()
#         self.drainage_module.dynamic()

# class ManagedLand(LandCover):
#     def __init__(self, var, config_section_name):
#         super(ManagedLand, self).__init__(
#             var,
#             config_section_name)

#         self.lc_parameters_module = ManagedLandParameters(self, config_section_name)
#         self.initial_condition_module = InitialConditionManagedLand(self)
#         self.root_zone_water_module = RootZoneWater(self)
#         self.snow_frost_module = SnowFrost(self)
#         self.evapotranspiration_module = Evapotranspiration(self)
#         self.interception_module = Interception(self)
#         self.infiltration_module = Infiltration(self)
#         self.capillary_rise_module = CapillaryRise(self)
#         self.drainage_module = Drainage(self)
        
#     def initial(self):
#         self.lc_parameters_module.initial()
#         self.initial_condition_module.initial()
#         self.root_zone_water_module.initial()
#         self.snow_frost_module.initial()
#         self.evapotranspiration_module.initial()        
#         self.interception_module.initial()
#         self.irrigation_module.initial()
#         self.infiltration_module.initial()
#         self.capillary_rise_module.initial()
#         self.drainage_module.initial()
        
#     def dynamic(self):
#         self.lc_parameters_module.dynamic()
#         self.initial_condition_module.dynamic()
#         self.root_zone_water_module.dynamic()
#         self.snow_frost_module.dynamic()
#         self.evapotranspiration_module.dynamic()
#         self.interception_module.dynamic()        
#         self.irrigation_module.dynamic()
#         self.infiltration_module.dynamic()
#         self.capillary_rise_module.dynamic()
#         self.drainage_module.dynamic()

# class ManagedLandIrrPaddy(ManagedLand):
#     def __init__(self, var, config_section_name):
#         super(ManagedLandIrrPaddy, self).__init__(
#             var,
#             config_section_name)
#         self.irrigation_module = IrrigationPaddy(self)
#         self.root_zone_water_module = RootZoneWaterIrrigatedLand(self)

# class ManagedLandIrrNonPaddy(ManagedLand):
#     def __init__(self, var, config_section_name):
#         super(ManagedLandIrrNonPaddy, self).__init__(
#             var,
#             config_section_name)
#         self.irrigation_module = IrrigationNonPaddy(self)
#         self.root_zone_water_module = RootZoneWaterIrrigatedLand(self)

# class ManagedLandWithFarmerBehaviour(LandCover):
#     def __init__(self, var, config_section_name):
#         super(ManagedLandWithFarmerBehaviour, self).__init__(
#             var,
#             config_section_name)
        
#         self.lc_parameters_module = ManagedLandWithFarmerBehaviourParameters(self, config_section_name)
#         self.initial_condition_module = InitialConditionManagedLand(self)
#         self.root_zone_water_module = RootZoneWaterIrrigatedLand(self)
#         self.snow_frost_module = SnowFrost(self)
#         self.potential_et_module = PotentialEvapotranspiration(self)
#         self.interception_module = Interception(self)
#         self.irrigation_demand_module = IrrigationMultipleCrops(self)
#         self.irrigation_supply_module = IrrigationSupply(self)
#         self.infiltration_module = Infiltration(self)
#         self.capillary_rise_module = CapillaryRise(self)
#         self.drainage_module = Drainage(self)
#         self.actual_et_module = ActualEvapotranspiration(self)        
#         self.crop_yield_module = CropYield(self)
#         self.income_module = Income(self)
#         self.accounting_module = Accounting(self)
#         self.investment_module = Investment(self)
#         self.grid_cell_mean_module = GridCellMean(self)
#         self.add_dimensions()
        
#     def initial(self):
#         self.lc_parameters_module.initial()
#         self.initial_condition_module.initial()
#         self.root_zone_water_module.initial()
#         self.snow_frost_module.initial()
#         self.potential_et_module.initial()        
#         self.interception_module.initial()
#         self.irrigation_demand_module.initial()
#         self.irrigation_supply_module.initial()
#         self.infiltration_module.initial()
#         self.capillary_rise_module.initial()
#         self.drainage_module.initial()
#         self.actual_et_module.initial()        
#         self.crop_yield_module.initial()
#         self.income_module.initial()
#         self.accounting_module.initial()
#         self.investment_module.initial()
#         self.grid_cell_mean_module.initial()
#         self.reporting_module = Reporting(
#             self,
#             self._configuration.outNCDir,
#             self._configuration.NETCDF_ATTRIBUTES,
#             self._configuration.irrNonPaddy,
#             variable_list_crop,
#             'irrNonPaddy')
        
#     def dynamic(self):
#         self.lc_parameters_module.dynamic()
#         self.initial_condition_module.dynamic()
        
#         self.root_zone_water_module.dynamic()
#         self.snow_frost_module.dynamic()
#         self.potential_et_module.dynamic()
#         self.interception_module.dynamic()
#         self.root_zone_water_module.dynamic()
#         self.infiltration_module.compute_infiltration_capacity()        
#         self.irrigation_demand_module.dynamic()        
#         self.irrigation_supply_module.dynamic()

#         # the order here (infiltration/cap rise/drainage)
#         # is the same as in CWATM
#         self.infiltration_module.dynamic()
#         self.root_zone_water_module.dynamic()
#         self.capillary_rise_module.dynamic()
#         self.drainage_module.dynamic()
        
#         self.root_zone_water_module.dynamic()
#         self.actual_et_module.dynamic()
#         self.crop_yield_module.dynamic()
#         self.income_module.dynamic()
#         self.accounting_module.dynamic()
#         self.investment_module.dynamic()
#         self.grid_cell_mean_module.dynamic()
#         self.reporting_module.report()
        
# class SealedLand(LandCover):
#     def __init__(self, var, config_section_name):
#         super(SealedLand, self).__init__(
#             var,
#             config_section_name)
        
#         self.lc_parameters_module = SealedLandParameters(self, config_section_name)
#         self.initial_condition_module = InitialConditionSealedLand(self)
#         self.snow_frost_module = SnowFrost(self)
#         self.interception_module = InterceptionSealed(self)
#         self.evapotranspiration_module = EvaporationSealed(self)
#         self.infiltration_module = InfiltrationSealed(self)
#         self.capillary_rise_module = CapillaryRise(self)
#         self.drainage_module = Drainage(self)

#     def initial(self):
#         self.lc_parameters_module.initial()
#         self.initial_condition_module.initial()
#         self.snow_frost_module.initial()
#         self.interception_module.initial()
#         self.evapotranspiration_module.initial()
#         self.infiltration_module.initial()
#         self.capillary_rise_module.initial()
#         self.drainage_module.initial()
        
#     def dynamic(self):
#         self.lc_parameters_module.dynamic()
#         self.initial_condition_module.dynamic()
#         self.snow_frost_module.dynamic()
#         self.interception_module.dynamic()
#         self.evapotranspiration_module.dynamic()
#         self.infiltration_module.dynamic()

# class OpenWater(LandCover):
#     def __init__(self, var, config_section_name):
#         super(OpenWater, self).__init__(
#             var,
#             config_section_name)        
#         self.lc_parameters_module = WaterParameters(self, config_section_name)
#         self.initial_condition_module = InitialConditionSealedLand(self)
#         self.snow_frost_module = SnowFrost(self)
#         self.interception_module = InterceptionWater(self)
#         self.evapotranspiration_module = EvaporationWater(self)
#         self.infiltration_module = InfiltrationWater(self)
#         # include these modules so that some variables are initialized to zero
#         self.capillary_rise_module = CapillaryRise(self)  
#         self.drainage_module = Drainage(self)

#     def initial(self):
#         self.lc_parameters_module.initial()
#         self.initial_condition_module.initial()
#         self.snow_frost_module.initial()
#         self.interception_module.initial()
#         self.evapotranspiration_module.initial()
#         self.infiltration_module.initial()
#         self.capillary_rise_module.initial()
#         self.drainage_module.initial()
        
#     def dynamic(self):
#         self.lc_parameters_module.dynamic()
#         self.initial_condition_module.dynamic()
#         self.snow_frost_module.dynamic()
#         self.interception_module.dynamic()
#         self.evapotranspiration_module.dynamic()
#         self.infiltration_module.dynamic()
        
# class Forest(NaturalVegetation):
#     """Class to represent forest land cover"""
        
# class Grassland(NaturalVegetation):
#     """Class to represent grassland"""

# class IrrPaddy(ManagedLandIrrPaddy):
#     """Class to represent irrigated paddy"""

# class IrrNonPaddy(ManagedLandWithFarmerBehaviour):
#     """Class to represent irrigated non-paddy"""
# # class IrrNonPaddy(ManagedLandIrrNonPaddy):
# #     """Class to represent irrigated non-paddy"""
    
# class Sealed(SealedLand):
#     """Class to represent sealed area"""

# class Water(OpenWater):
#     """Class to represent water"""
