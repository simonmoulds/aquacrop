#!/usr/bin/env python
# -*- coding: utf-8 -*-

from configparser import ConfigParser, ExtendedInterpolation
import os
import sys
import numpy as np
import string
import time
import datetime
import shutil
import glob
import warnings
import errno

from hm import file_handling
from hm.Configuration import Configuration
from hm.Messages import ModelConfigError, ModelConfigWarning

import logging
logger = logging.getLogger(__name__)

def interpret_logical_string(x):
    if (x == "1") or (x == "True"):
        return True
    elif (x == "0") or (x == "False"):
        return False
    else:
        return None
    
class AquaCropConfiguration(Configuration):
    """Class for AquaCrop configuration options."""
    
    def repair_ini_key_names(self):
        """Change or modify configuration options.

        This function checks the validity of user-supplied 
        configuration options, ensures that all required 
        options are supplied, and infills missing keys with 
        sensible defaults where possible.
        """
        self.repair_initial_condition_config()
        self.repair_soil_hydraulic_parameters()
        self.repair_soil_parameters()
        self.repair_crop_parameters()
        self.repair_farm_parameters()
        self.repair_irrigation_management_parameters()
        self.repair_field_management_parameters()

    def check_config_file_for_required_entry(self, section_name, entry_name, allow_none=False, allow_empty=False):
        if entry_name not in list(vars(self)[section_name].keys()):
            raise ModelConfigError(entry_name + ' in section ' + section_name + ' must be provided')
        else:
            entry = vars(self)[section_name][entry_name]
            if not allow_none and (entry == "None"):
                raise ModelConfigError(entry_name + ' in section ' + section_name + 'cannot by none')
            elif not allow_empty and (entry == ""):
                raise ModelConfigError(entry_name + ' in section ' + section_name + 'cannot by empty')
            else:
                pass

    def check_logical_input(self, section_name, entry_name, allow_missing=False, allow_none=False, allow_empty=False):
        if not allow_missing:
            if entry_name not in list(vars(self)[section_name].keys()):
                raise ModelConfigError(entry_name + ' in section ' + section_name + ' must be provided')
        else:
            entry = vars(self)[section_name][entry_name]
            if allow_none and allow_empty:
                valid_entries = ["1","0","True","False","None",""]
                msg_prefix = "1, 0, True, False, None, or empty"
            elif allow_none:
                valid_entries = ["1","0","True","False","None"]
                msg_prefix = "1, 0, True, False or None"
            elif allow_empty:
                valid_entries = ["1","0","True","False",""]
                msg_prefix = "1, 0, True, False or empty"
            else:
                valid_entries = ["1","0","True","False"]
                msg_prefix = "1, 0, True, or False"
            if entry not in valid_entries:
                raise ModelConfigError(
                    entry_name +
                    ' in section ' +
                    section_name +
                    'is not interpretable as logical: must be one of ' +
                    msg_prefix
                )                
            
    def repair_initial_condition_config(self):
        if 'initialConditionType' not in list(self.INITIAL_WATER_CONTENT.keys()):
            raise ModelConfigError('Config file must include initialConditionType in INITIAL_WATER_CONTENT section')
        else:
            init_cond_type = self.INITIAL_WATER_CONTENT['initialConditionType'].upper()            
            if init_cond_type not in ['FILE','PERCENT','PROPERTY']:
                raise ModelConfigError('initialConditionType must be one of file, percent, property')
            else:
                if init_cond_type == 'FILE':
                    if 'initialConditionNC' not in list(self.INITIAL_WATER_CONTENT.keys()):
                        raise ModelConfigError('initialConditionType is FILE, but initialConditionNC is not specified')
                    # TODO: check file exists in the file system
                elif init_cond_type == 'PERCENT':
                    if 'initialConditionPercent' not in list(self.INITIAL_WATER_CONTENT.keys()):
                        raise ModelConfigError('initialConditionType is PERCENT, but initialConditionPercent is not specified')
                elif init_cond_type == 'PROPERTY':
                    if 'initialConditionProperty' not in list(self.INITIAL_WATER_CONTENT.keys()):
                        raise ModelConfigError('initialConditionType is PROPERTY, but initialConditionProperty is not specified')
                    else:
                        soil_hydraulic_property = self.INITIAL_WATER_CONTENT['initialConditionProperty'].upper()
                        if soil_hydraulic_property not in ['SAT','WP','FC']:
                            raise ModelConfigError('initialConditionProperty must be one of sat, wp, fc')
            if init_cond_type == 'FILE':
                if 'initialConditionInterpMethod' not in list(self.INITIAL_WATER_CONTENT.keys()):
                    raise ModelConfigError('if initialConditionProperty is FILE then initialConditionInterpMethod must be specified as one of DEPTH or LAYER')
            else:
                self.INITIAL_WATER_CONTENT['initialConditionInterpMethod'] = 'layer'

    def repair_soil_hydraulic_parameters_nc(self):
        if 'soilHydraulicParametersNC' not in list(self.SOIL_HYDRAULIC_PARAMETERS.keys()):
            raise ModelConfigError('soilHydraulicParametersNC is not specified')
        else:
            soil_param_nc = self.SOIL_HYDRAULIC_PARAMETERS['soilHydraulicParametersNC']
            if soil_param_nc == "":
                raise ModelConfigError('soilHydraulicParametersNC cannot by empty')
            elif soil_param_nc == "None":                
                raise ModelConfigError('soilHydraulicParametersNC cannot by None')
            else:
                # file_exists = os.path.isfile(soil_param_nc)
                # if not file_exists:
                #     raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), soil_param_nc)
                pass        
                
    def repair_soil_hydraulic_parameters(self):
        if 'calculateSoilHydraulicParametersFromSoilTexture' in list(self.SOIL_HYDRAULIC_PARAMETERS.keys()):
            calc_shp = self.SOIL_HYDRAULIC_PARAMETERS['calculateSoilHydraulicParametersFromSoilTexture']
            if calc_shp != "0":
                warnings.warn('The option to calculate soil hydraulic parameters from soil textural characteristics is not yet implemented', ModelConfigWarning)
                
        self.repair_soil_hydraulic_parameters_nc()
        self.check_config_file_for_required_entry(
            "SOIL_HYDRAULIC_PARAMETERS",
            "saturatedHydraulicConductivityVarName"
        )
        self.check_config_file_for_required_entry(
            "SOIL_HYDRAULIC_PARAMETERS",
            "saturatedVolumetricWaterContentVarName"
        )
        self.check_config_file_for_required_entry(
            "SOIL_HYDRAULIC_PARAMETERS",
            "fieldCapacityVolumetricWaterContentVarName"
        )
        self.check_config_file_for_required_entry(
            "SOIL_HYDRAULIC_PARAMETERS",
            "wiltingPointVolumetricWaterContentVarName"
        )
        self.check_config_file_for_required_entry(
            "SOIL_HYDRAULIC_PARAMETERS",
            "dzSoilLayer"
        )
        self.check_config_file_for_required_entry(
            "SOIL_HYDRAULIC_PARAMETERS",
            "dzSoilCompartment"
        )
        dzSoilLayer = self.SOIL_HYDRAULIC_PARAMETERS['dzSoilLayer']
        try:
            dzSoilLayer = np.array([np.float64(x.strip()) for x in dzSoilLayer.split(",")])
            self.SOIL_HYDRAULIC_PARAMETERS['dzSoilLayer'] = dzSoilLayer
        except:
            raise ModelConfigError('dzSoilLayer in section SOIL_HYDRAULIC_PARAMETERS could not be parsed')
        dzSoilCompartment = self.SOIL_HYDRAULIC_PARAMETERS['dzSoilCompartment']
        try:
            dzSoilCompartment = np.array([np.float64(x.strip()) for x in dzSoilCompartment.split(",")])
            self.SOIL_HYDRAULIC_PARAMETERS['dzSoilCompartment'] = dzSoilCompartment
        except:
            raise ModelConfigError('dzSoilCompartment in section SOIL_HYDRAULIC_PARAMETERS could not be parsed')        
        
    def repair_soil_parameters_nc(self):
        if 'soilParametersNC' not in list(self.SOIL_PARAMETERS.keys()):
            self.SOIL_PARAMETERS['soilParametersNC'] = "None"
        else:
            soil_param_nc = self.SOIL_PARAMETERS['soilParametersNC']
            if soil_param_nc == "":
                self.SOIL_PARAMETERS['soilParametersNC'] = "None"
            else:
                # file_exists = os.path.isfile(soil_param_nc)
                # if not file_exists:
                #     raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), soil_param_nc)
                pass
            
        if 'adjustReadilyAvailableWater' not in list(self.SOIL_PARAMETERS.keys()):
            warnings.warn('adjustReadilyAvailableWater in section SOIL_PARAMETERS not provided: setting to 0 (False)', ModelConfigWarning)
            self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] == "0"
        else:
            self.check_logical_input('SOIL_PARAMETERS','adjustReadilyAvailableWater')

        if 'adjustCurveNumber' not in list(self.SOIL_PARAMETERS.keys()):
            warnings.warn('adjustCurveNumber in section SOIL_PARAMETERS not provided: setting to 1 (True)', ModelConfigError)
        else:
            self.check_logical_input('SOIL_PARAMETERS','adjustCurveNumber')
            
    def repair_soil_parameters(self):
        self.repair_soil_parameters_nc()        
        
    def repair_crop_parameters(self):
        if 'cropID' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropID'] = "None"
            
        if 'nCrop' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['nCrop'] = 1
            
        if 'nRotation' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['nRotation'] = 1

        if 'landCoverFractionNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['landCoverFractionNC'] = "None"
            
        if 'landCoverFractionVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['landCoverFractionVarName'] = "None"
            
        if 'cropAreaNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropAreaNC'] = "None"
            
        if 'cropAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropAreaVarName'] = "None"
            
        if 'croplandAreaNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['croplandAreaNC'] = "None"
            
        if 'croplandAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['croplandAreaVarName'] = "None"
            
        if 'AnnualChangeInCropArea' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['AnnualChangeInCropArea'] = 0
            
    def repair_farm_parameters(self):
        if 'FARM_PARAMETERS' not in self.allSections:
            self.FARM_PARAMETERS = {}

        if 'nFarm' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['nFarm'] = 1

        if 'farmAreaNC' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaNC'] = "None"

        if 'farmAreaVarName' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaVarName'] = "None"
            
        if 'AnnualChangeInFarmArea' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['AnnualChangeInFarmArea'] = 0

    def repair_irrigation_management_parameters(self):
        # TODO: make irrMgmtParameterFileNC = None - CHECK NAME IN CONFIG FILE
        pass
    
    def repair_field_management_parameters(self):
        pass
