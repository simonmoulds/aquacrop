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

# from hm import file_handling
# from hm.Configuration import Configuration
# from hm.Messages import ModelConfigError, ModelConfigWarning
from hm.config import Configuration

import logging
logger = logging.getLogger(__name__)

valid_none_values = ['None', 'NONE', 'none', '']
valid_true_values = ['1', 'True', 'true', 'TRUE']
valid_false_values = ['0', 'False', 'false', 'FALSE'] + valid_none_values

def interpret_logical_string(x):
    if x in valid_true_values:
        return True
    elif x in valid_false_values:
        return False
    else:
        return None

def interpret_string(x):
    if x in valid_none_values:
        return None
    else:
        return x

def interpret_num_csv(x):
    return np.array([np.float64(val.strip()) for val in x.split(',')])

def interpret_str_csv(x):
    return [str(val.strip()) for val in x.split(',')]

class AquaCropConfiguration(Configuration):
    """Class for AquaCrop configuration options."""

    def set_config(self, system_arguments = None):
        super().set_config(
            system_arguments = system_arguments
        )
        self.set_model_grid_options()
        self.set_pseudo_coord_options()
        self.set_initial_condition_options()
        self.set_groundwater_options()
        self.set_soil_profile_options()
        self.set_soil_hydrology_options()
        self.set_soil_parameter_options()
        self.set_crop_parameter_options()
        self.set_farm_parameter_options()
        self.set_irrig_management_options()
        self.set_field_management_options()
        self.set_reporting_options()

    def set_pseudo_coord_options(self):
        if 'PSEUDO_COORDS' not in self.config_sections:
            self.PSEUDO_COORDS = {}
        else:
            for key, value in self.PSEUDO_COORDS.items():
                self.PSEUDO_COORDS[key] = interpret_num_csv(self.PSEUDO_COORDS[key])
            
    def check_config_file_for_required_entry(self, section_name, entry_name, allow_none=False, allow_empty=False):
        if entry_name not in list(vars(self)[section_name].keys()):
            raise KeyError(
                entry_name + ' in section ' + section_name + ' must be provided'
            )
        else:
            entry = vars(self)[section_name][entry_name]
            if not allow_none and (entry == "None"):
                raise ValueError(
                    entry_name + ' in section ' + section_name + 'cannot by none'
                )
            elif not allow_empty and (entry == ""):
                raise ValueError(
                    entry_name + ' in section ' + section_name + 'cannot by empty'
                )
            else:
                pass

    def set_model_grid_options(self):
        reqd_soil_hydrology_entries = [
            'mask'# ,
            # 'dzLayer'# ,
            # 'dzComp'
        ]
        for opt in reqd_soil_hydrology_entries:
            self.check_config_file_for_required_entry(
                'MODEL_GRID', opt
            )
        
        # try:
        #     self.MODEL_GRID['dzLayer'] = interpret_num_csv(self.MODEL_GRID['dzLayer'])
        # except:
        #     raise KeyError(
        #         'dzLayer in section MODEL_GRID could not be parsed'
        #     )
        
        # try:
        #     self.SOIL_HYDRAULIC_PARAMETERS['dzComp'] = interpret_num_csv(self.SOIL_HYDRAULIC_PARAMETERS['dzComp'])
        # except:
        #     raise KeyError(
        #         'dzComp in section MODEL_GRID could not be parsed'
        #     )        
        
    # def check_logical_entry(self, section_name, entry_name, allow_missing=False, allow_none=False, allow_empty=False):
    #     if not allow_missing:
    #         if entry_name not in list(vars(self)[section_name].keys()):
    #             raise ValueError(
    #                 entry_name + ' in section ' + section_name + ' must be provided'
    #             )
    #     else:
    #         entry = vars(self)[section_name][entry_name]
    #         if interpret_logical_string(entry) is None:
    #             raise KeyError(
    #                 entry_name +
    #                 ' in section ' +
    #                 section_name +
    #                 'is not interpretable as logical: must be one of ' +
    #                 msg_prefix
    #             )                
    #         # if allow_none and allow_empty:
    #         #     valid_entries = ["1","0","True","False","None",""]
    #         #     msg_prefix = "1, 0, True, False, None, or empty"
    #         # elif allow_none:
    #         #     valid_entries = ["1","0","True","False","None"]
    #         #     msg_prefix = "1, 0, True, False or None"
    #         # elif allow_empty:
    #         #     valid_entries = ["1","0","True","False",""]
    #         #     msg_prefix = "1, 0, True, False or empty"
    #         # else:
    #         #     valid_entries = ["1","0","True","False"]
    #         #     msg_prefix = "1, 0, True, or False"
    #         # if entry not in valid_entries:
    #         #     raise ModelConfigError(
    #         #         entry_name +
    #         #         ' in section ' +
    #         #         section_name +
    #         #         'is not interpretable as logical: must be one of ' +
    #         #         msg_prefix
    #         #     )                            
    def set_initial_condition_options(self):
        if 'initialConditionType' not in list(self.INITIAL_WATER_CONTENT.keys()):
            raise KeyError(
                'Config file must include initialConditionType '
                'in INITIAL_WATER_CONTENT section'
            )
        else:
            init_cond_type = self.INITIAL_WATER_CONTENT['initialConditionType'].upper()            
            if init_cond_type not in ['FILE','PERCENT','PROPERTY']:
                raise ValueError(
                    'initialConditionType must be one of '
                    'file, percent, property'
                )
            else:
                if init_cond_type == 'FILE':
                    if 'initialConditionNC' not in list(self.INITIAL_WATER_CONTENT.keys()):
                        raise KeyError(
                            'initialConditionType is FILE, but '
                            'initialConditionNC is not specified'
                        )
                    
                elif init_cond_type == 'PERCENT':
                    if 'initialConditionPercent' not in list(self.INITIAL_WATER_CONTENT.keys()):
                        raise KeyError(
                            'initialConditionType is PERCENT, '
                            'but initialConditionPercent is not '
                            'specified'
                        )
                    
                elif init_cond_type == 'PROPERTY':
                    if 'initialConditionProperty' not in list(self.INITIAL_WATER_CONTENT.keys()):
                        raise KeyError(
                            'initialConditionType is PROPERTY, '
                            'but initialConditionProperty is not '
                            'specified'
                        )
                    
                    else:
                        prop = self.INITIAL_WATER_CONTENT['initialConditionProperty'].upper()
                        if prop not in ['SAT','WP','FC']:
                            raise ValueError(
                                'initialConditionProperty must '
                                'be one of sat, wp, fc'
                            )
                        
            if init_cond_type == 'FILE':
                if 'initialConditionInterpMethod' not in list(self.INITIAL_WATER_CONTENT.keys()):
                    raise KeyError(
                        'if initialConditionProperty is FILE then '
                        'initialConditionInterpMethod must be '
                        'specified as one of DEPTH or LAYER'
                    )
                
            else:
                self.INITIAL_WATER_CONTENT['initialConditionInterpMethod'] = 'layer'

    def set_groundwater_options(self):
        if 'WaterTable' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['WaterTable'] = False
        else:
            self.WATER_TABLE['WaterTable'] = interpret_logical_string(self.WATER_TABLE['WaterTable'])
            
        if 'VariableWaterTable' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['VariableWaterTable'] = False
        else:
            self.WATER_TABLE['VariableWaterTable'] = interpret_logical_string(self.WATER_TABLE['VariableWaterTable'])
            
        if 'DailyGroundwaterNC' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['DailyGroundwaterNC'] = False
        else:
            self.WATER_TABLE['DailyGroundwaterNC'] = interpret_logical_string(self.WATER_TABLE['DailyGroundwaterNC'])
            
        if 'GroundwaterNC' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['GroundwaterNC'] = None
        else:
            self.WATER_TABLE['GroundwaterNC'] = interpret_string(self.WATER_TABLE['GroundwaterNC'])

    def set_soil_profile_options(self):
        self.check_config_file_for_required_entry('SOIL_PROFILE', 'dzComp')        
        try:
            self.SOIL_PROFILE['dzComp'] = interpret_num_csv(self.SOIL_PROFILE['dzComp'])
        except:
            raise KeyError(
                'dzComp in section MODEL_GRID could not be parsed'
            )
        
        self.check_config_file_for_required_entry('SOIL_PROFILE', 'dzLayer')        
        try:
            self.SOIL_PROFILE['dzLayer'] = interpret_num_csv(self.SOIL_PROFILE['dzLayer'])
        except:
            raise KeyError(
                'dzLayer in section MODEL_GRID could not be parsed'
            )        
        
    def set_soil_hydrology_options(self):
        if 'calculateFromSoilTexture' in list(self.SOIL_HYDRAULIC_PARAMETERS.keys()):
            calc = interpret_logical_string(self.SOIL_HYDRAULIC_PARAMETERS['calculateFromSoilTexture'])
            if calc:
                warnings.warn(
                    'The option to calculate soil hydraulic '
                    'parameters from soil textural characteristics '
                    'is not yet implemented'
                )
        reqd_soil_hydrology_entries = [
            'soilHydraulicParametersNC',
            'saturatedHydraulicConductivityVarName',
            'saturatedVolumetricWaterContentVarName',
            'fieldCapacityVolumetricWaterContentVarName',
            'wiltingPointVolumetricWaterContentVarName'# ,
            # 'dzSoilLayer',
            # 'dzSoilCompartment'
        ]
        for opt in reqd_soil_hydrology_entries:
            self.check_config_file_for_required_entry(
                'SOIL_HYDRAULIC_PARAMETERS',
                opt
            )
        
        # try:
        #     self.SOIL_HYDRAULIC_PARAMETERS['dzSoilLayer'] = interpret_num_csv(self.SOIL_HYDRAULIC_PARAMETERS['dzSoilLayer'])
        # except:
        #     raise KeyError(
        #         'dzSoilLayer in section SOIL_HYDRAULIC_PARAMETERS '
        #         'could not be parsed'
        #     )
        
        # try:
        #     self.SOIL_HYDRAULIC_PARAMETERS['dzSoilCompartment'] = interpret_num_csv(self.SOIL_HYDRAULIC_PARAMETERS['dzSoilCompartment'])
        # except:
        #     raise KeyError(
        #         'dzSoilCompartment in section SOIL_HYDRAULIC_PARAMETERS '
        #         'could not be parsed'
        #     )        
            
    def set_soil_parameter_options(self):
        if 'soilParametersNC' not in list(self.SOIL_PARAMETERS.keys()):
            self.SOIL_PARAMETERS['soilParametersNC'] = None
        else:
            if self.SOIL_PARAMETERS['soilParametersNC'] == "":
                self.SOIL_PARAMETERS['soilParametersNC'] = None
            else:
                pass
            
        if 'adjustReadilyAvailableWater' not in list(self.SOIL_PARAMETERS.keys()):
            warnings.warn(
                'adjustReadilyAvailableWater in section '
                'SOIL_PARAMETERS not provided: setting '
                'to 0 (False)'
            )            
            self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] = False
        else:
            self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] = interpret_logical_string(self.SOIL_PARAMETERS['adjustReadilyAvailableWater'])

        if 'adjustCurveNumber' not in list(self.SOIL_PARAMETERS.keys()):
            warnings.warn(
                'adjustCurveNumber in section SOIL_PARAMETERS '
                'not provided: setting to 1 (True)'
            )
            self.SOIL_PARAMETERS['adjustCurveNumber'] = False
        else:
            self.SOIL_PARAMETERS['adjustCurveNumber'] = interpret_logical_string(self.SOIL_PARAMETERS['adjustCurveNumber'])
        
    def set_crop_parameter_options(self):
        if 'cropID' not in list(self.CROP_PARAMETERS.keys()):
            # self.CROP_PARAMETERS['cropID'] = None
            self.CROP_PARAMETERS['cropID'] = [1]
        else:
            self.CROP_PARAMETERS['cropID'] = interpret_str_csv(self.CROP_PARAMETERS['cropID'])
            
        if 'nCrop' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['nCrop'] = 1
        else:
            self.CROP_PARAMETERS['nCrop'] = int(self.CROP_PARAMETERS['nCrop'])
            
        if 'nRotation' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['nRotation'] = 1
        else:
            self.CROP_PARAMETERS['nRotation'] = int(self.CROP_PARAMETERS['nRotation'])

        if 'landCoverFractionNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['landCoverFractionNC'] = None
        else:
            self.CROP_PARAMETERS['landCoverFractionNC'] = interpret_string(self.CROP_PARAMETERS['landCoverFractionNC'])
            
        if 'landCoverFractionVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['landCoverFractionVarName'] = None
        else:
            self.CROP_PARAMETERS['landCoverFractionVarName'] = interpret_string(self.CROP_PARAMETERS['landCoverFractionVarName'])
            
        if 'cropAreaNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropAreaNC'] = None
        else:
            self.CROP_PARAMETERS['cropAreaNC'] = interpret_string(self.CROP_PARAMETERS['cropAreaNC'])
            
        if 'cropAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropAreaVarName'] = None
        else:
            self.CROP_PARAMETERS['cropAreaVarName'] = interpret_string(self.CROP_PARAMETERS['cropAreaVarName'])
            
        if 'croplandAreaNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['croplandAreaNC'] = None
        else:
            self.CROP_PARAMETERS['croplandAreaNC'] = interpret_string(self.CROP_PARAMETERS['croplandAreaNC'])
            
        if 'croplandAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['croplandAreaVarName'] = None
        else:
            self.CROP_PARAMETERS['croplandAreaVarName'] = interpret_string(self.CROP_PARAMETERS['croplandAreaVarName'])
            
        if 'AnnualChangeInCropArea' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['AnnualChangeInCropArea'] = False
        else:
            self.CROP_PARAMETERS['AnnualChangeInCropArea'] = interpret_logical_string(self.CROP_PARAMETERS['AnnualChangeInCropArea'])

        if 'CalendarType' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['CalendarType'] = 1
        else:
            self.CROP_PARAMETERS['CalendarType'] = int(self.CROP_PARAMETERS['CalendarType'])

        if 'SwitchGDD' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['SwitchGDD'] = False
        else:
            self.CROP_PARAMETERS['SwitchGDD'] = interpret_logical_string(self.CROP_PARAMETERS['SwitchGDD'])

        if 'GDDmethod' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['GDDmethod'] = 1
        else:
            self.CROP_PARAMETERS['GDDmethod'] = int(self.CROP_PARAMETERS['GDDmethod'])
            
    def set_farm_parameter_options(self):
        if 'FARM_PARAMETERS' not in self.config_sections:
            self.FARM_PARAMETERS = {}

        if 'nFarm' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['nFarm'] = 1
        else:
            self.FARM_PARAMETERS['nFarm'] = int(self.FARM_PARAMETERS['nFarm'])

        if 'farmAreaNC' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaNC'] = None
        else:
            self.FARM_PARAMETERS['farmAreaNC'] = interpret_string(self.FARM_PARAMETERS['farmAreaNC'])

        if 'farmAreaVarName' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaVarName'] = None
        else:
            self.FARM_PARAMETERS['farmAreaVarName'] = interpret_string(self.FARM_PARAMETERS['farmAreaVarName'])
            
        if 'AnnualChangeInFarmArea' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['AnnualChangeInFarmArea'] = False
        else:
            self.FARM_PARAMETERS['AnnualChangeInFarmArea'] = interpret_logical_string(self.FARM_PARAMETERS['AnnualChangeInFarmArea'])

    def set_irrig_management_options(self):
        pass

    def set_field_management_options(self):
        if 'fieldManagementNC' not in list(self.FIELD_MANAGEMENT.keys()):
            self.FIELD_MANAGEMENT['fieldManagementNC'] = None
        else:
            self.FIELD_MANAGEMENT['fieldManagementNC'] = interpret_string(self.FIELD_MANAGEMENT['fieldManagementNC'])

    def set_reporting_options(self):
        if 'REPORTING' not in self.config_sections:
            self.REPORTING = {}

        if 'report' not in list(self.REPORTING.keys()):
            self.REPORTING['report'] = False
    
