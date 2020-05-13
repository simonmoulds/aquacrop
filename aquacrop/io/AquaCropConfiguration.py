#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import warnings

from hm.config import Configuration

import logging
logger = logging.getLogger(__name__)

VALID_NONE_VALUES = ['None', 'NONE', 'none', '']

DEFAULT_INPUT_FILE_OPTS = {
    'filename': None,
    'varname': None,
    'is_1d': False,
    'xy_dimname': None
}


class AquaCropConfiguration(Configuration):
    """Class for AquaCrop configuration options."""

    def set_config(self, system_arguments=None):
        super().set_config(
            system_arguments=system_arguments
        )
        self.set_model_grid_options()
        self.set_weather_options()
        self.set_carbon_dioxide_options()
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

    def check_config_file_for_required_entry(
            self,
            section_name,
            entry_name,
            allow_none=False,
            allow_empty=False
    ):
        if entry_name not in list(vars(self)[section_name].keys()):
            raise KeyError(
                entry_name
                + ' in section '
                + section_name
                + ' must be provided'
            )
        else:
            entry = vars(self)[section_name][entry_name]
            if not allow_none and (entry in VALID_NONE_VALUES):
                raise ValueError(
                    entry_name
                    + ' in section '
                    + section_name
                    + 'cannot by empty or none'
                )

            else:
                pass

    def set_model_grid_options(self):
        pass

    def set_weather_options(self):
        weather_sections = ['PRECIPITATION', 'TMIN', 'TMAX', 'TAVG', 'ETREF']
        for section in weather_sections:
            for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
                if opt not in vars(self)[section].keys():
                    vars(self)[section][opt] = default_value

    def set_carbon_dioxide_options(self):
        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.CARBON_DIOXIDE.keys():
                self.CARBON_DIOXIDE[opt] = default_value

    def set_initial_condition_options(self):
        self.check_config_file_for_required_entry(
            'INITIAL_WATER_CONTENT',
            'initialConditionType'
        )
        init_type = self.INITIAL_WATER_CONTENT['initialConditionType'].upper()
        if init_type == 'FILE':
            if 'initialConditionNC' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'initialConditionType is FILE, but '
                    'initialConditionNC is not specified'
                )

            elif 'initialConditionInterpMethod' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'if initialConditionProperty is FILE then '
                    'initialConditionInterpMethod must be '
                    'specified as one of DEPTH or LAYER'
                )

        elif init_type == 'PERCENT':
            if 'initialConditionPercent' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'initialConditionType is PERCENT, '
                    'but initialConditionPercent is not '
                    'specified'
                )

        elif init_type == 'PROPERTY':
            if 'initialConditionProperty' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'initialConditionType is PROPERTY, '
                    'but initialConditionProperty is not '
                    'specified'
                )

            else:
                prop = self.INITIAL_WATER_CONTENT['initialConditionProperty']\
                           .upper()
                if prop not in ['SAT', 'WP', 'FC']:
                    raise ValueError(
                        'initialConditionProperty must '
                        'be one of sat, wp, fc'
                    )

        else:
            raise ValueError(
                'initialConditionType must be one of '
                'file, percent, property'
            )

    def set_groundwater_options(self):
        if 'WaterTable' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['WaterTable'] = False

        if 'VariableWaterTable' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['VariableWaterTable'] = False

        if 'DailyGroundwaterNC' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['DailyGroundwaterNC'] = False

        if 'GroundwaterNC' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['GroundwaterNC'] = None

    def set_soil_profile_options(self):
        pass

    def set_soil_hydrology_options(self):
        reqd_soil_hydrology_entries = [
            # 'soilHydraulicParametersNC',
            'saturatedHydraulicConductivityVarName',
            'saturatedVolumetricWaterContentVarName',
            'fieldCapacityVolumetricWaterContentVarName',
            'wiltingPointVolumetricWaterContentVarName'  # ,
            # 'dzSoilLayer',
            # 'dzSoilCompartment'
        ]
        for opt in reqd_soil_hydrology_entries:
            self.check_config_file_for_required_entry(
                'SOIL_HYDRAULIC_PARAMETERS',
                opt
            )

        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.SOIL_HYDRAULIC_PARAMETERS.keys():
                self.SOIL_HYDRAULIC_PARAMETERS[opt] = default_value

    def set_soil_parameter_options(self):
        # self.check_config_file_for_required_entry(
        #     'SOIL_PARAMETERS',
        #     'soilParametersNC'
        # )
        # if 'soilParametersNC' not in list(self.SOIL_PARAMETERS.keys()):
        #     self.SOIL_PARAMETERS['soilParametersNC'] = None
        # # else:
        # #     self.SOIL_PARAMETERS['soilParametersNC'] = interpret_string(
        # #         self.SOIL_PARAMETERS['soilParametersNC']
        # #     )
        if 'soilParametersNC' not in list(self.SOIL_PARAMETERS.keys()):
            self.SOIL_PARAMETERS['soilParametersNC'] = ""
        if self.SOIL_PARAMETERS['soilParametersNC'] in VALID_NONE_VALUES:
            self.SOIL_PARAMETERS['soilParametersNC'] = None

        if 'adjustReadilyAvailableWater' not in list(self.SOIL_PARAMETERS.keys()):
            # warnings.warn(
            #     'adjustReadilyAvailableWater in section '
            #     'SOIL_PARAMETERS not provided: setting '
            #     'to 0 (False)'
            # )
            self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] = False
        # else:
        #     self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] = interpret_logical_string(
        #         self.SOIL_PARAMETERS['adjustReadilyAvailableWater'])

        if 'adjustCurveNumber' not in list(self.SOIL_PARAMETERS.keys()):
            # warnings.warn(
            #     'adjustCurveNumber in section SOIL_PARAMETERS '
            #     'not provided: setting to 1 (True)'
            # )
            self.SOIL_PARAMETERS['adjustCurveNumber'] = False
        # else:
        #     self.SOIL_PARAMETERS['adjustCurveNumber'] = interpret_logical_string(
        #         self.SOIL_PARAMETERS['adjustCurveNumber'])

    def set_crop_parameter_options(self):

        if 'cropID' not in list(self.CROP_PARAMETERS.keys()):
            # self.CROP_PARAMETERS['cropID'] = None
            self.CROP_PARAMETERS['cropID'] = [1]
        # else:
        #     self.CROP_PARAMETERS['cropID'] = interpret_str_csv(
        #         self.CROP_PARAMETERS['cropID'])

        if 'nCrop' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['nCrop'] = 1
        # else:
        #     self.CROP_PARAMETERS['nCrop'] = int(self.CROP_PARAMETERS['nCrop'])

        if 'nRotation' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['nRotation'] = 1
        # else:
        #     self.CROP_PARAMETERS['nRotation'] = int(
        #         self.CROP_PARAMETERS['nRotation'])

        if 'landCoverFractionNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['landCoverFractionNC'] = None
        # else:
        #     self.CROP_PARAMETERS['landCoverFractionNC'] = interpret_string(
        #         self.CROP_PARAMETERS['landCoverFractionNC'])

        if 'landCoverFractionVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['landCoverFractionVarName'] = None
        # else:
        #     self.CROP_PARAMETERS['landCoverFractionVarName'] = interpret_string(
        #         self.CROP_PARAMETERS['landCoverFractionVarName'])

        if 'cropAreaNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropAreaNC'] = None
        # else:
        #     self.CROP_PARAMETERS['cropAreaNC'] = interpret_string(
        #         self.CROP_PARAMETERS['cropAreaNC'])

        if 'cropAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['cropAreaVarName'] = None
        # else:
        #     self.CROP_PARAMETERS['cropAreaVarName'] = interpret_string(
        #         self.CROP_PARAMETERS['cropAreaVarName'])

        if 'croplandAreaNC' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['croplandAreaNC'] = None
        # else:
        #     self.CROP_PARAMETERS['croplandAreaNC'] = interpret_string(
        #         self.CROP_PARAMETERS['croplandAreaNC'])

        if 'croplandAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['croplandAreaVarName'] = None
        # else:
        #     self.CROP_PARAMETERS['croplandAreaVarName'] = interpret_string(
        #         self.CROP_PARAMETERS['croplandAreaVarName'])

        if 'AnnualChangeInCropArea' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['AnnualChangeInCropArea'] = False
        # else:
        #     self.CROP_PARAMETERS['AnnualChangeInCropArea'] = interpret_logical_string(
        #         self.CROP_PARAMETERS['AnnualChangeInCropArea'])

        if 'CalendarType' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['CalendarType'] = 1
        # else:
        #     self.CROP_PARAMETERS['CalendarType'] = int(
        #         self.CROP_PARAMETERS['CalendarType'])

        if 'SwitchGDD' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['SwitchGDD'] = False
        # else:
        #     self.CROP_PARAMETERS['SwitchGDD'] = interpret_logical_string(
        #         self.CROP_PARAMETERS['SwitchGDD'])

        if 'GDDmethod' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['GDDmethod'] = 1
        # else:
        #     self.CROP_PARAMETERS['GDDmethod'] = int(
        #         self.CROP_PARAMETERS['GDDmethod'])

    def set_farm_parameter_options(self):
        if 'FARM_PARAMETERS' not in self.config_sections:
            self.FARM_PARAMETERS = {}

        if 'nFarm' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['nFarm'] = 1
        # else:
        #     self.FARM_PARAMETERS['nFarm'] = int(self.FARM_PARAMETERS['nFarm'])

        if 'farmAreaNC' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaNC'] = None
        # else:
        #     self.FARM_PARAMETERS['farmAreaNC'] = interpret_string(
        #         self.FARM_PARAMETERS['farmAreaNC'])

        if 'farmAreaVarName' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaVarName'] = None
        # else:
        #     self.FARM_PARAMETERS['farmAreaVarName'] = interpret_string(
        #         self.FARM_PARAMETERS['farmAreaVarName'])

        if 'AnnualChangeInFarmArea' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['AnnualChangeInFarmArea'] = False
        # else:
        #     self.FARM_PARAMETERS['AnnualChangeInFarmArea'] = interpret_logical_string(
        #         self.FARM_PARAMETERS['AnnualChangeInFarmArea'])

    def set_irrig_management_options(self):
        pass

    def set_field_management_options(self):
        if 'fieldManagementNC' not in list(self.FIELD_MANAGEMENT.keys()):
            self.FIELD_MANAGEMENT['fieldManagementNC'] = None
        # else:
        #     self.FIELD_MANAGEMENT['fieldManagementNC'] = interpret_string(
        #         self.FIELD_MANAGEMENT['fieldManagementNC'])

    def set_reporting_options(self):
        if 'REPORTING' not in self.config_sections:
            self.REPORTING = {}

        if 'report' not in list(self.REPORTING.keys()):
            self.REPORTING['report'] = False
