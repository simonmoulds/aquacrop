#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hm.config import Configuration
import warnings
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
        self.set_etref_preprocess_options()
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
        # make sure that pseudo-coordinates are lists
        if 'PSEUDO_COORDS' not in self.config_sections:
            self.PSEUDO_COORDS = {}
        else:
            for key, value in self.PSEUDO_COORDS.items():
                if isinstance(value, (int, float)):
                    self.PSEUDO_COORDS[key] = [value]
                elif isinstance(value, (str)):
                    self.PSEUDO_COORDS[key] = eval(value)

    def check_config_file_for_required_entry(
            self,
            section_name,
            entry_name,
            allow_none=False,
            allow_empty=False
    ):
        if entry_name not in list(vars(self)[section_name].keys()):
            raise KeyError(
                entry_name + ' in section ' + section_name + ' must be provided'
            )
        else:
            entry = vars(self)[section_name][entry_name]
            if not allow_none and (entry in VALID_NONE_VALUES):
                raise ValueError(
                    entry_name + ' in section ' + section_name + 'cannot by empty or none'
                )
            else:
                pass

    def set_model_grid_options(self):
        pass

    def set_weather_options(self):
        weather_sections = ['PRECIPITATION', 'TMIN', 'TMAX', 'TAVG', 'ETREF', 'CARBON_DIOXIDE']
        for section in weather_sections:
            for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
                if opt not in vars(self)[section].keys():
                    vars(self)[section][opt] = default_value

    def set_etref_preprocess_options(self):
        if 'preprocess' in vars(self)['ETREF'].keys():
            if vars(self)['ETREF']['preprocess'] is True:
                vars(self)['ETREF']['report'] = True
                vars(self)['ETREF']['daily_total'] = 'ETref'
            else:
                vars(self)['ETREF']['preprocess'] = False
        else:
            vars(self)['ETREF']['preprocess'] = False
                
    def set_initial_condition_options(self):
        self.check_config_file_for_required_entry(
            'INITIAL_WATER_CONTENT',
            'type'
        )
        init_type = self.INITIAL_WATER_CONTENT['type'].upper()
        if init_type == 'FILE':
            if 'filename' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'type is FILE, but '
                    'filename is not specified'
                )

            elif 'interp_method' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'if type is FILE then interp_method must be specified as one of DEPTH or LAYER'
                )

            # set default options
            for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
                if opt not in self.INITIAL_WATER_CONTENT.keys():
                    self.INITIAL_WATER_CONTENT[opt] = default_value

        elif init_type == 'PERCENT':
            if 'percent' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'type is PERCENT, but percent is not specified'
                )

        elif init_type == 'PROPERTY':
            if 'property' not in list(self.INITIAL_WATER_CONTENT.keys()):
                raise KeyError(
                    'type is PROPERTY, but property is not specified'
                )

            else:
                prop = self.INITIAL_WATER_CONTENT['property'].upper()
                if prop not in ['SAT', 'WP', 'FC']:
                    raise ValueError(
                        'property must be one of sat, wp, fc'
                    )

        else:
            raise ValueError(
                'type must be one of file, percent, property'
            )

    def set_groundwater_options(self):
        if 'water_table' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['water_table'] = False

        if 'dynamic' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['dynamic'] = False

        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.WATER_TABLE.keys():
                self.WATER_TABLE[opt] = default_value
            
        if 'coupled' not in list(self.WATER_TABLE.keys()):
            self.WATER_TABLE['coupled'] = False
            
        if self.WATER_TABLE['coupled']:
            if 'directory' not in list(self.WATER_TABLE.keys()):
                raise ValueError()            
            elif 'time_lag' not in list(self.WATER_TABLE.keys()):
                raise ValueError()
            elif 'max_wait_time' not in list(self.WATER_TABLE.keys()):
                raise ValueError()
            elif 'wait_interval' not in list(self.WATER_TABLE.keys()):
                raise ValueError()
        else:
            self.WATER_TABLE['directory'] = ""
            self.WATER_TABLE['time_lag'] = None
            self.WATER_TABLE['max_wait_time'] = None
            self.WATER_TABLE['wait_interval'] = None

    def set_soil_profile_options(self):
        pass

    def set_soil_hydrology_options(self):
        # FIXME
        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.SOIL_HYDRAULIC_PARAMETERS.keys():
                self.SOIL_HYDRAULIC_PARAMETERS[opt] = default_value

    def set_soil_parameter_options(self):
        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.SOIL_PARAMETERS.keys():
                self.SOIL_PARAMETERS[opt] = default_value
        # if ('filename' not in list(self.SOIL_PARAMETERS.keys())) | \
        #    (self.SOIL_PARAMETERS['filename'] in VALID_NONE_VALUES):
        #     self.SOIL_PARAMETERS['filename'] = None

        if 'adjustReadilyAvailableWater' not in list(self.SOIL_PARAMETERS.keys()):
            self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] = False
        elif self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] is True:
            warnings.warn(
                'Adjustment of readily evaporable water is not currently '
                'implemented: setting to False'
            )
            self.SOIL_PARAMETERS['adjustReadilyAvailableWater'] = False

        if 'adjustCurveNumber' not in list(self.SOIL_PARAMETERS.keys()):
            self.SOIL_PARAMETERS['adjustCurveNumber'] = False

    def set_crop_parameter_options(self):
        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.CROP_PARAMETERS.keys():
                self.CROP_PARAMETERS[opt] = default_value
                
        if 'crop_id' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['crop_id'] = [1]
            
        if 'calendar_type' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['calendar_type'] = 1

        if 'switch_gdd' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['switch_gdd'] = False

        if 'gdd_method' not in list(self.CROP_PARAMETERS.keys()):
            self.CROP_PARAMETERS['gdd_method'] = 1

        # if 'nRotation' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['nRotation'] = 1

        # if 'landCoverFractionNC' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['landCoverFractionNC'] = None

        # if 'landCoverFractionVarName' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['landCoverFractionVarName'] = None

        # if 'cropAreaNC' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['cropAreaNC'] = None

        # if 'cropAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['cropAreaVarName'] = None

        # if 'croplandAreaNC' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['croplandAreaNC'] = None

        # if 'croplandAreaVarName' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['croplandAreaVarName'] = None

        # if 'AnnualChangeInCropArea' not in list(self.CROP_PARAMETERS.keys()):
        #     self.CROP_PARAMETERS['AnnualChangeInCropArea'] = False

    def set_farm_parameter_options(self):
        if 'FARM_PARAMETERS' not in self.config_sections:
            self.FARM_PARAMETERS = {}

        if 'nFarm' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['nFarm'] = 1

        if 'farmAreaNC' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaNC'] = None

        if 'farmAreaVarName' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['farmAreaVarName'] = None

        if 'AnnualChangeInFarmArea' not in list(self.FARM_PARAMETERS.keys()):
            self.FARM_PARAMETERS['AnnualChangeInFarmArea'] = False

    def set_irrig_management_options(self):
        pass

    def set_field_management_options(self):
        for opt, default_value in DEFAULT_INPUT_FILE_OPTS.items():
            if opt not in self.FIELD_MANAGEMENT.keys():
                self.FIELD_MANAGEMENT[opt] = default_value
        # if 'filename' not in list(self.FIELD_MANAGEMENT.keys()):
        #     self.FIELD_MANAGEMENT['filename'] = None

    def set_reporting_options(self):
        if 'REPORTING' not in self.config_sections:
            self.REPORTING = {}

        if 'report' not in list(self.REPORTING.keys()):
            self.REPORTING['report'] = False
