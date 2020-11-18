#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from .LandCover import LandCover, Cropland

class LandSurface(object):
    def __init__(self, model):
        self.model = model
        self.land_cover_modules = {
            'cropland' : Cropland(self.model)
        }
        self.land_cover_module_names = [
            'cropland'
        ]
        self.land_cover_module_names_with_soil = [
            'cropland'
        ]
        # TODO
        # self.force_cover_fraction_sum_to_equal_one = (
        #     bool(int(self.var._configuration.LAND_COVER['forceCoverFractionSumToEqualOne']))
        # )
        # self.grid_cell_area_module = GridCellArea(LandSurface_variable)
        self.force_cover_fraction_sum_to_equal_one = False

        # TEMP:
        self.reporting_vars = self.model.config.REPORTING['daily_total'] + self.model.config.REPORTING['year_max']
        print(self.reporting_vars)
        
    def initial(self):
        for module in self.land_cover_module_names:
            self.land_cover_modules[module].initial()

        # self.grid_cell_area_module.initial()        
        self.initialize_cover_fraction()
        self.initialize_land_cover_variables_to_aggregate()
        self.aggregate_land_cover_variables()

    def initialize_cover_fraction(self):
        self.cover_fraction = {
            module : np.zeros((self.model.domain.nxy)) for module in self.land_cover_module_names
        }
        self.total_cover_fraction = np.zeros((self.model.domain.nxy))
        self.update_cover_fraction()
            
    def update_cover_fraction(self):
        for index,module in enumerate(self.land_cover_module_names):
            self.cover_fraction[module] = getattr(
                self.land_cover_modules[module],
                'cover_fraction')
        self.total_cover_fraction = np.sum(
            list(self.cover_fraction.values()),
            axis=0)
        self.correct_cover_fraction()
        
    def correct_cover_fraction(self):
        """Function to correct cover fraction, such that in 
        each grid square the various land cover fractions sum 
        to one
        """
        if self.force_cover_fraction_sum_to_equal_one:
            for index,module in enumerate(self.land_cover_module_names):
                vars(self.land_cover_modules[module])['cover_fraction'] /= self.total_cover_fraction
                self.cover_fraction[module] /= self.total_cover_fraction
        self.total_cover_fraction = np.sum(
            list(self.cover_fraction.values()),
            axis=0)
        self.total_cover_fraction_soil = np.sum(
            [self.cover_fraction[key] for key in self.land_cover_module_names_with_soil],
            axis=0)

    def initialize_land_cover_variables_to_aggregate(self):
        """Function to create lists of variable names to be 
        aggregated.

        variables_to_aggregate...
        soil_variables_to_aggregate...
        variables_to_aggregate_by_averaging...
        """
        # List of land cover variables which should be aggregated.
        # This should include variables which are common to all land
        # cover types; variables which are particular to a land cover
        # should be reported separately.
        self.variables_to_aggregate = []
        # List of soil variables to aggregate. This requires a
        # separate list because some variables are only relevant
        # for land cover types which interact with the soil column
        # (i.e. not impervious areas), e.g. th, wc
        self.soil_variables_to_aggregate = []
        # List of variable names which should be averaged rather than
        # summed across land cover types, e.g. th, wc
        self.variables_to_aggregate_by_averaging = []
        
        self.aggregate_land_cover_variables()

    # def aggregate_land_cover_variables(self):
    #     """Function to aggregate values computed separately 
    #     for each land cover
    #     """
    #     self.aggregate_variables_for_all_land_covers()
    #     self.aggregate_variables_for_land_covers_with_soil()
        
    # def aggregate_variables_for_all_land_covers(self):
    #     for var_name in self.variables_to_aggregate:
    #         var = self.aggregate_single_land_cover_variable(
    #             var_name,
    #             self.land_cover_module_names,
    #             self.total_cover_fraction
    #         )
    #         vars(self.model)[var_name] = var.copy()  # remove farm, crop dimension
            
    # def aggregate_variables_for_land_covers_with_soil(self):
    #     for var_name in self.soil_variables_to_aggregate:
    #         var = self.aggregate_single_land_cover_variable(
    #             var_name,
    #             self.land_cover_module_names_with_soil,
    #             self.total_cover_fraction_soil
    #             )            
    #         vars(self.model)[var_name] = var.copy()

    # def aggregate_single_land_cover_variable(self, var_name, module_names, total_cover_fraction):
    #     var_list = []
    #     for module in module_names:
    #         attr = getattr(self.land_cover_modules[module], var_name)
    #         # attempt to retrieve FarmCropArea from module to
    #         # use as a weight. If not present then we assume
    #         # the size of these dimensions is one.
    #         try:
    #             farm_crop_area = getattr(
    #                 self.land_cover_modules[module],
    #                 'FarmCropArea'
    #             )
    #             if var_name in self.soil_variables_to_aggregate:
    #                 farm_crop_area = np.broadcast_to(
    #                     farm_crop_area[...,None,:],
    #                     attr.shape
    #                 )
                    
    #         except AttributeError:
    #             farm_crop_area = np.ones_like(attr)

    #         # weighted average along farm, crop dimensions. We use
    #         # np.ma.average to handle the situation where the
    #         # sum of weights along the given axis equals zero
    #         attr_masked = np.ma.average(
    #             attr,
    #             axis=(0,1),
    #             weights=farm_crop_area
    #         )
    #         mask = attr_masked.mask
    #         attr = attr_masked.data
    #         attr[mask] = 0.
            
    #         # weight variable by area allocated to land cover
    #         # represented by module
    #         attr *= (self.cover_fraction[module] * self.model.domain.area)
    #         var_list.append(attr)

    #     # sum variables; if average is required, divide by total
    #     # cover fraction.
    #     # N.B. by not averaging we end up with a volume, because
    #     # we have multiplied by the area allocated to the
    #     # respective land cover
    #     var = np.sum(var_list, axis=0)
    #     if var_name in self.variables_to_aggregate_by_averaging:
    #         var /= (self.model.domain.area * total_cover_fraction)
    #     return var

    # def copy_reporting_vars(self):
    #     for k,v in self.land_cover_modules.items():
    #         for var in self.reporting_vars:
    #             pass
            
    def dynamic(self):
        for module in self.land_cover_module_names:
            self.land_cover_modules[module].dynamic()
        self.correct_cover_fraction()
        # self.aggregate_land_cover_variables()
