#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

from .RootZoneWater import fraction_of_compartment_in_root_zone, water_storage_in_root_zone
import aquacrop_fc

class Germination(object):
    
    def __init__(self, Germination_variable):
        self.var = Germination_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.DelayedGDDs = np.copy(arr_zeros)
        self.var.DelayedCDs = np.copy(arr_zeros.astype(np.int32))
        self.var.Germination = np.copy(arr_zeros.astype(np.int32))

    def dynamic(self):
        self.dynamic_numpy()
        
    # def dynamic_fortran(self):
    #     """Function to check if crop has germinated"""
    #     layer_ix = self.var.layerIndex + 1
    #     aquacrop_fc.germination_w.update_germ_w(
    #         self.var.Germination.T,
    #         self.var.DelayedCDs.T,
    #         self.var.DelayedGDDs.T,
    #         self.var.GDD.T,
    #         self.var.th.T,
    #         self.var.th_fc.T,
    #         self.var.th_wilt.T,
    #         self.var.zGerm.T,
    #         self.var.GermThr.T,
    #         self.var.dz,
    #         self.var.dz_sum,
    #         layer_ix,
    #         self.var.GrowingSeasonIndex.T,
    #         self.var.nFarm,
    #         self.var.nCrop,
    #         self.var.nComp,
    #         self.var.nLayer,
    #         self.var.domain.nxy
    #     )
        
    def dynamic_numpy(self):
        """Function to check if crop has germinated"""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()
            
        WcProp = self.compute_proportional_water_content_in_root_zone()
        above_germination_threshold = (
            self.var.GrowingSeasonIndex
            & (WcProp >= self.var.GermThr)
            & (np.logical_not(self.var.Germination))
        )
        self.var.Germination[above_germination_threshold] = True
        self.increment_delayed_growth_time_counters()
        self.update_ageing_days_counter()
        # self.update_time_since_germination()
        
        self.var.Germination[np.logical_not(self.var.GrowingSeasonIndex)] = False
        self.var.DelayedCDs[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.DelayedGDDs[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        
    def reset_initial_conditions(self):
        self.var.DelayedGDDs[self.var.GrowingSeasonDayOne] = 0
        self.var.DelayedCDs[self.var.GrowingSeasonDayOne] = 0
        self.var.Germination[self.var.GrowingSeasonDayOne] = False
        self.var.AgeDays[self.var.GrowingSeasonDayOne] = 0
        self.var.AgeDays_NS[self.var.GrowingSeasonDayOne] = 0
        # self.var.time_since_germination[self.var.GrowingSeasonDayOne] = 0

    def compute_proportional_water_content_in_root_zone(self):
        # root_factor = RootZoneWater.fraction_of_compartment_in_root_zone(
        root_factor = fraction_of_compartment_in_root_zone(
            self.var.zGerm,
            self.var.dz_sum_xy,
            self.var.dz_xy,
            self.var.nComp
        )
        Wr = water_storage_in_root_zone(self.var.th, self.var.dz_xy, root_factor)
        WrFC = water_storage_in_root_zone(self.var.th_fc_comp, self.var.dz_xy, root_factor)
        WrWP = water_storage_in_root_zone(self.var.th_wilt_comp, self.var.dz_xy, root_factor)
        # Wr = RootZoneWater.water_storage_in_root_zone(self.var.th, self.var.dz_xy, root_factor)
        # WrFC = RootZoneWater.water_storage_in_root_zone(self.var.th_fc_comp, self.var.dz_xy, root_factor)
        # WrWP = RootZoneWater.water_storage_in_root_zone(self.var.th_wilt_comp, self.var.dz_xy, root_factor)
        WrTAW = WrFC - WrWP
        WcProp = 1 - np.divide((WrFC - Wr), WrTAW, out=np.zeros_like(WrTAW), where=WrTAW!=0)
        return WcProp
        
    def increment_delayed_growth_time_counters(self):
        germination_yet_to_occur = (
            self.var.GrowingSeasonIndex
            & (np.logical_not(self.var.Germination))
        )
        self.var.DelayedCDs[germination_yet_to_occur] += 1
        self.var.DelayedGDDs[germination_yet_to_occur] += self.var.GDD[germination_yet_to_occur]
        
    def update_ageing_days_counter(self):
        DAPadj = (self.var.DAP - self.var.DelayedCDs)
        cond6 = (DAPadj > self.var.MaxCanopyCD) & self.var.GrowingSeasonIndex
        self.var.AgeDays[cond6] = (DAPadj - self.var.MaxCanopyCD)[cond6]
        cond7 = (self.var.DAP > self.var.MaxCanopyCD) & self.var.GrowingSeasonIndex  # NB not originally in this function
        self.var.AgeDays_NS[cond7] = (self.var.DAP - self.var.MaxCanopyCD)[cond7]

    # def update_time_since_germination(self):
    #     # Check if in yield formation period
    #     if self.var.CalendarType == 1:
    #         self.var.time_since_germination = self.var.DAP - self.var.DelayedCDs
    #     elif self.var.CalendarType == 2:
    #         self.var.time_since_germination = self.var.GDDcum - self.var.DelayedGDDs
            
