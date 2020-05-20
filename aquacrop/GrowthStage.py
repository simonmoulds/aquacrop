#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class GrowthStage(object):
    def __init__(self, GrowthStage_variable):
        self.var = GrowthStage_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.GrowthStage = arr_zeros.copy()
        self.var.time_since_germination = arr_zeros.copy()
        self.var.time_since_germination_previous = arr_zeros.copy()
        
    def dynamic(self):
        aquacrop_fc.growth_stage_w.update_growth_stage_w(
            np.int32(self.var.GrowthStage).T,
            self.var.Canopy10Pct.T,
            self.var.MaxCanopy.T,
            self.var.Senescence.T,
            self.var.GDDcum.T,
            np.int32(self.var.DAP).T,
            self.var.DelayedCDs.T,
            self.var.DelayedGDDs.T,
            int(self.var.CalendarType),
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm,
            self.var.nCrop,
            self.var.domain.nxy
        )        
    # def reset_initial_conditions(self):
    #     self.var.GrowthStage[self.var.GrowingSeasonDayOne] = 0
    #     self.var.time_since_germination[self.var.GrowingSeasonDayOne] = 0
    #     self.var.time_since_germination_previous[self.var.GrowingSeasonDayOne] = 0

    # def update_time_since_germination(self):
    #     # Check if in yield formation period
    #     self.var.time_since_germination_previous = self.var.time_since_germination.copy()
    #     if self.var.CalendarType == 1:
    #         self.var.time_since_germination = self.var.DAP - self.var.DelayedCDs
    #     elif self.var.CalendarType == 2:
    #         self.var.time_since_germination = self.var.GDDcum - self.var.DelayedGDDs
        
    # def dynamic(self):
    #     self.update_time_since_germination()
    #     cond1 = (self.var.GrowingSeasonIndex & (self.var.time_since_germination <= self.var.Canopy10Pct))
    #     cond2 = (self.var.GrowingSeasonIndex & np.logical_not(cond1) & (self.var.time_since_germination <= self.var.MaxCanopy))
    #     cond3 = (self.var.GrowingSeasonIndex & np.logical_not(cond1 | cond2) & (self.var.time_since_germination <= self.var.Senescence))
    #     cond4 = (self.var.GrowingSeasonIndex & np.logical_not(cond1 | cond2 | cond3) & (self.var.time_since_germination > self.var.Senescence))
    #     self.var.GrowthStage[cond1] = 1
    #     self.var.GrowthStage[cond2] = 2
    #     self.var.GrowthStage[cond3] = 3
    #     self.var.GrowthStage[cond4] = 4
    #     self.var.GrowthStage[np.logical_not(self.var.GrowingSeasonIndex)] = 0
