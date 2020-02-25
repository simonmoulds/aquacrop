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
            self.var.nFarm, self.var.nCrop, self.var.domain.nxy
        )
