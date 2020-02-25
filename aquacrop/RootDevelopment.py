#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class RootDevelopment(object):
    def __init__(self, RootDevelopment_variable):
        self.var = RootDevelopment_variable

    def initial(self):
        self.var.rCor = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.Zroot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))

    def reset_initial_conditions(self):
        self.var.rCor[self.var.GrowingSeasonDayOne] = 1
        self.var.Zroot[self.var.GrowingSeasonDayOne] = self.var.Zmin[self.var.GrowingSeasonDayOne]
            
    def dynamic(self):
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()

        aquacrop_fc.root_dev_w.update_root_dev_w(
            self.var.Zroot.T, 
            self.var.rCor.T, 
            self.var.Zmin.T, 
            self.var.Zmax.T, 
            self.var.PctZmin.T, 
            self.var.Emergence.T, 
            self.var.MaxRooting.T, 
            self.var.fshape_r.T, 
            self.var.fshape_ex.T, 
            self.var.SxBot.T,
            self.var.SxTop.T,
            self.var.DAP.T,
            self.var.GDD.T,
            self.var.GDDcum.T,
            self.var.DelayedCDs.T,
            self.var.DelayedGDDs.T,
            self.var.TrRatio.T,
            self.var.Germination.T, 
            self.var.zRes.T,
            self.var.groundwater.WaterTable, 
            self.var.groundwater.zGW, 
            self.var.CalendarType, 
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm, self.var.nCrop, self.var.domain.nxy
        )
            
