#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class GrowingDegreeDay(object):
    def __init__(self, GrowingDegreeDay_variable):
        self.var = GrowingDegreeDay_variable

    def initial(self):
        self.var.GDDcum = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.GDD = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        
    def dynamic(self):
        aquacrop_fc.gdd_w.update_gdd_w(
            self.var.GDD.T, 
            self.var.GDDcum.T, 
            self.var.GDDmethod, 
            self.var.weather.tmax.T, 
            self.var.weather.tmin.T,
            self.var.Tbase.T,
            self.var.Tupp.T,
            self.var.GrowingSeasonIndex.T, 
            self.var.nFarm, self.var.nCrop, self.var.nCell
            )
