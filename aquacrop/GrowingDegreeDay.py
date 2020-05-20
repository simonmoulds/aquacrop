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
        self.var.GDDcum = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.GDD = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))

    def dynamic(self):
        # self.dynamic_fortran()
        self.dynamic_numpy()
        
    def dynamic_fortran(self):
        # print(self.var.GDD.shape)
        # print(self.var.GDDcum.shape)
        # print(self.var.GDDmethod)
        # print(self.var.model.tmax.values.shape)
        # print(self.var.model.tmin.values.shape)
        # print(self.var.Tbase.shape)
        # print(self.var.Tupp.shape)
        # print(self.var.GrowingSeasonIndex.shape)
        aquacrop_fc.gdd_w.update_gdd_w(
            self.var.GDD.T, 
            self.var.GDDcum.T, 
            self.var.GDDmethod, 
            self.var.model.tmax.values.T, 
            self.var.model.tmin.values.T,
            self.var.Tbase.T,
            self.var.Tupp.T,
            self.var.GrowingSeasonIndex.T, 
            self.var.nFarm, self.var.nCrop, self.var.domain.nxy
            )
        
    def dynamic_numpy(self):        
        self.growing_degree_day()
        self.var.GDDcum[self.var.GrowingSeasonIndex] += self.var.GDD[self.var.GrowingSeasonIndex]
        self.var.GDDcum[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        
    # def reset_initial_conditions(self):
    #     self.var.GDDcum[self.var.GrowingSeasonDayOne] = 0

    def growing_degree_day(self):
        tmax = self.var.model.tmax.values.copy()
        tmin = self.var.model.tmin.values.copy()
        if self.var.GDDmethod == 1:
            Tmean = ((tmax + tmin) / 2)
            Tmean = np.clip(Tmean, self.var.Tbase, self.var.Tupp)
        elif self.var.GDDmethod == 2:
            tmax = np.clip(tmax, self.var.Tbase, self.var.Tupp)
            tmin = np.clip(tmin, self.var.Tbase, self.var.Tupp)
            Tmean = ((tmax + tmin) / 2)
        elif self.var.GDDmethod == 3:
            tmax = np.clip(tmax, self.var.Tbase, self.var.Tupp)
            tmin = np.clip(tmin, None, self.var.Tupp)
            Tmean = ((tmax + tmin) / 2)
            Tmean = np.clip(Tmean, self.var.Tbase, None)
        self.var.GDD = (Tmean - self.var.Tbase)

