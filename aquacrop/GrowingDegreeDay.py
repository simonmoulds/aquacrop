#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

# class GrowingDegreeDay(object):
#     """Class to represent growing degree days"""

#     def __init__(self, GrowingDegreeDay_variable):
#         self.var = GrowingDegreeDay_variable

#     def initial(self):
#         self.var.GDDcum = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         self.var.GDD = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        
#     def reset_initial_conditions(self):
#         self.var.GDDcum[self.var.GrowingSeasonDayOne] = 0

#     def growing_degree_day(self):
#         tmax = self.var.weather.tmax.copy()
#         tmin = self.var.weather.tmin.copy()
#         if self.var.GDDmethod == 1:
#             Tmean = ((tmax + tmin) / 2)
#             Tmean = np.clip(Tmean, self.var.Tbase, self.var.Tupp)
#         elif self.var.GDDmethod == 2:
#             tmax = np.clip(tmax, self.var.Tbase, self.var.Tupp)
#             tmin = np.clip(tmin, self.var.Tbase, self.var.Tupp)
#             Tmean = ((tmax + tmin) / 2)
#         elif self.var.GDDmethod == 3:
#             tmax = np.clip(tmax, self.var.Tbase, self.var.Tupp)
#             tmin = np.clip(tmin, None, self.var.Tupp)
#             Tmean = ((tmax + tmin) / 2)
#             Tmean = np.clip(Tmean, self.var.Tbase, None)
#         self.var.GDD = (Tmean - self.var.Tbase)

#     def dynamic(self):
#         self.growing_degree_day()
#         self.var.GDDcum[self.var.GrowingSeasonIndex] += self.var.GDD[self.var.GrowingSeasonIndex]
#         self.var.GDDcum[np.logical_not(self.var.GrowingSeasonIndex)] = 0

class GrowingDegreeDay(object):
    """Class to represent growing degree days"""

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
