
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hm import file_handling

import logging
logger = logging.getLogger(__name__)

class CropYield(object):
    def __init__(self, CropYield_variable):
        self.var = CropYield_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.CropMature = np.copy(arr_zeros).astype(bool)
        self.var.Y = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        self.var.CropMature[self.var.GrowingSeasonDayOne] = False
        
    def dynamic(self):
        """Function to calculate crop yield"""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()

        cond1 = self.var.GrowingSeasonIndex
        self.var.Y[cond1] = ((self.var.B / 100) * self.var.HIadj)[cond1]

        # print('Y    :',self.var.Y[...,500])
        # print('B    :',self.var.B[...,500])
        # print('HIadj:',self.var.HIadj[...,500])
        
        cond11 = (cond1 & (((self.var.CalendarType == 1) & ((self.var.DAP - self.var.DelayedCDs) >= self.var.Maturity)) | ((self.var.CalendarType == 2) & ((self.var.GDDcum - self.var.DelayedGDDs) >= self.var.Maturity))))
        self.var.CropMature[cond11] = True
        self.var.Y[np.logical_not(self.var.GrowingSeasonIndex)] = 0
