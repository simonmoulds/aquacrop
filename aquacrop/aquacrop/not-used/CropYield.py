#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging

logger = logging.getLogger(__name__)


class CropYield(object):
    def __init__(self, CropYield_variable):
        self.var = CropYield_variable

    def initial(self):
        arr_zeros = np.zeros(
            (self.var.nFarm, self.var.nCrop, self.var.domain.nxy)
        )
        self.var.CropMature = np.copy(arr_zeros).astype(bool)
        self.var.Y = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        self.var.CropMature[self.var.GrowingSeasonDayOne] = False

    def dynamic(self):
        """Function to calculate crop yield"""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()
        self.var.Y[self.var.GrowingSeasonIndex] = (
            (self.var.B / 100) * self.var.HIadj
        )[self.var.GrowingSeasonIndex]
        is_mature_calendar_type_one = (
            (self.var.CalendarType == 1)
            & ((self.var.DAP - self.var.DelayedCDs) >= self.var.Maturity)
        )
        is_mature_calendar_type_two = (
            (self.var.CalendarType == 2)
            & ((self.var.GDDcum - self.var.DelayedGDDs) >= self.var.Maturity)
        )
        is_mature = (
            self.var.GrowingSeasonIndex
            & (is_mature_calendar_type_one | is_mature_calendar_type_two)
        )
        self.var.CropMature[is_mature] = True
        self.var.Y[np.logical_not(self.var.GrowingSeasonIndex)] = 0
