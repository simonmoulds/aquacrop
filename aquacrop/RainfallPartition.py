#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

class RainfallPartition(object):
    """Class to infiltrate incoming water"""
    
    def __init__(self, RainfallPartition_variable):
        self.var = RainfallPartition_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm,self.var.nCrop, self.var.nCell))        
        self.var.Runoff = np.copy(arr_zeros)
        self.var.Infl = np.copy(arr_zeros)

    def compute_relative_wetness_of_topsoil(self):
        wet_top_comp = (
            self.var.weighting_factor_for_cn_adjustment
            * ((self.var.th - self.var.th_wilt_comp)
               / (self.var.th_fc_comp - self.var.th_wilt_comp))
        )
        include_compartment = self.var.dz_sum_xy <= self.var.zCN[...,None,:]
        all_false = np.all((include_compartment == False), axis=2)
        all_false = np.broadcast_to(all_false[:,:,None,:], include_compartment.shape)
        include_compartment[all_false] = True        
        wet_top_comp *= include_compartment
        wet_top = np.sum(wet_top_comp, axis=2)  # sum along compartment dimension
        wet_top = np.clip(wet_top, 0, 1)
        return wet_top
        
    def adjust_curve_number(self):
        CN = np.copy(self.var.CN)
        wet_top = self.compute_relative_wetness_of_topsoil()
        CN = np.round(self.var.CNbot + (self.var.CNtop - self.var.CNbot) * wet_top)
        return CN
    
    @staticmethod
    def compute_potential_maximum_soil_water_retention(CN):
        return ((25400. / CN) - 254.)
        
    def dynamic(self):
        if self.var.adjustCurveNumber:
            CN = self.adjust_curve_number()
        S = self.compute_potential_maximum_soil_water_retention(CN)
        initial_abstraction = ((5. / 100) * S)
        bunds = ((self.var.Bunds == 1) & (self.var.zBund >= 0.001))
        runoff = (
            np.logical_not(bunds)
            & (self.var.weather.precipitation > initial_abstraction)
        )
        self.var.Runoff[np.logical_not(runoff)] = 0
        self.var.Infl[np.logical_not(runoff)] = self.var.weather.precipitation[np.logical_not(runoff)]
        self.var.Runoff[runoff] = (
            ((self.var.weather.precipitation - initial_abstraction) ** 2)
            / (self.var.weather.precipitation + (1 - (5. / 100)) * S)
        )[runoff]
        self.var.Infl[runoff] = (self.var.weather.precipitation - self.var.Runoff)[runoff]
