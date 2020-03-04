#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class RainfallPartition(object):
    """Class to infiltrate incoming water"""
    
    def __init__(self, RainfallPartition_variable):
        self.var = RainfallPartition_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm,self.var.nCrop, self.var.domain.nxy))        
        self.var.Runoff = np.copy(arr_zeros)
        self.var.Infl = np.copy(arr_zeros)
        
    # def dynamic(self):
    #     layer_ix = self.var.layerIndex + 1
    #     aquacrop_fc.rainfall_partition_w.update_rain_part_w(
    #         self.var.Runoff.T,
    #         self.var.Infl.T,
    #         self.var.model.prec.values.T,
    #         # self.var.weather.precipitation.T,
    #         self.var.th.T,
    #         np.int32(self.var.DaySubmerged).T,
    #         np.int32(self.var.Bunds).T,
    #         self.var.zBund.T,
    #         self.var.th_fc.T,
    #         self.var.th_wilt.T,
    #         np.int32(self.var.CN).T,
    #         np.int32(self.var.adjustCurveNumber),
    #         self.var.zCN.T,
    #         self.var.CNbot.T,
    #         self.var.CNtop.T,
    #         self.var.dz,
    #         self.var.dz_sum,
    #         layer_ix,
    #         self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
    #         )
    def compute_relative_wetness_of_topsoil(self):

        wet_top_comp = (
            self.var.weighting_factor_for_cn_adjustment
            * ((self.var.th - self.var.th_wilt_comp)
               / (self.var.th_fc_comp - self.var.th_wilt_comp))
        )
        include_compartment = np.round(self.var.dz_sum_xy, 3) <= np.round(self.var.zCN[...,None,:], 3)
        all_false = np.all((include_compartment == False), axis=2)
        all_false = np.broadcast_to(all_false[:,:,None,:], include_compartment.shape)
        include_compartment[all_false] = True
        # print('wet_top_comp:',wet_top_comp[0,0,:,0])
        # print('comp_sto:    ',include_compartment[0,0,:,0])
        # print('dz_sum:      ',self.var.dz_sum_xy[0,0,:,0])
        # print('zCN:         ',self.var.zCN[0,0,0])
        wet_top_comp *= include_compartment
        # print(wet_top_comp[0,0,:,0])
        wet_top = np.sum(wet_top_comp, axis=2)  # sum along compartment dimension
        wet_top = np.clip(wet_top, 0, 1)
        return wet_top
        
    def adjust_curve_number(self):
        CN = np.copy(self.var.CN)
        wet_top = self.compute_relative_wetness_of_topsoil()
        CN = np.round(self.var.CNbot + (self.var.CNtop - self.var.CNbot) * wet_top)
        # print('wet_top:',wet_top)
        # print('th:',self.var.th[0,0,:,0])
        return CN
    
    @staticmethod
    def compute_potential_maximum_soil_water_retention(CN):
        return ((25400. / CN) - 254.)
        
    def dynamic(self):
        prec = self.var.model.prec.values[None,None,:]
        if self.var.adjustCurveNumber:
            CN = self.adjust_curve_number()
        S = self.compute_potential_maximum_soil_water_retention(CN)
        initial_abstraction = ((5. / 100) * S)
        bunds = ((self.var.Bunds == 1) & (self.var.zBund >= 0.001))
        runoff = (
            np.logical_not(bunds)
            & (prec > initial_abstraction)
        )
        self.var.Runoff[np.logical_not(runoff)] = 0
        self.var.Infl[np.logical_not(runoff)] = prec[np.logical_not(runoff)]
        self.var.Runoff[runoff] = (
            ((prec - initial_abstraction) ** 2)
            / (prec + (1 - (5. / 100)) * S)
        )[runoff]        
        self.var.Infl[runoff] = (prec - self.var.Runoff)[runoff]
        # print('CN    : ',CN)
        # print('S     : ',S)
        # print('Runoff: ',self.var.Runoff)
        # print('Infl  : ',self.var.Infl)
