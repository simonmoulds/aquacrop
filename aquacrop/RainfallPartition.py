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
        arr_zeros = np.zeros((self.var.nFarm,self.var.nCrop, self.var.nCell))        
        self.var.Runoff = np.copy(arr_zeros)
        self.var.Infl = np.copy(arr_zeros)
        
    def dynamic(self):
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.rainfall_partition_w.update_rain_part_w(
            self.var.Runoff.T,
            self.var.Infl.T,
            self.var.weather.precipitation.T,
            self.var.th.T,
            np.int32(self.var.DaySubmerged).T,
            np.int32(self.var.Bunds).T,
            self.var.zBund.T,
            self.var.th_fc.T,
            self.var.th_wilt.T,
            np.int32(self.var.CN).T,
            np.int32(self.var.adjustCurveNumber),
            self.var.zCN.T,
            self.var.CNbot.T,
            self.var.CNtop.T,
            self.var.dz,
            self.var.dz_sum,
            layer_ix,
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.nCell
            )
