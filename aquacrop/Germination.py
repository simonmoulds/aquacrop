#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class Germination(object):
    def __init__(self, Germination_variable):
        self.var = Germination_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.DelayedGDDs = np.copy(arr_zeros)
        self.var.DelayedCDs = np.copy(arr_zeros.astype(np.int32))
        self.var.Germination = np.copy(arr_zeros.astype(np.int32))
            
    def dynamic(self):
        """Function to check if crop has germinated"""
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.germination_w.update_germ_w(
            self.var.Germination.T,
            self.var.DelayedCDs.T,
            self.var.DelayedGDDs.T,
            self.var.GDD.T,
            self.var.th.T,
            self.var.th_fc.T,
            self.var.th_wilt.T,
            self.var.zGerm.T,
            self.var.GermThr.T,
            self.var.dz,
            self.var.dz_sum,
            layer_ix,
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm,
            self.var.nCrop,
            self.var.nComp,
            self.var.nLayer,
            self.var.domain.nxy
        )
