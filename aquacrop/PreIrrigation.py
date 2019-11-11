#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class PreIrrigation(object):
    
    def __init__(self, PreIrrigation_variable):
        self.var = PreIrrigation_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.PreIrr = np.copy(arr_zeros)
        self.var.IrrNet = np.copy(arr_zeros)
        
    def dynamic(self):
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.pre_irr_w.update_pre_irr_w(
            self.var.PreIrr.T,
            self.var.th.T,
            self.var.IrrMethod.T,
            self.var.DAP.T,
            self.var.Zroot.T,
            self.var.Zmin.T,
            self.var.NetIrrSMT.T,
            self.var.th_fc.T,
            self.var.th_wilt.T,
            self.var.dz,
            self.var.dz_sum,
            layer_ix,
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.nCell
            )        
