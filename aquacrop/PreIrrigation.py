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
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.PreIrr = np.copy(arr_zeros)
        self.var.IrrNet = np.copy(arr_zeros)

    def dynamic(self):
        self.dynamic_numpy()
        
    def dynamic_fortran(self):
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
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
            )
        
    def dynamic_numpy(self):
        rootdepth = np.maximum(self.var.Zmin, self.var.Zroot)
        rootdepth = np.round(rootdepth * 100) / 100
        thCrit = (
            self.var.th_wilt_comp
            + ((self.var.NetIrrSMT[...,None,:] / 100)
               * (self.var.th_fc_comp - self.var.th_wilt_comp)
            )
        )        
        irrigate = ((self.var.IrrMethod == 4) & (self.var.DAP == 1))
        irrigate_compartment = (
            irrigate[...,None,:]
            & ((self.var.dz_sum_xy - self.var.dz_xy) < rootdepth[...,None,:])
            & (self.var.th < thCrit)
        )        
        PreIrr_req = ((thCrit - self.var.th) * 1000 * self.var.dz_xy)
        PreIrr_req[np.logical_not(irrigate_compartment)] = 0
        self.var.PreIrr = np.sum(PreIrr_req, axis=2)  # sum over compartments
        self.var.th[irrigate_compartment] = thCrit[irrigate_compartment]
