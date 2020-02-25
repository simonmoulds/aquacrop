#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class Infiltration(object):
    """Class to infiltrate incoming water (rainfall and irrigation)"""
    
    def __init__(self, Infiltration_variable):
        self.var = Infiltration_variable

    def initial(self):        
        cond1 = (self.var.Bunds == 0) & (self.var.zBund > 0.001)
        SurfaceStorage = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        SurfaceStorage[cond1] = self.var.BundWater[cond1]
        SurfaceStorage = np.clip(SurfaceStorage, None, self.var.zBund)
        self.var.SurfaceStorage = np.copy(SurfaceStorage)
        self.var.SurfaceStorageIni = np.copy(SurfaceStorage)

    def reset_initial_conditions(self):
        pass
        
    def dynamic(self):
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()
            
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.infiltration_w.update_infl_w(
            self.var.Infl.T,
            self.var.SurfaceStorage.T,
            self.var.FluxOut.T,
            self.var.DeepPerc.T,
            self.var.Runoff.T,
            self.var.th.T,
            self.var.Irr.T,
            self.var.AppEff.T,
            self.var.Bunds.T,
            self.var.zBund.T,
            self.var.th_sat.T,
            self.var.th_fc.T,
            self.var.th_fc_adj.T,
            self.var.k_sat.T,
            self.var.tau.T,
            self.var.dz,
            layer_ix,
            self.var.nFarm,
            self.var.nCrop,
            self.var.nComp,
            self.var.nLayer,
            self.var.domain.nxy
            )
