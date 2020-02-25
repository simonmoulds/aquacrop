#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class RootZoneWater(object):
    def __init__(self, RootZoneWater_variable):
        self.var = RootZoneWater_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.thRZ_Act = np.copy(arr_zeros)
        self.var.thRZ_Sat = np.copy(arr_zeros)
        self.var.thRZ_Fc = np.copy(arr_zeros)
        self.var.thRZ_Wp = np.copy(arr_zeros)
        self.var.thRZ_Dry = np.copy(arr_zeros)
        self.var.thRZ_Aer = np.copy(arr_zeros)
        self.var.TAW = np.copy(arr_zeros)
        self.var.Dr = np.copy(arr_zeros)
        self.var.Wr = np.copy(arr_zeros)

    def dynamic(self):
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
            self.var.thRZ_Act.T, 
            self.var.thRZ_Sat.T, 
            self.var.thRZ_Fc.T, 
            self.var.thRZ_Wp.T, 
            self.var.thRZ_Dry.T, 
            self.var.thRZ_Aer.T, 
            self.var.TAW.T, 
            self.var.Dr.T, 
            self.var.th.T, 
            self.var.th_sat.T, 
            self.var.th_fc.T, 
            self.var.th_wilt.T, 
            self.var.th_dry.T, 
            self.var.Aer.T, 
            self.var.Zroot.T, 
            self.var.Zmin.T, 
            self.var.dz, 
            self.var.dz_sum, 
            layer_ix, 
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
        )
            
