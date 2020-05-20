#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class CheckGroundwaterTable(object):
    def __init__(self, CheckGroundwaterTable_variable):
        self.var = CheckGroundwaterTable_variable

    def initial(self):
        self.var.th_fc_adj = np.copy(self.var.th_fc_comp)
        self.var.WTinSoil = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy), dtype=np.int32)

    def dynamic(self):        
        self.dynamic_fortran()
        
    def dynamic_fortran(self):
        layer_ix = self.var.layerIndex + 1
        # th_f = np.require(self.var.th.T, requirements=['F'])
        # th_fc_adj_f = np.require(self.var.th_fc_adj.T, requirements=['F'])
        # wt_in_soil_f = np.require(np.int32(self.var.WTinSoil.T), requirements=['F'])
        # th_sat_f = np.require(self.var.th_sat.T, requirements=['F'])
        # th_fc_f = np.require(self.var.th_fc.T, requirements=['F'])
        # aquacrop_fc.check_gw_table_w.update_check_gw_table_w(
        #     th_f,
        #     th_fc_adj_f,
        #     wt_in_soil_f,
        #     th_sat_f,
        #     th_fc_f,
        #     int(self.var.groundwater.WaterTable),
        #     int(self.var.groundwater.DynamicWaterTable),
        #     self.var.groundwater.zGW,
        #     self.var.dz,
        #     layer_ix,
        #     self.var.nFarm,
        #     self.var.nCrop,
        #     self.var.nComp,
        #     self.var.nLayer,
        #     self.var.domain.nxy
        # )
        aquacrop_fc.check_gw_table_w.update_check_gw_table_w(
            self.var.th.T,
            self.var.th_fc_adj.T,
            np.int32(self.var.WTinSoil).T,
            self.var.th_sat.T,
            self.var.th_fc.T,
            int(self.var.groundwater.WaterTable),
            int(self.var.groundwater.DynamicWaterTable),
            self.var.groundwater.zGW,
            self.var.dz,
            layer_ix,
            self.var.nFarm,
            self.var.nCrop,
            self.var.nComp,
            self.var.nLayer,
            self.var.domain.nxy
        )

