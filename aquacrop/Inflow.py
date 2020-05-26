#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class Inflow(object):
    """Class to check for presence of a groundwater table and, if 
    present, to adjust compartment water contents and field 
    capacities accordingly.
    """
    def __init__(self, GroundwaterInflow_variable):
        self.var = GroundwaterInflow_variable

    def initial(self):
        self.var.GwIn = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))

    def dynamic(self):
        self.dynamic_fortran()

    def dynamic_fortran(self):
        self.var.GwIn = np.zeros((self.var.nCrop, self.var.nFarm, self.var.domain.nxy))
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.inflow_w.update_inflow_w(
            self.var.GwIn.T,
            self.var.th.T,
            np.int32(self.var.groundwater.WaterTable),
            self.var.groundwater.zGW.T,
            self.var.th_sat.T,
            self.var.dz,
            layer_ix,
            self.var.nFarm,
            self.var.nCrop,
            self.var.nComp,
            self.var.nLayer,
            self.var.domain.nxy
            )
        
    def dynamic_numpy(self):
        """Function to calculate capillary rise in the presence of a 
        shallow groundwater table
        """
        zGW_comp = np.broadcast_to(
            self.var.groundwater.zGW[None,None,None,:],
            (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy)
        )
        GwIn = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        zBot = np.cumsum(self.var.dz_xy, axis=2)  # sum along comp axis
        zTop = zBot - self.var.dz_xy
        zMid = (zTop + zBot) / 2

        # For compartments below water table, set to saturation
        dth = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        cond1 = (np.broadcast_to(self.var.WTinSoil[:,:,None,:].astype(np.bool), self.var.th.shape)
                 & np.greater_equal(zMid, zGW_comp)
                 & np.less(self.var.th, self.var.th_sat_comp))
        dth[cond1] = (self.var.th_sat_comp - self.var.th)[cond1]
        self.var.th[cond1] = self.var.th_sat_comp[cond1]

        # Update groundwater inflow
        GwIn_comp = dth * 1000 * self.var.dz_xy
        self.var.GwIn = np.sum(GwIn_comp, axis=2)  # sum along comp axis

# class Inflow(object):
#     """Class to check for presence of a groundwater table and, if 
#     present, to adjust compartment water contents and field 
#     capacities accordingly.
#     """
#     def __init__(self, GroundwaterInflow_variable):
#         self.var = GroundwaterInflow_variable

#     def initial(self):
#         self.GwIn = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        
#     def dynamic(self):
#         # """Function to calculate capillary rise in the presence of a 
#         # shallow groundwater table
#         # """
#         self.var.GwIn = np.zeros((self.var.nCrop, self.var.nFarm, self.var.domain.nxy))
#         layer_ix = self.var.layerIndex + 1
#         aquacrop_fc.inflow_w.update_inflow_w(
#             self.var.GwIn.T,
#             self.var.th.T,
#             np.int32(self.var.groundwater.WaterTable),
#             self.var.groundwater.zGW.T,
#             self.var.th_sat.T,
#             self.var.dz,
#             layer_ix,
#             self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
#             )
