#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class CheckGroundwaterTable(object):
    """Class to check for presence of a groundwater table and, if 
    present, to adjust compartment water contents and field 
    capacities where necessary
    """
    def __init__(self, CheckGroundwaterTable_variable):
        self.var = CheckGroundwaterTable_variable

    def initial(self):
        self.var.th_fc_adj = np.copy(self.var.th_fc_comp)
        self.var.WTinSoil = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell),dtype=bool)

    def reset_initial_conditions(self):
        self.var.WTinSoil[self.var.GrowingSeasonDayOne] = False

    def compute_mid_point_of_compartments(self):
        """Estimate the depth to the mid-point of each compartment.
        """
        # get the mid point of each compartment
        zBot = np.cumsum(self.var.dz)
        zTop = zBot - self.var.dz
        zMid = (zTop + zBot) / 2
        zMid = np.broadcast_to(zMid[None,None,:,None], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
        return zMid

    def compute_Xmax(self):
        Xmax = np.zeros((self.var.nFarm,self.var.nCrop,self.var.nComp,self.var.nCell))
        cond1 = self.var.th_fc_comp <= 0.1
        cond2 = self.var.th_fc_comp >= 0.3
        cond3 = np.logical_not(cond1 | cond2) # i.e. 0.1 < fc < 0.3
        Xmax[cond1] = 1
        Xmax[cond2] = 2
        pF = 2 + 0.3 * (self.var.th_fc_comp - 0.1) / 0.2
        Xmax_cond3 = np.exp(pF * np.log(10)) / 100
        Xmax[cond3] = Xmax_cond3[cond3]
        return Xmax

    def compute_th_fc_adj(self):
        # Copy depth to groundwater, and add crop/farm dimension for
        # convenience
        zGW_comp = np.broadcast_to(
            self.var.groundwater.zGW[None,None,None,:],
            (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell)
        )

        zMid = self.compute_mid_point_of_compartments()
        WTinSoilComp = (zMid >= zGW_comp)
        print(self.var.th[0,0,:,0])
        self.var.th[WTinSoilComp] = self.var.th_sat_comp[WTinSoilComp]

        # Flatten WTinSoilComp to provide an array with dimensions
        # (ncrop, nCell), indicating crops where the water table is
        # in the soil profile
        self.var.WTinSoil = np.sum(WTinSoilComp, axis=2).astype(bool)  # CHECK            
        Xmax = self.compute_Xmax()

        # Index of the compartment to which each element belongs
        # (shallow -> deep, i.e. 1 is the shallowest)
        compartment = (
            np.arange(1, self.var.nComp + 1)[None,None,:,None]
            * np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))[:,:,None,:]
        )

        # Index of the lowest compartment (i.e. the maximum value) for which
        # cond4 is met, cast to all compartments (achieved by multiplying
        # compartments by cond4 to set elements that do not equal the
        # condition to zero, but retain the compartment number of elements
        # that do meet the condition
        cond4 = (zGW_comp < 0) | ((zGW_comp - zMid) >= Xmax)
        cond4_max_compartment = (
            np.amax(compartment * cond4, axis=2)[:,:,None,:]
            * np.ones((self.var.nComp))[None,None,:,None]
        )

        # Now, identify compartments that are shallower than the deepest
        # compartment for which cond4 is met
        cond4 = (compartment <= cond4_max_compartment)
        cond5 = (self.var.th_fc_comp >= self.var.th_sat_comp) & np.logical_not(cond4)
        cond6 = (zMid >= zGW_comp) & np.logical_not(cond4 | cond5)
        cond7 = np.logical_not(cond4 | cond5 | cond6)
        dV = self.var.th_sat_comp - self.var.th_fc_comp
        dFC = (dV / (Xmax ** 2)) * ((zMid - (zGW_comp - Xmax)) ** 2)

        self.var.th_fc_adj[cond4] = self.var.th_fc_comp[cond4]
        self.var.th_fc_adj[cond5] = self.var.th_fc_comp[cond5]
        self.var.th_fc_adj[cond6] = self.var.th_sat_comp[cond6]
        self.var.th_fc_adj[cond7] = self.var.th_fc_comp[cond7] + dFC[cond7]
        
    def dynamic(self):
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()
        if self.var.groundwater.WaterTable:
            self.compute_th_fc_adj()
        else:
            self.var.WTinSoil = np.full((self.var.nFarm, self.var.nCrop, self.var.nCell), False)
            self.var.th_fc_adj = np.copy(self.var.th_fc_comp)

# class CheckGroundwaterTable(object):
#     """Class to check for presence of a groundwater table and, if 
#     present, to adjust compartment water contents and field 
#     capacities where necessary
#     """
#     def __init__(self, CheckGroundwaterTable_variable):
#         self.var = CheckGroundwaterTable_variable

#     def initial(self):
#         self.var.th_fc_adj = np.copy(self.var.th_fc_comp)
#         self.var.WTinSoil = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell), dtype=np.int32)

#     def reset_initial_conditions(self):
#         self.var.WTinSoil[self.var.GrowingSeasonDayOne] = False

#     def compute_mid_point_of_compartments(self):
#         """Estimate the depth to the mid-point of each compartment.
#         """
#         # get the mid point of each compartment
#         zBot = np.cumsum(self.var.dz)
#         zTop = zBot - self.var.dz
#         zMid = (zTop + zBot) / 2
#         zMid = np.broadcast_to(zMid[None,None,:,None], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
#         return zMid

#     def compute_Xmax(self):
#         Xmax = np.zeros((self.var.nFarm,self.var.nCrop,self.var.nComp,self.var.nCell))
#         cond1 = self.var.th_fc_comp <= 0.1
#         cond2 = self.var.th_fc_comp >= 0.3
#         cond3 = np.logical_not(cond1 | cond2) # i.e. 0.1 < fc < 0.3
#         Xmax[cond1] = 1
#         Xmax[cond2] = 2
#         pF = 2 + 0.3 * (self.var.th_fc_comp - 0.1) / 0.2
#         Xmax_cond3 = np.exp(pF * np.log(10)) / 100
#         Xmax[cond3] = Xmax_cond3[cond3]
#         return Xmax

#     def compute_th_fc_adj(self):
#         # Copy depth to groundwater, and add crop/farm dimension for
#         # convenience
#         zGW_comp = np.broadcast_to(
#             self.var.groundwater.zGW[None,None,None,:],
#             (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell)
#         )

#         zMid = self.compute_mid_point_of_compartments()
#         WTinSoilComp = (zMid >= zGW_comp)
#         self.var.th[WTinSoilComp] = self.var.th_sat_comp[WTinSoilComp]

#         # Flatten WTinSoilComp to provide an array with dimensions
#         # (ncrop, nCell), indicating crops where the water table is
#         # in the soil profile
#         self.var.WTinSoil = np.sum(WTinSoilComp, axis=2).astype(bool)  # CHECK            
#         Xmax = self.compute_Xmax()

#         # Index of the compartment to which each element belongs
#         # (shallow -> deep, i.e. 1 is the shallowest)
#         compartment = (
#             np.arange(1, self.var.nComp + 1)[None,None,:,None]
#             * np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))[:,:,None,:]
#         )

#         # Index of the lowest compartment (i.e. the maximum value) for which
#         # cond4 is met, cast to all compartments (achieved by multiplying
#         # compartments by cond4 to set elements that do not equal the
#         # condition to zero, but retain the compartment number of elements
#         # that do meet the condition
#         cond4 = (zGW_comp < 0) | ((zGW_comp - zMid) >= Xmax)
#         cond4_max_compartment = (
#             np.amax(compartment * cond4, axis=2)[:,:,None,:]
#             * np.ones((self.var.nComp))[None,None,:,None]
#         )

#         # Now, identify compartments that are shallower than the deepest
#         # compartment for which cond4 is met
#         cond4 = (compartment <= cond4_max_compartment)
#         cond5 = (self.var.th_fc_comp >= self.var.th_sat_comp) & np.logical_not(cond4)
#         cond6 = (zMid >= zGW_comp) & np.logical_not(cond4 | cond5)
#         cond7 = np.logical_not(cond4 | cond5 | cond6)
#         dV = self.var.th_sat_comp - self.var.th_fc_comp
#         dFC = (dV / (Xmax ** 2)) * ((zMid - (zGW_comp - Xmax)) ** 2)

#         self.var.th_fc_adj[cond4] = self.var.th_fc_comp[cond4]
#         self.var.th_fc_adj[cond5] = self.var.th_fc_comp[cond5]
#         self.var.th_fc_adj[cond6] = self.var.th_sat_comp[cond6]
#         self.var.th_fc_adj[cond7] = self.var.th_fc_comp[cond7] + dFC[cond7]
        
#     def dynamic(self):
#         # if np.any(self.var.GrowingSeasonDayOne):
#         #     self.reset_initial_conditions()
#         # if self.var.groundwater.WaterTable:
#         #     self.compute_th_fc_adj()
#         # else:
#         #     self.var.WTinSoil = np.full((self.var.nFarm, self.var.nCrop, self.var.nCell), False)
#         #     self.var.th_fc_adj = np.copy(self.var.th_fc_comp)

#         aquacrop_fc.check_gw_table_w.update_check_gw_table_w(
#             np.asfortranarray(self.var.th),
#             np.asfortranarray(self.var.th_fc_adj),
#             np.asfortranarray(self.var.WTinSoil),
#             np.asfortranarray(self.var.th_sat),
#             np.asfortranarray(self.var.th_fc),
#             self.var.groundwater.WaterTable,
#             self.var.groundwater.WaterTable,
#             np.asfortranarray(self.var.groundwater.zGW),
#             np.asfortranarray(self.var.dz),
#             np.asfortranarray(self.var.layerIndex),
#             self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.nCell
#         )
            
