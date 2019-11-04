#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

# class RootDevelopment(object):
#     def __init__(self, RootDevelopment_variable):
#         self.var = RootDevelopment_variable

#     def initial(self):
#         self.var.rCor = np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         self.var.Zroot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))

#     def reset_initial_conditions(self):
#         self.var.rCor[self.var.GrowingSeasonDayOne] = 1
#         self.var.Zroot[self.var.GrowingSeasonDayOne] = self.var.Zmin[self.var.GrowingSeasonDayOne]
        
#     def potential_root_development(self, time_since_germination):
#         Zr = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         Zini = self.var.Zmin * (self.var.PctZmin / 100)
#         t0 = np.round(self.var.Emergence / 2)
#         tmax = np.copy(self.var.MaxRooting)
#         cond1 = (self.var.GrowingSeasonIndex & (time_since_germination >= tmax))
#         Zr[cond1] = self.var.Zmax[cond1]
#         cond2 = (self.var.GrowingSeasonIndex & np.logical_not(cond1) & (time_since_germination <= t0))
#         Zr[cond2] = Zini[cond2]
#         cond3 = (self.var.GrowingSeasonIndex & (np.logical_not(cond1 | cond2)))
#         X_divd = (time_since_germination - t0).astype('float64')
#         X_divs = (tmax - t0).astype('float64')
#         X = np.divide(X_divd, X_divs, out=np.zeros_like(X_divs), where=X_divs!=0)
#         Zr_exp = np.divide(1, self.var.fshape_r, out=np.zeros_like(self.var.fshape_r), where=cond3)
#         Zr_pow = np.power(X, Zr_exp, out=np.zeros_like(Zr), where=cond3)
#         Zr[cond3] = (Zini + (self.var.Zmax - Zini) * Zr_pow)[cond3]
#         cond4 = (self.var.GrowingSeasonIndex & (Zr < self.var.Zmin))
#         Zr[cond4] = self.var.Zmin[cond4]
#         return Zr

#     def compute_rate_of_change_in_root_development(self):
#         ZrPrev = self.potential_root_development(self.var.time_since_germination_previous)
#         Zr = self.potential_root_development(self.var.time_since_germination)
#         dZr = Zr - ZrPrev
#         cond1 = (self.var.GrowingSeasonIndex & (self.var.TrRatio < 0.9999))
#         cond11 = (cond1 & (self.var.fshape_ex >= 0))        
#         dZr[cond11] = (dZr * self.var.TrRatio)[cond11]
#         cond12 = (cond1 & np.logical_not(cond11))
#         fAdj_divd = (np.exp(self.var.TrRatio * self.var.fshape_ex) - 1)
#         fAdj_divs = (np.exp(self.var.fshape_ex) - 1)
#         fAdj = np.divide(fAdj_divd, fAdj_divs, out=np.zeros_like(Zr), where=fAdj_divs!=0)
#         dZr[cond12] = (dZr * fAdj)[cond12]
#         dZr[np.logical_not(self.var.Germination)] = 0
#         return dZr

#     def compute_root_depth(self):
#         """Compute root depth."""
#         dZr = self.compute_rate_of_change_in_root_development()        
#         self.var.Zroot += dZr
#         self.var.Zroot[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        
#     def adjust_root_depth_for_restrictive_layer(self):
#         """Adjust root depth for restrictive soil layer.
        
#         This function adjusts the computed root depth to
#         account for a restrictive soil layer which limits
#         the depth of root expansion.
#         """
#         adjust_for_restrictive_layer = (
#             self.var.GrowingSeasonIndex
#             & (self.var.zRes > 0)
#             & (self.var.Zroot > self.var.zRes)
#         )
#         if np.any(adjust_for_restrictive_layer):
#             self.var.rCor[adjust_for_restrictive_layer] = np.divide(
#                 (2 * (self.var.Zroot / self.var.zRes) * ((self.var.SxTop + self.var.SxBot) / 2) - self.var.SxTop),
#                 self.var.SxBot,
#                 out=np.zeros_like(self.var.Zroot),
#                 where=adjust_for_restrictive_layer
#             )[adjust_for_restrictive_layer]
#             self.var.Zroot[adjust_for_restrictive_layer] = self.var.zRes[adjust_for_restrictive_layer]

#     def adjust_root_depth_for_groundwater(self):
#         """Adjust root depth for groundwater.

#         This function adjusts the computed root depth to
#         account for the presence of groundwater in the root
#         zone.
#         """
#         if self.var.groundwater.WaterTable:
#             zGW = np.broadcast_to(self.var.groundwater.zGW[None,None,:], (self.var.nFarm, self.var.nCrop, self.var.nCell))
#             cond12 = ((zGW > 0) & (self.var.Zroot > zGW))
#             self.var.Zroot[cond12] = np.clip(zGW, self.var.Zmin, None)[cond12]
    
#     def dynamic(self):
#         """Function to calculate root zone expansion"""
#         if np.any(self.var.GrowingSeasonDayOne):
#             self.reset_initial_conditions()

#         self.compute_root_depth()
#         self.adjust_root_depth_for_restrictive_layer()
#         self.adjust_root_depth_for_groundwater()

class RootDevelopment(object):
    def __init__(self, RootDevelopment_variable):
        self.var = RootDevelopment_variable

    def initial(self):
        self.var.rCor = np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.Zroot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))

    def reset_initial_conditions(self):
        self.var.rCor[self.var.GrowingSeasonDayOne] = 1
        self.var.Zroot[self.var.GrowingSeasonDayOne] = self.var.Zmin[self.var.GrowingSeasonDayOne]
        
    # def potential_root_development(self, time_since_germination):
    #     Zr = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
    #     Zini = self.var.Zmin * (self.var.PctZmin / 100)
    #     t0 = np.round(self.var.Emergence / 2)
    #     tmax = np.copy(self.var.MaxRooting)
    #     cond1 = (self.var.GrowingSeasonIndex & (time_since_germination >= tmax))
    #     Zr[cond1] = self.var.Zmax[cond1]
    #     cond2 = (self.var.GrowingSeasonIndex & np.logical_not(cond1) & (time_since_germination <= t0))
    #     Zr[cond2] = Zini[cond2]
    #     cond3 = (self.var.GrowingSeasonIndex & (np.logical_not(cond1 | cond2)))
    #     X_divd = (time_since_germination - t0).astype('float64')
    #     X_divs = (tmax - t0).astype('float64')
    #     X = np.divide(X_divd, X_divs, out=np.zeros_like(X_divs), where=X_divs!=0)
    #     Zr_exp = np.divide(1, self.var.fshape_r, out=np.zeros_like(self.var.fshape_r), where=cond3)
    #     Zr_pow = np.power(X, Zr_exp, out=np.zeros_like(Zr), where=cond3)
    #     Zr[cond3] = (Zini + (self.var.Zmax - Zini) * Zr_pow)[cond3]
    #     cond4 = (self.var.GrowingSeasonIndex & (Zr < self.var.Zmin))
    #     Zr[cond4] = self.var.Zmin[cond4]
    #     return Zr

    # def compute_rate_of_change_in_root_development(self):
    #     ZrPrev = self.potential_root_development(self.var.time_since_germination_previous)
    #     Zr = self.potential_root_development(self.var.time_since_germination)
    #     dZr = Zr - ZrPrev
    #     cond1 = (self.var.GrowingSeasonIndex & (self.var.TrRatio < 0.9999))
    #     cond11 = (cond1 & (self.var.fshape_ex >= 0))        
    #     dZr[cond11] = (dZr * self.var.TrRatio)[cond11]
    #     cond12 = (cond1 & np.logical_not(cond11))
    #     fAdj_divd = (np.exp(self.var.TrRatio * self.var.fshape_ex) - 1)
    #     fAdj_divs = (np.exp(self.var.fshape_ex) - 1)
    #     fAdj = np.divide(fAdj_divd, fAdj_divs, out=np.zeros_like(Zr), where=fAdj_divs!=0)
    #     dZr[cond12] = (dZr * fAdj)[cond12]
    #     dZr[np.logical_not(self.var.Germination)] = 0
    #     return dZr

    # def compute_root_depth(self):
    #     """Compute root depth."""
    #     dZr = self.compute_rate_of_change_in_root_development()        
    #     self.var.Zroot += dZr
    #     self.var.Zroot[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        
    # def adjust_root_depth_for_restrictive_layer(self):
    #     """Adjust root depth for restrictive soil layer.
        
    #     This function adjusts the computed root depth to
    #     account for a restrictive soil layer which limits
    #     the depth of root expansion.
    #     """
    #     adjust_for_restrictive_layer = (
    #         self.var.GrowingSeasonIndex
    #         & (self.var.zRes > 0)
    #         & (self.var.Zroot > self.var.zRes)
    #     )
    #     if np.any(adjust_for_restrictive_layer):
    #         self.var.rCor[adjust_for_restrictive_layer] = np.divide(
    #             (2 * (self.var.Zroot / self.var.zRes) * ((self.var.SxTop + self.var.SxBot) / 2) - self.var.SxTop),
    #             self.var.SxBot,
    #             out=np.zeros_like(self.var.Zroot),
    #             where=adjust_for_restrictive_layer
    #         )[adjust_for_restrictive_layer]
    #         self.var.Zroot[adjust_for_restrictive_layer] = self.var.zRes[adjust_for_restrictive_layer]

    # def adjust_root_depth_for_groundwater(self):
    #     """Adjust root depth for groundwater.

    #     This function adjusts the computed root depth to
    #     account for the presence of groundwater in the root
    #     zone.
    #     """
    #     if self.var.groundwater.WaterTable:
    #         zGW = np.broadcast_to(self.var.groundwater.zGW[None,None,:], (self.var.nFarm, self.var.nCrop, self.var.nCell))
    #         cond12 = ((zGW > 0) & (self.var.Zroot > zGW))
    #         self.var.Zroot[cond12] = np.clip(zGW, self.var.Zmin, None)[cond12]
    
    def dynamic(self):
        # """Function to calculate root zone expansion"""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()

        aquacrop_fc.root_dev_w.update_root_dev_w(
            self.var.Zroot.T, 
            self.var.rCor.T, 
            self.var.Zmin.T, 
            self.var.Zmax.T, 
            self.var.PctZmin.T, 
            self.var.Emergence.T, 
            self.var.MaxRooting.T, 
            self.var.fshape_r.T, 
            self.var.fshape_ex.T, 
            self.var.SxBot.T,
            self.var.SxTop.T,
            self.var.DAP.T,
            self.var.GDD.T,
            self.var.GDDcum.T,
            self.var.DelayedCDs.T,
            self.var.DelayedGDDs.T,
            self.var.TrRatio.T,
            self.var.Germination.T, 
            self.var.zRes.T,
            self.var.groundwater.WaterTable, 
            self.var.groundwater.zGW, 
            self.var.CalendarType, 
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm, self.var.nCrop, self.var.nCell
        )
            
