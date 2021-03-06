#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

def fraction_of_compartment_in_root_zone(rootdepth, dz_sum, dz, ncomp):
    rootdepth_comp = (
        rootdepth[:,:,None,:]
        * np.ones((ncomp))[None,None,:,None]
    )
    comp_sto = (np.round((dz_sum - dz) * 1000) < np.round(rootdepth_comp * 1000))
    # Fraction of compartment covered by root zone (zero in compartments
    # NOT covered by the root zone)
    factor = 1 - ((dz_sum - rootdepth_comp) / dz)
    factor = np.clip(factor, 0, 1)
    factor[np.logical_not(comp_sto)] = 0
    return factor

def water_storage_in_root_zone(th, dz, root_factor):
    # Water storages in root zone (mm) - initially compute value in each
    # compartment, then sum to get overall root zone storages
    W_comp = root_factor * 1000 * th * dz
    W = np.sum(W_comp, axis=2)
    return W    

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
        self.dynamic_fortran()
        
    def dynamic_fortran(self):
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
        
    def dynamic_numpy(self):
        """Function to calculate actual and total available water in the 
        root zone at current time step
        """
        # self.fraction_of_compartment_in_root_zone()
        rootdepth = np.maximum(self.var.Zmin, self.var.Zroot)
        rootdepth = np.round(rootdepth * 100) / 100
        root_factor = fraction_of_compartment_in_root_zone(
            rootdepth,
            self.var.dz_sum_xy,
            self.var.dz_xy,
            self.var.nComp
        )

        Wr = water_storage_in_root_zone(self.var.th, self.var.dz_xy, root_factor)
        WrS = water_storage_in_root_zone(self.var.th_sat_comp, self.var.dz_xy, root_factor)
        WrFC = water_storage_in_root_zone(self.var.th_fc_comp, self.var.dz_xy, root_factor)
        WrWP = water_storage_in_root_zone(self.var.th_wilt_comp, self.var.dz_xy, root_factor)
        WrDry = water_storage_in_root_zone(self.var.th_dry_comp, self.var.dz_xy, root_factor)
        
        # Convert depths to m3/m3
        self.var.thRZ_Act = np.divide(Wr, rootdepth * 1000, out=np.zeros_like(Wr), where=rootdepth!=0)
        self.var.thRZ_Sat = np.divide(WrS, rootdepth * 1000, out=np.zeros_like(WrS), where=rootdepth!=0)
        self.var.thRZ_Fc  = np.divide(WrFC, rootdepth * 1000, out=np.zeros_like(WrFC), where=rootdepth!=0)
        self.var.thRZ_Wp  = np.divide(WrWP, rootdepth * 1000, out=np.zeros_like(WrWP), where=rootdepth!=0)
        self.var.thRZ_Dry = np.divide(WrDry, rootdepth * 1000, out=np.zeros_like(WrDry), where=rootdepth!=0)

        # Calculate total available water and root zone depletion
        self.var.TAW = np.clip((WrFC - WrWP), 0, None)
        self.var.Dr = np.clip((WrFC - Wr), 0, None)
        self.var.Wr = np.copy(Wr)

        Aer_comp = np.broadcast_to(self.var.Aer[:,:,None,:], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        WrAer_comp = root_factor * 1000 * (self.var.th_sat_comp - (Aer_comp / 100)) * self.var.dz_xy
        WrAer = np.sum(WrAer_comp, axis=2)
        self.var.thRZ_Aer = np.divide(WrAer, rootdepth * 1000, out=np.zeros_like(WrAer), where=rootdepth!=0)
