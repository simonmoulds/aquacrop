#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

from .io.CarbonDioxide import refconc

import aquacrop_fc

class Transpiration(object):
    def __init__(self, Transpiration_variable):
        self.var = Transpiration_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        arr_ones = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.Ksa_Aer = np.copy(arr_zeros)
        self.var.TrPot0 = np.copy(arr_zeros)
        self.var.TrPot_NS = np.copy(arr_zeros)
        self.var.TrAct = np.copy(arr_zeros)
        self.var.TrAct0 = np.copy(arr_zeros)
        self.var.AgeDays = np.copy(arr_zeros)
        self.var.AgeDays_NS = np.copy(arr_zeros)
        self.var.AerDays = np.copy(arr_zeros)
        self.var.AerDaysComp  = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        self.var.Tpot = np.copy(arr_zeros)        
        self.var.TrRatio = np.copy(arr_ones)
        self.var.DaySubmerged = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        self.var.AgeDays[self.var.GrowingSeasonDayOne] = 0  # not sure if required
        self.var.AgeDays_NS[self.var.GrowingSeasonDayOne] = 0  # not sure if required
        cond = self.var.GrowingSeasonDayOne
        cond_comp = np.broadcast_to(cond[:,:,None,:], self.var.AerDaysComp.shape)
        self.var.AerDays[cond] = 0
        self.var.AerDaysComp[cond_comp] = 0        
        self.var.Tpot[cond] = 0
        self.var.TrRatio[cond] = 1
        # self.var.TrAct[cond] = 0  # TEMP - may not require?
        self.var.DaySubmerged[cond] = 0
        
    def dynamic(self):
        """Function to calculate crop transpiration on current day"""

        # reset initial conditions
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()

        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.transpiration_w.update_transpiration_w(
            self.var.TrPot0.T, 
            self.var.TrPot_NS.T, 
            self.var.TrAct.T,
            self.var.TrAct0.T, 
            self.var.Tpot.T, 
            self.var.TrRatio.T,
            np.int32(self.var.AerDays).T, 
            np.int32(self.var.AerDaysComp).T, 
            self.var.th.T, 
            self.var.thRZ_Act.T, 
            self.var.thRZ_Sat.T, 
            self.var.thRZ_Fc.T,
            self.var.thRZ_Wp.T, 
            self.var.thRZ_Dry.T, 
            self.var.thRZ_Aer.T, 
            self.var.TAW.T, 
            self.var.Dr.T,
            np.int32(self.var.AgeDays).T,
            np.int32(self.var.AgeDays_NS).T,
            np.int32(self.var.DaySubmerged).T,
            self.var.SurfaceStorage.T, 
            self.var.IrrNet.T, 
            self.var.IrrNetCum.T, 
            self.var.CC.T, 
            self.var.model.etref.values.T, 
            # self.var.weather.referencePotET.T, 
            self.var.th_sat.T, 
            self.var.th_fc.T, 
            self.var.th_wilt.T, 
            self.var.th_dry.T, 
            np.int32(self.var.MaxCanopyCD).T, 
            self.var.Kcb.T, 
            self.var.Ksw_StoLin.T, 
            self.var.CCadj.T, 
            self.var.CCadj_NS.T,
            self.var.CCprev.T, 
            self.var.CCxW.T,
            self.var.CCxW_NS.T,
            self.var.Zroot.T,
            self.var.rCor.T,
            self.var.Zmin.T,
            self.var.a_Tr.T,
            self.var.Aer.T,
            self.var.fage.T, 
            np.int32(self.var.LagAer).T, 
            self.var.SxBot.T, 
            self.var.SxTop.T, 
            np.int32(self.var.ETadj).T,
            self.var.p_lo2.T, 
            self.var.p_up2.T, 
            self.var.fshape_w2.T, 
            np.int32(self.var.IrrMethod).T,
            self.var.NetIrrSMT.T,
            self.var.CurrentConc.T, 
            refconc,
            # self.var.RefConc,
            np.int32(self.var.DAP).T,
            np.int32(self.var.DelayedCDs).T,
            self.var.dz.T, 
            self.var.dz_sum.T, 
            np.int32(layer_ix).T, 
            np.int32(self.var.GrowingSeasonIndex).T,
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
            )
