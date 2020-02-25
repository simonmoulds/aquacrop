#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class SoilEvaporation(object):
    """Class to represent daily soil evaporation"""
    def __init__(self, SoilEvaporation_variable):
        self.var = SoilEvaporation_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.Epot = np.copy(arr_zeros)
        self.var.Stage2 = np.copy(arr_zeros.astype(bool))
        self.var.EvapZ = np.copy(arr_zeros)
        self.var.Wstage2 = np.copy(arr_zeros)
        self.var.Wsurf = np.copy(arr_zeros)
        self.var.Wevap_Act = np.copy(arr_zeros)
        self.var.Wevap_Sat = np.copy(arr_zeros)
        self.var.Wevap_Fc = np.copy(arr_zeros)
        self.var.Wevap_Wp = np.copy(arr_zeros)
        self.var.Wevap_Dry = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        pass

    def dynamic(self):
        
        # thh = np.asfortranarray(np.float64(self.var.th))
        self.var.EsAct = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        EvapTimeSteps = 20
        aquacrop_fc.soil_evaporation_w.update_soil_evap_w(
            np.float64(self.var.model.prec.values).T,
            np.float64(self.var.model.etref.values).T,
            # np.float64(self.var.weather.precipitation).T,
            # np.float64(self.var.weather.referencePotET).T,
            self.var.EsAct.T,
            self.var.Epot.T,
            self.var.Irr.T,
            np.int32(self.var.IrrMethod).T,
            self.var.Infl.T,
            self.var.th.T,
            self.var.th_sat_comp.T,
            self.var.th_fc_comp.T,
            self.var.th_wilt_comp.T,
            self.var.th_dry_comp.T,
            self.var.SurfaceStorage.T,
            self.var.WetSurf.T,
            self.var.Wsurf.T,
            self.var.Wstage2.T,
            self.var.CC.T,
            self.var.CCadj.T,
            self.var.CCxAct.T,
            self.var.EvapZ.T,
            self.var.EvapZmin.T,
            self.var.EvapZmax.T,
            self.var.REW.T,
            self.var.Kex.T,
            self.var.CCxW.T,
            self.var.fwcc.T,
            self.var.fevap.T,
            self.var.fWrelExp.T,
            self.var.dz.T,
            self.var.dz_sum.T,
            np.int32(self.var.Mulches).T,
            self.var.fMulch.T,
            self.var.MulchPctGS.T,
            self.var.MulchPctOS.T,
            np.int32(self.var.GrowingSeasonIndex).T,
            np.int32(self.var.Senescence).T,
            np.int32(self.var.PrematSenes).T,
            np.int32(self.var.CalendarType),
            np.int32(self.var.DAP).T,
            np.int32(self.var.DelayedCDs).T,
            np.int32(self.var.DelayedGDDs).T,
            self.var.model.time.timestep,
            # self.var._modelTime.timeStepPCR,
            EvapTimeSteps,
            self.var.nFarm,
            self.var.nCrop,
            self.var.nComp,
            self.var.domain.nxy)
        # self.var.th = np.ascontiguousarray(thh).copy()
