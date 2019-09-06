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
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
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
        
        self.var.EsAct = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        EvapTimeSteps = 20
        aquacrop_fc.soil_evaporation_w.soil_evap_w(
            np.asfortranarray(np.float64(self.var.weather.precipitation)),
            np.asfortranarray(np.float64(self.var.weather.referencePotET)),
            np.asfortranarray(self.var.EsAct),
            np.asfortranarray(self.var.Epot),
            np.asfortranarray(self.var.Irr),
            np.asfortranarray(np.int32(self.var.IrrMethod)),
            np.asfortranarray(self.var.Infl),
            np.asfortranarray(self.var.th),
            np.asfortranarray(self.var.th_sat_comp),
            np.asfortranarray(self.var.th_fc_comp),
            np.asfortranarray(self.var.th_wilt_comp),
            np.asfortranarray(self.var.th_dry_comp),
            np.asfortranarray(self.var.SurfaceStorage),
            np.asfortranarray(self.var.WetSurf),
            np.asfortranarray(self.var.Wsurf),
            np.asfortranarray(self.var.Wstage2),
            np.asfortranarray(self.var.CC),
            np.asfortranarray(self.var.CCadj),
            np.asfortranarray(self.var.CCxAct),
            np.asfortranarray(self.var.EvapZ),
            np.asfortranarray(self.var.EvapZmin),
            np.asfortranarray(self.var.EvapZmax),
            np.asfortranarray(self.var.REW),
            np.asfortranarray(self.var.Kex),
            np.asfortranarray(self.var.CCxW),
            np.asfortranarray(self.var.fwcc),
            np.asfortranarray(self.var.fevap),
            np.asfortranarray(self.var.fWrelExp),
            np.asfortranarray(self.var.dz),
            np.asfortranarray(self.var.dz_sum),
            np.asfortranarray(np.int32(self.var.Mulches)),
            np.asfortranarray(self.var.fMulch),
            np.asfortranarray(self.var.MulchPctGS),
            np.asfortranarray(self.var.MulchPctOS),
            np.asfortranarray(np.int32(self.var.GrowingSeasonIndex)),
            np.asfortranarray(np.int32(self.var.Senescence)),
            np.asfortranarray(np.int32(self.var.PrematSenes)),
            np.int32(self.var.CalendarType),
            np.asfortranarray(np.int32(self.var.DAP)),
            np.asfortranarray(np.int32(self.var.DelayedCDs)),
            np.asfortranarray(np.int32(self.var.DelayedGDDs)),
            self.var._modelTime.timeStepPCR,
            EvapTimeSteps,
            self.var.nFarm,
            self.var.nCrop,
            self.var.nComp,
            self.var.nCell)
