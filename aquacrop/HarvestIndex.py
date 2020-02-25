#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class HarvestIndex(object):
    def __init__(self, model_object):
        self.var = model_object

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.YieldForm = np.copy(arr_zeros).astype(np.int32)
        self.var.HI = np.copy(arr_zeros)
        self.var.HIt = np.copy(arr_zeros)
        self.var.PctLagPhase = np.copy(arr_zeros)

    def dynamic(self):        
        aquacrop_fc.harvest_index_w.update_harvest_index_w(
            self.var.HI.T, 
            self.var.PctLagPhase.T,
            self.var.YieldForm.T,
            self.var.CCprev.T, 
            self.var.CCmin.T, 
            self.var.CCx.T, 
            self.var.HIini.T, 
            self.var.HI0.T, 
            self.var.HIGC.T, 
            self.var.HIstart.T, 
            self.var.HIstartCD.T, 
            self.var.tLinSwitch.T, 
            self.var.dHILinear.T, 
            self.var.GDDcum.T, 
            self.var.DAP.T, 
            self.var.DelayedCDs.T, 
            self.var.DelayedGDDs.T, 
            self.var.CropType.T, 
            self.var.CalendarType, 
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm, self.var.nCrop, self.var.domain.nxy
            )        
        
class HarvestIndexAdjusted(object):
    def __init__(self, HarvestIndex_variable):
        self.var = HarvestIndex_variable

    def initial(self):
        arr_ones = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.Fpre = np.copy(arr_ones)
        self.var.Fpost = np.copy(arr_ones)
        self.var.fpost_dwn = np.copy(arr_ones)
        self.var.fpost_upp = np.copy(arr_ones)
        self.var.Fpol = np.copy(arr_zeros)
        self.var.sCor1 = np.copy(arr_zeros)
        self.var.sCor2 = np.copy(arr_zeros)
        self.var.HIadj = np.copy(arr_zeros)
        self.var.PreAdj = np.copy(arr_zeros).astype(np.int32)
        
    def dynamic(self):        
        aquacrop_fc.harvest_index_w.adjust_harvest_index_w(
            self.var.HIadj.T,
            self.var.PreAdj.T,
            self.var.Fpre.T, 
            self.var.Fpol.T, 
            self.var.Fpost.T, 
            self.var.fpost_dwn.T, 
            self.var.fpost_upp.T, 
            self.var.sCor1.T, 
            self.var.sCor2.T,
            self.var.YieldForm.T,
            self.var.HI.T, 
            self.var.HI0.T, 
            self.var.dHI0.T, 
            self.var.B.T, 
            self.var.B_NS.T, 
            self.var.dHI_pre.T, 
            self.var.CC.T, 
            self.var.CCmin.T, 
            self.var.Ksw_Exp.T, 
            self.var.Ksw_Sto.T, 
            self.var.Ksw_Pol.T, 
            self.var.Kst_PolC.T, 
            self.var.Kst_PolH.T, 
            self.var.CanopyDevEndCD.T, 
            self.var.HIstartCD.T, 
            self.var.HIendCD.T, 
            self.var.YldFormCD.T, 
            self.var.FloweringCD.T, 
            self.var.a_HI.T, 
            self.var.b_HI.T, 
            self.var.exc.T, 
            self.var.DAP.T, 
            self.var.DelayedCDs.T, 
            self.var.CropType.T, 
            self.var.GrowingSeasonIndex.T, 
            self.var.nFarm, self.var.nCrop, self.var.domain.nxy
        )
